import argparse
import asyncio
import json
import logging
import os
import re
from datetime import datetime
from typing import Literal

import dotenv
dotenv.load_dotenv()

from forecasting_tools import GeneralLlm, MetaculusClient

from bot import DanDanDonkeyBot, _extract_community_float
from ensemble import (
    COMMUNITY_WEIGHT_DEFAULT,
    COMMUNITY_WEIGHT_ECON,
    DEBATE_SPREAD_THRESHOLD,
    ENSEMBLE_MODELS,
    EXTREMIZATION_ALPHA,
)
from llm_calls import MAX_TOKENS_FORECAST, MAX_TOKENS_RESEARCH
from postmortem import compute_postmortem_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("LiteLLM").propagate = False

logger = logging.getLogger(__name__)

EXAMPLE_QUESTIONS = [
    "https://www.metaculus.com/questions/25523/nigel-farage-uk-pm-before-jan-1-2035/",
]


###############################################################################
# SUMMARY REPORT
###############################################################################

def save_summary_report(
    forecast_reports: list,
    output_path: str = "reports/summary.md",
) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    rows = []
    for report in forecast_reports:
        if isinstance(report, Exception):
            continue
        try:
            question  = report.question
            q_short   = question.question_text[:80] + ("..." if len(question.question_text) > 80 else "")
            url       = getattr(question, "page_url", "")
            community_raw = _extract_community_float(question)
            community_str = f"{community_raw*100:.1f}%" if community_raw is not None else "N/A"

            final_pred = None
            reasoning  = ""
            for pred_report in getattr(report, "prediction_reports", []):
                pred = getattr(pred_report, "prediction", None)
                if isinstance(pred, float):
                    final_pred = pred
                reasoning = getattr(pred_report, "reasoning", "") or ""

            if final_pred is None:
                continue

            final_str = f"{final_pred*100:.1f}%"
            delta_str = "N/A"
            if community_raw is not None:
                delta     = (final_pred - community_raw) * 100
                delta_str = f"{delta:+.1f}pp"

            p1_probs, p2_probs = [], []
            p2_ran = "Phase 2 — Skipped" not in reasoning

            for line in reasoning.split("\n"):
                if "Phase 1" in line and "%" in line:
                    p1_probs += [float(m) for m in re.findall(r'(\d+\.?\d*)%', line)]
                if "Phase 2" in line and "%" in line and "Skipped" not in line:
                    p2_probs += [float(m) for m in re.findall(r'(\d+\.?\d*)%', line)]

            spread_match = re.search(r'spread[:\s]+(\d+)pp', reasoning)
            spread_str   = f"{spread_match.group(1)}pp" if spread_match else "N/A"
            p1_str = f"{sum(p1_probs)/len(p1_probs):.1f}%" if p1_probs else "N/A"
            p2_str = f"{sum(p2_probs)/len(p2_probs):.1f}%" if p2_probs else ("skipped" if not p2_ran else "N/A")

            rows.append({
                "question": q_short, "url": url,
                "community": community_str,
                "p1_avg": p1_str, "p1_spread": spread_str,
                "p2_ran": p2_ran, "p2_avg": p2_str,
                "final": final_str, "delta": delta_str,
            })
        except Exception as e:
            logger.warning(f"Could not parse report for summary: {e}")

    p2_count = sum(1 for r in rows if r.get("p2_ran", False))
    p2_rate  = f"{p2_count}/{len(rows)} ({p2_count/len(rows)*100:.0f}%)" if rows else "0/0"
    now      = datetime.now().strftime("%Y-%m-%d %H:%M")

    md = [
        "# DanDanDonkeyBot v9 — Forecast Summary",
        f"*Generated: {now} | Questions: {len(rows)}*",
        f"*Ensemble: {', '.join(ENSEMBLE_MODELS.keys())} | α={EXTREMIZATION_ALPHA}*",
        f"*Community weight: {COMMUNITY_WEIGHT_DEFAULT} (default) / {COMMUNITY_WEIGHT_ECON} (econ) | "
        f"Phase 2 threshold: {DEBATE_SPREAD_THRESHOLD*100:.0f}pp | Phase 2 triggered: {p2_rate}*",
        "",
        "| # | Question | Community | P1 Avg | P1 Spread | P2? | P2 Avg | Final | Δ Community |",
        "|---|----------|-----------|--------|-----------|-----|--------|-------|-------------|",
    ]
    for i, r in enumerate(rows, 1):
        q_cell  = f"[{r['question']}]({r['url']})" if r['url'] else r['question']
        p2_flag = "✓" if r.get("p2_ran") else "–"
        md.append(
            f"| {i} | {q_cell} | {r['community']} | {r['p1_avg']} | "
            f"{r['p1_spread']} | {p2_flag} | {r['p2_avg']} | {r['final']} | {r['delta']} |"
        )

    md += ["", "## Diverged from Community (>5pp)", ""]
    diverged = [
        r for r in rows
        if r["delta"] != "N/A"
        and abs(float(r["delta"].replace("pp", "").replace("+", ""))) > 5
    ]
    if diverged:
        for r in diverged:
            p2_flag = " [debated]" if r.get("p2_ran") else " [no debate]"
            md.append(f"- **{r['delta']}**{p2_flag} — {r['question']}")
    else:
        md.append("- None — all within 5pp of community.")

    stats = compute_postmortem_stats()
    if "error" not in stats:
        md += [
            "", "## Postmortem Stats (resolved questions)", "",
            f"- Resolved: {stats['n_resolved']}",
            f"- Bot Brier: {stats['avg_brier']:.4f}",
            f"- Community Brier: {stats.get('avg_community_brier', 'N/A')}",
            f"- Phase 2 Brier: {stats.get('phase2_brier', 'N/A')}",
            f"- No-Phase-2 Brier: {stats.get('no_phase2_brier', 'N/A')}",
        ]
        if stats.get("recommendations"):
            md += ["", "### Tuning Recommendations"]
            for rec in stats["recommendations"]:
                md.append(f"- {rec}")

    with open(output_path, "w") as f:
        f.write("\n".join(md))
    logger.info(f"Summary report saved to {output_path}")


###############################################################################
# ENTRYPOINT
###############################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DanDanDonkeyBot v9")
    parser.add_argument(
        "--mode",
        choices=["tournament", "metaculus_cup", "test_questions", "postmortem"],
        default="tournament",
    )
    args = parser.parse_args()

    if args.mode == "postmortem":
        stats = compute_postmortem_stats()
        print(json.dumps(stats, indent=2))
        exit(0)

    run_mode: Literal["tournament", "metaculus_cup", "test_questions"] = args.mode

    danbot = DanDanDonkeyBot(
        research_reports_per_question=1,
        predictions_per_research_report=3,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=True,
        folder_to_save_reports_to="reports",
        skip_previously_forecasted_questions=True,
        extra_metadata_in_explanation=True,
        llms={
            "researcher": GeneralLlm(
                model="openrouter/deepseek/deepseek-r1",
                temperature=0.2, timeout=120, allowed_tries=2,
                max_tokens=MAX_TOKENS_RESEARCH,
            ),
            "default": GeneralLlm(
                model="openrouter/deepseek/deepseek-r1",
                temperature=0.3, timeout=240, allowed_tries=2,
                max_tokens=MAX_TOKENS_FORECAST,
            ),
            "summarizer": GeneralLlm(
                model="openrouter/deepseek/deepseek-r1",
                temperature=0.2, timeout=120, allowed_tries=2,
            ),
            "parser": GeneralLlm(
                model="openrouter/deepseek/deepseek-r1",
                temperature=0.1, timeout=120, allowed_tries=2,
            ),
        },
    )

    client = MetaculusClient()

    if run_mode == "tournament":
        reports1 = asyncio.run(danbot.forecast_on_tournament(
            client.CURRENT_AI_COMPETITION_ID, return_exceptions=True))
        reports2 = asyncio.run(danbot.forecast_on_tournament(
            client.CURRENT_MINIBENCH_ID, return_exceptions=True))
        forecast_reports = reports1 + reports2

    elif run_mode == "metaculus_cup":
        danbot.skip_previously_forecasted_questions = False
        forecast_reports = asyncio.run(danbot.forecast_on_tournament(
            client.CURRENT_METACULUS_CUP_ID, return_exceptions=True))

    elif run_mode == "test_questions":
        danbot.skip_previously_forecasted_questions = False
        questions = [client.get_question_by_url(url) for url in EXAMPLE_QUESTIONS]
        forecast_reports = asyncio.run(danbot.forecast_questions(questions, return_exceptions=True))

    danbot.log_report_summary(forecast_reports)
    save_summary_report(forecast_reports)
