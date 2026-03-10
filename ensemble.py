import asyncio
import logging
import math
import re
from typing import Dict, List, Optional

from llm_calls import ENSEMBLE_MODELS, EXTREMIZATION_ALPHA, call_model

logger = logging.getLogger(__name__)

EXTREMIZATION_ALPHA = 1.2
DEBATE_SPREAD_THRESHOLD = 0.15

COMMUNITY_WEIGHT_DEFAULT = 0.40
COMMUNITY_WEIGHT_ECON    = 0.55

ECON_KEYWORDS = [
    "gdp", "inflation", "unemployment", "interest rate", "fed ", "federal reserve",
    "stock", "market cap", "earnings", "revenue", "recession", "economic", "economy",
    "trade deficit", "currency", "exchange rate", "bond", "yield", "monetary",
    "fiscal", "budget", "debt", "deficit", "oil price", "commodity",
]

###############################################################################
# ENSEMBLE MATH
###############################################################################

def parse_probability(text: str) -> Optional[float]:
    for pattern in [
        r'[Rr]evised\s+[Pp]robability[:\s]+(\d+\.?\d*)\s*%',
        r'[Pp]robability[:\s]+(\d+\.?\d*)\s*%',
        r'\b(\d+\.?\d*)\s*%',
    ]:
        matches = re.findall(pattern, text)
        if matches:
            return max(0.01, min(0.99, float(matches[-1]) / 100))
    return None


def extremize_log_odds(probabilities: List[float], alpha: float = EXTREMIZATION_ALPHA) -> float:
    if not probabilities:
        return 0.5
    clipped   = [max(0.02, min(0.98, p)) for p in probabilities]
    log_odds  = [math.log(p / (1 - p)) for p in clipped]
    avg_lo    = sum(log_odds) / len(log_odds)
    extremized = alpha * avg_lo
    return max(0.01, min(0.99, 1 / (1 + math.exp(-extremized))))


def community_anchored_blend(
    model_pred: float,
    community_pred: Optional[float],
    community_weight: float = COMMUNITY_WEIGHT_DEFAULT,
    apply_post_blend_extremization: bool = False,
) -> float:
    if community_pred is None:
        return model_pred
    blended = (1 - community_weight) * model_pred + community_weight * community_pred
    if apply_post_blend_extremization:
        bc  = max(0.02, min(0.98, blended))
        lo  = math.log(bc / (1 - bc))
        blended = max(0.01, min(0.99, 1 / (1 + math.exp(-1.1 * lo))))
    logger.info(
        f"  Community blend: ensemble={model_pred:.3f}, community={community_pred:.3f} "
        f"→ {blended:.3f} (weight={community_weight}, post_extremize={apply_post_blend_extremization})"
    )
    return max(0.01, min(0.99, blended))


def get_community_weight(question_text: str) -> float:
    if any(kw in question_text.lower() for kw in ECON_KEYWORDS):
        logger.info("  Economics question detected → community_weight=0.55")
        return COMMUNITY_WEIGHT_ECON
    return COMMUNITY_WEIGHT_DEFAULT


###############################################################################
# DEBATE PROMPT STRUCTURES
###############################################################################

_PHASE1_STRUCTURE = """

## Your Forecast (follow this structure exactly — keep each section to 2-4 sentences)

**A. Reference Class & Base Rate**
Name the reference class. What % of similar past situations resolved Yes?
Give a concrete number (e.g. "~30% of similar X situations resolve Yes").
If you cannot find a base rate, say so explicitly — do not skip this section.

**B. Inside View Adjustment**
Name ONE or TWO specific facts from the research brief that shift you from the base rate.
State clearly: does each fact push the probability UP or DOWN, and by roughly how much?

**C. Probability Range**
Before your final number, state a range (e.g. "I estimate 30–50%").
In one sentence: what would push you to the low end? The high end?

**D. Community Prior Check**
State the Metaculus community prediction (if in research above).
If you deviate by more than 10pp, name the specific evidence that justifies it.
If you have no specific counter-evidence, default to within 10pp of the community.

**E. Final Probability**
One line only: "Probability: ZZ%"
"""

_PHASE2_DEBATE_STRUCTURE = """

## Debate Task (keep each section to 2-3 sentences)

**1. Diagnose the disagreement**
What is the single biggest gap between the forecasters above?
Is it: (a) different reference class / base rate, (b) different weight on a specific
piece of evidence, or (c) different interpretation of the community prior?

**2. Steel-man the outlier**
Take the forecast furthest from the median. Write its strongest possible case
using only evidence from the research brief. No dismissals.

**3. What the outlier missed**
Name one specific thing the outlier ignored or underweighted from the research.

**4. Coherence check**
In one sentence: does your Phase 1 reasoning actually justify your Phase 1 number,
or is there a mismatch you need to correct?

**5. Revised forecast**
Range first (e.g. "35–45%"), then: "Revised Probability: ZZ%"
"""


###############################################################################
# DEBATE ENGINE
###############################################################################

async def run_debate_ensemble(
    question_text: str,
    base_prompt: str,
    question_context: str = "",
) -> tuple[float, str, bool]:
    logger.info("=== DEBATE ENSEMBLE: Phase 1 — Independent structured forecasts ===")

    full_phase1_prompt = base_prompt + _PHASE1_STRUCTURE

    phase1_tasks = {
        name: call_model(model_id, full_phase1_prompt)
        for name, model_id in ENSEMBLE_MODELS.items()
    }
    raw1 = await asyncio.gather(*phase1_tasks.values(), return_exceptions=True)
    phase1_results: Dict[str, str] = {
        name: (r if isinstance(r, str) else "")
        for name, r in zip(phase1_tasks.keys(), raw1)
    }

    phase1_probs: Dict[str, Optional[float]] = {}
    for name, text in phase1_results.items():
        prob = parse_probability(text)
        phase1_probs[name] = prob
        logger.info(f"  Phase 1 – {name}: {f'{prob*100:.1f}%' if prob else 'PARSE FAIL'}")

    valid_p1 = {n: p for n, p in phase1_probs.items() if p is not None}
    if not valid_p1:
        logger.error("All Phase 1 calls failed — returning 0.5")
        return 0.5, "All Phase 1 model calls failed.", False

    p1_vals   = list(valid_p1.values())
    p1_spread = max(p1_vals) - min(p1_vals)
    p1_summary = (
        f"Phase 1 range: {min(p1_vals)*100:.0f}% – {max(p1_vals)*100:.0f}%  "
        f"(spread: {p1_spread*100:.0f}pp)"
    )
    logger.info(f"  {p1_summary}")

    phase2_triggered = p1_spread > DEBATE_SPREAD_THRESHOLD
    phase2_results: Dict[str, str] = {}
    final_probs: List[float] = []

    if phase2_triggered:
        logger.info(
            f"=== DEBATE ENSEMBLE: Phase 2 triggered "
            f"(spread {p1_spread*100:.0f}pp > threshold {DEBATE_SPREAD_THRESHOLD*100:.0f}pp) ==="
        )
        _P1_CONTEXT_CAP = 800
        debate_context = "\n\n".join(
            f"### {name} — Phase 1: {f'{p*100:.1f}%' if p else 'PARSE FAIL'}\n"
            f"{text[:_P1_CONTEXT_CAP]}"
            for (name, text), p in zip(phase1_results.items(), phase1_probs.values())
        )
        revision_prompt = (
            f"You are one of five independent forecasters debating this question:\n\n"
            f"**{question_text}**\n\n"
            f"{question_context}\n\n"
            f"**{p1_summary}** — there is significant disagreement. Work out why.\n\n"
            f"## All Phase 1 Forecasts\n\n"
            f"{debate_context}\n"
            + _PHASE2_DEBATE_STRUCTURE
        )
        phase2_tasks = {
            name: call_model(model_id, revision_prompt)
            for name, model_id in ENSEMBLE_MODELS.items()
        }
        raw2 = await asyncio.gather(*phase2_tasks.values(), return_exceptions=True)
        phase2_results = {
            name: (r if isinstance(r, str) else "")
            for name, r in zip(phase2_tasks.keys(), raw2)
        }
        for name, text in phase2_results.items():
            prob = parse_probability(text)
            if prob is not None:
                final_probs.append(prob)
                logger.info(f"  Phase 2 – {name}: {prob*100:.1f}%")
            else:
                fallback = phase1_probs.get(name)
                if fallback is not None:
                    final_probs.append(fallback)
                    logger.info(f"  Phase 2 – {name}: parse fail → P1 fallback {fallback*100:.1f}%")
        if not final_probs:
            logger.warning("Phase 2 produced no valid probs — falling back to Phase 1")
            final_probs = p1_vals
    else:
        logger.info(
            f"=== DEBATE ENSEMBLE: Phase 2 SKIPPED "
            f"(spread {p1_spread*100:.0f}pp ≤ threshold {DEBATE_SPREAD_THRESHOLD*100:.0f}pp) ==="
        )
        final_probs = p1_vals

    final_prob   = extremize_log_odds(final_probs, alpha=EXTREMIZATION_ALPHA)
    final_spread = (max(final_probs) - min(final_probs)) * 100 if len(final_probs) > 1 else 0

    logger.info(
        f"  Ensemble (α={EXTREMIZATION_ALPHA}): {final_prob*100:.1f}%  "
        f"from {[f'{p*100:.1f}%' for p in final_probs]}  "
        f"(Phase 2 {'ran' if phase2_triggered else 'skipped'}, "
        f"final spread: {final_spread:.0f}pp)"
    )

    reasoning = "## Multi-Model Debate Results\n\n"
    reasoning += f"### Phase 1 — Independent Structured Forecasts\n*{p1_summary}*\n"
    for name, prob in phase1_probs.items():
        reasoning += f"- **{name}**: {f'{prob*100:.1f}%' if prob else 'failed'}\n"

    if phase2_triggered:
        reasoning += "\n### Phase 2 — Argumentative Debate (triggered: spread > threshold)\n"
        for name, text in phase2_results.items():
            prob = parse_probability(text)
            reasoning += f"- **{name}**: {f'{prob*100:.1f}%' if prob else 'failed'}\n"
            reasoning += f"  _{text[:500].strip()}_\n\n"
    else:
        reasoning += (
            f"\n### Phase 2 — Skipped\n"
            f"*Spread ({p1_spread*100:.0f}pp) ≤ threshold ({DEBATE_SPREAD_THRESHOLD*100:.0f}pp). "
            f"Models agree — Phase 1 aggregated directly.*\n"
        )

    reasoning += (
        f"\n### Ensemble Aggregation\n"
        f"Input probs: {[f'{p*100:.1f}%' for p in final_probs]}\n"
        f"Phase 2: {'ran' if phase2_triggered else 'skipped'} "
        f"(threshold: {DEBATE_SPREAD_THRESHOLD*100:.0f}pp)\n"
        f"Log-odds extremization (α={EXTREMIZATION_ALPHA}): **{final_prob*100:.1f}%**\n"
    )

    return final_prob, reasoning, phase2_triggered
