import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional

from ensemble import (
    COMMUNITY_WEIGHT_DEFAULT,
    DEBATE_SPREAD_THRESHOLD,
    EXTREMIZATION_ALPHA,
    get_community_weight,
)

logger = logging.getLogger(__name__)

_POSTMORTEM_PATH = "reports/postmortem.jsonl"


def log_prediction(
    question_url: str,
    question_text: str,
    prediction: float,
    community_pred: Optional[float],
    ensemble_pred: float,
    phase2_ran: bool,
    question_type: str = "binary",
    metadata: dict = None,
) -> None:
    os.makedirs(os.path.dirname(_POSTMORTEM_PATH), exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question_url": question_url,
        "question_text": question_text[:200],
        "question_type": question_type,
        "prediction": round(prediction, 4),
        "community_pred": round(community_pred, 4) if community_pred else None,
        "ensemble_pred": round(ensemble_pred, 4),
        "phase2_ran": phase2_ran,
        "alpha": EXTREMIZATION_ALPHA,
        "community_weight": get_community_weight(question_text),
        "debate_threshold": DEBATE_SPREAD_THRESHOLD,
        "resolution": None,
        "brier_score": None,
        **(metadata or {}),
    }
    try:
        with open(_POSTMORTEM_PATH, "a") as f:
            f.write(json.dumps(record) + "\n")
        logger.info(f"  Postmortem logged: {question_url}")
    except Exception as e:
        logger.warning(f"Failed to log postmortem: {e}")


def compute_postmortem_stats(postmortem_path: str = _POSTMORTEM_PATH) -> dict:
    if not os.path.exists(postmortem_path):
        return {"error": "No postmortem file found"}

    records = []
    with open(postmortem_path) as f:
        for line in f:
            try:
                r = json.loads(line.strip())
                if r.get("resolution") is not None and r.get("prediction") is not None:
                    records.append(r)
            except json.JSONDecodeError:
                continue

    if not records:
        return {"error": "No resolved predictions found"}

    brier_scores = [(r["prediction"] - r["resolution"]) ** 2 for r in records]
    avg_brier    = sum(brier_scores) / len(brier_scores)

    by_type = {}
    for r in records:
        qt = r.get("question_type", "binary")
        by_type.setdefault(qt, []).append((r["prediction"] - r["resolution"]) ** 2)
    brier_by_type = {k: sum(v) / len(v) for k, v in by_type.items()}

    community_brier = [
        (r["community_pred"] - r["resolution"]) ** 2
        for r in records if r.get("community_pred") is not None
    ]
    avg_community_brier = sum(community_brier) / len(community_brier) if community_brier else None

    p2_yes = [bs for r, bs in zip(records, brier_scores) if r.get("phase2_ran")]
    p2_no  = [bs for r, bs in zip(records, brier_scores) if not r.get("phase2_ran")]

    bins = {}
    for r in records:
        bin_key = round(r["prediction"] * 10) / 10
        bins.setdefault(bin_key, []).append(r["resolution"])
    calibration = {
        str(k): {"predicted": k, "actual": sum(v) / len(v), "count": len(v)}
        for k, v in sorted(bins.items())
    }

    recommendations = []
    if avg_community_brier is not None:
        if avg_brier > avg_community_brier * 1.1:
            recommendations.append(
                f"Bot Brier ({avg_brier:.4f}) > Community ({avg_community_brier:.4f}). "
                f"Consider increasing COMMUNITY_WEIGHT_DEFAULT above {COMMUNITY_WEIGHT_DEFAULT}."
            )
        elif avg_brier < avg_community_brier * 0.95:
            recommendations.append("Bot outperforming community — current community weight may be too high.")
    if p2_yes and p2_no:
        p2_yes_avg = sum(p2_yes) / len(p2_yes)
        p2_no_avg  = sum(p2_no)  / len(p2_no)
        if p2_yes_avg > p2_no_avg * 1.1:
            recommendations.append(
                f"Phase 2 Brier ({p2_yes_avg:.4f}) worse than Phase 1-only ({p2_no_avg:.4f}). "
                f"Consider raising DEBATE_SPREAD_THRESHOLD."
            )

    return {
        "n_resolved": len(records),
        "avg_brier": round(avg_brier, 4),
        "brier_by_type": brier_by_type,
        "avg_community_brier": round(avg_community_brier, 4) if avg_community_brier else None,
        "phase2_brier": round(sum(p2_yes) / len(p2_yes), 4) if p2_yes else None,
        "no_phase2_brier": round(sum(p2_no) / len(p2_no), 4) if p2_no else None,
        "calibration_bins": calibration,
        "recommendations": recommendations,
    }
