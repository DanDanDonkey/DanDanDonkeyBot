import argparse
import asyncio
import logging
import math
import re
from datetime import datetime, timezone
from typing import Dict, List, Literal, Optional

import dotenv
import json
import os

from forecasting_tools import (
    AskNewsSearcher,
    BinaryQuestion,
    ForecastBot,
    GeneralLlm,
    MetaculusClient,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    DateQuestion,
    DatePercentile,
    Percentile,
    ConditionalQuestion,
    ConditionalPrediction,
    PredictionAffirmed,
    BinaryPrediction,
    PredictedOptionList,
    ReasonedPrediction,
    SmartSearcher,
    clean_indents,
    structure_output,
)

dotenv.load_dotenv()
logger = logging.getLogger(__name__)

METACULUS_OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY")

###############################################################################
# ENSEMBLE CONFIG
###############################################################################

ENSEMBLE_MODELS = {
    "DeepSeek-R1":     "openrouter/deepseek/deepseek-r1",
    "DeepSeek-V3":     "openrouter/deepseek/deepseek-chat",
    "Qwen3-235B":      "openrouter/qwen/qwen3-235b-a22b",
    "Llama4-Maverick": "openrouter/meta-llama/llama-4-maverick",
    "Mistral-Small":   "openrouter/mistralai/mistral-small-3.1-24b-instruct",
}

EXTREMIZATION_ALPHA = 1.2

COMMUNITY_WEIGHT_DEFAULT = 0.40
COMMUNITY_WEIGHT_ECON    = 0.55

DEBATE_SPREAD_THRESHOLD = 0.15

MAX_TOKENS_RESEARCH  = 1500
MAX_TOKENS_FORECAST  = 1500

_SRC_CAP_ASKNEWS   = 2500
_SRC_CAP_DEFAULT   = 600
_EXTERNAL_DATA_CAP = 6000

ECON_KEYWORDS = [
    "gdp", "inflation", "unemployment", "interest rate", "fed ", "federal reserve",
    "stock", "market cap", "earnings", "revenue", "recession", "economic", "economy",
    "trade deficit", "currency", "exchange rate", "bond", "yield", "monetary",
    "fiscal", "budget", "debt", "deficit", "oil price", "commodity",
]

def _get_community_weight(question_text: str) -> float:
    q_lower = question_text.lower()
    if any(kw in q_lower for kw in ECON_KEYWORDS):
        logger.info("  Economics question detected → community_weight=0.55")
        return COMMUNITY_WEIGHT_ECON
    return COMMUNITY_WEIGHT_DEFAULT


###############################################################################
# POSTMORTEM / BRIER SCORE TRACKING
###############################################################################

_POSTMORTEM_PATH = "reports/postmortem.jsonl"


def _log_prediction(question_url: str, question_text: str, prediction: float,
                    community_pred: Optional[float], ensemble_pred: float,
                    phase2_ran: bool, question_type: str = "binary",
                    metadata: dict = None) -> None:
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
        "community_weight": _get_community_weight(question_text),
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
    avg_brier = sum(brier_scores) / len(brier_scores)

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
            recommendations.append(
                f"Bot outperforming community — current community weight may be too high."
            )
    if p2_yes and p2_no:
        p2_yes_avg = sum(p2_yes) / len(p2_yes)
        p2_no_avg  = sum(p2_no) / len(p2_no)
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


###############################################################################
# ENSEMBLE MATH
###############################################################################

def _parse_probability(text: str) -> Optional[float]:
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
    clipped = [max(0.02, min(0.98, p)) for p in probabilities]
    log_odds = [math.log(p / (1 - p)) for p in clipped]
    avg_log_odds = sum(log_odds) / len(log_odds)
    extremized = alpha * avg_log_odds
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
        blended_clipped = max(0.02, min(0.98, blended))
        lo = math.log(blended_clipped / (1 - blended_clipped))
        blended = max(0.01, min(0.99, 1 / (1 + math.exp(-1.1 * lo))))

    logger.info(
        f"  Community blend: ensemble={model_pred:.3f}, community={community_pred:.3f} "
        f"→ {blended:.3f} (weight={community_weight}, post_extremize={apply_post_blend_extremization})"
    )
    return max(0.01, min(0.99, blended))


def _extract_community_float(question: MetaculusQuestion) -> Optional[float]:
    try:
        cp = question.community_prediction
        if isinstance(cp, float):
            return max(0.02, min(0.98, cp))
    except Exception:
        pass
    return None


###############################################################################
# DATA SOURCE HELPERS
###############################################################################

def _safe_get(url: str, timeout: int = 12, params: dict = None) -> dict | list | None:
    import requests
    try:
        resp = requests.get(url, params=params, timeout=timeout)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        logger.warning(f"HTTP GET failed for {url[:80]}: {e}")
    return None


async def fetch_asknews(query: str) -> str:
    try:
        from asknews_sdk import AskNewsSDK
        client_id     = os.environ.get("ASKNEWS_CLIENT_ID", "")
        client_secret = os.environ.get("ASKNEWS_CLIENT_SECRET", "")
        api_key       = os.environ.get("ASKNEWS_API_KEY", "")

        if api_key:
            ask = AskNewsSDK(api_key=api_key)
        elif client_id and client_secret:
            ask = AskNewsSDK(client_id=client_id, client_secret=client_secret)
        else:
            logger.warning("AskNews: no credentials in env. Set ASKNEWS_API_KEY or "
                           "ASKNEWS_CLIENT_ID + ASKNEWS_CLIENT_SECRET.")
            return ""

        result = ask.news.search_news(
            query=query[:200], n_articles=10,
            return_type="string", strategy="latest news",
        )
        context = result.as_string if hasattr(result, "as_string") else str(result)
        return f"**AskNews (Real-time):**\n{context[:2000]}"
    except Exception as e:
        logger.warning(f"AskNews fetch failed: {e}")
        return ""


def fetch_arxiv(query_text: str) -> str:
    arxiv_kws = [
        "ai", "artificial intelligence", "machine learning", "model", "algorithm",
        "study", "research", "trial", "paper", "published", "evidence",
        "science", "technology", "climate", "nuclear", "quantum", "breakthrough",
        "discovered", "drug", "vaccine", "disease", "cancer", "gene", "protein",
        "physics", "chemistry", "biology", "engineering",
    ]
    if not any(kw in query_text.lower() for kw in arxiv_kws):
        return ""
    try:
        import arxiv as arxiv_lib
        client = arxiv_lib.Client()
        search = arxiv_lib.Search(
            query=query_text[:120], max_results=5,
            sort_by=arxiv_lib.SortCriterion.SubmittedDate,
        )
        results = list(client.results(search))
        if not results:
            return ""
        lines = [
            f"  - ({r.published.year if r.published else '?'}) "
            f"**{r.title}**: {(r.summary or '')[:200].replace(chr(10), ' ')}..."
            for r in results
        ]
        return "**Academic Preprints (arXiv):**\n" + "\n".join(lines)
    except Exception as e:
        logger.warning(f"arXiv fetch failed: {e}")
        return ""


def fetch_semantic_scholar(query_text: str) -> str:
    academic_kws = [
        "study", "research", "trial", "paper", "published", "evidence",
        "science", "technology", "ai", "model", "disease", "drug", "vaccine",
        "climate", "nuclear", "quantum", "breakthrough", "discovered",
    ]
    if not any(kw in query_text.lower() for kw in academic_kws):
        return ""
    try:
        keywords = " ".join(query_text.split()[:8])
        data = _safe_get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={"query": keywords, "limit": 5,
                    "fields": "title,year,abstract,citationCount"},
            timeout=12,
        )
        if not data or not data.get("data"):
            return ""
        lines = [
            f"  - ({p.get('year','?')}, {p.get('citationCount',0)} cites) "
            f"**{p.get('title','?')}**: {(p.get('abstract') or '')[:200]}..."
            for p in data["data"][:5]
        ]
        return "**Academic Literature (Semantic Scholar):**\n" + "\n".join(lines)
    except Exception as e:
        logger.warning(f"Semantic Scholar fetch failed: {e}")
        return ""


def fetch_gdelt_articles(query_text: str, max_articles: int = 8) -> str:
    import requests
    try:
        clean_query = query_text[:80].replace('"', '').replace("'", "")
        resp = requests.get(
            "https://api.gdeltproject.org/api/v2/doc/doc",
            params={"query": clean_query, "mode": "ArtList",
                    "maxrecords": str(max_articles), "format": "json", "sort": "DateDesc"},
            timeout=15,
        )
        if resp.status_code != 200:
            return ""
        ct = resp.headers.get("Content-Type", "")
        if "json" not in ct and "javascript" not in ct:
            return ""
        articles = resp.json().get("articles", [])
        if not articles:
            return ""
        lines = [
            f"  - {a.get('title','No title')} ({a.get('domain','?')}, {a.get('seendate','?')[:10]})"
            for a in articles[:max_articles]
        ]
        return "**Recent News (GDELT):**\n" + "\n".join(lines)
    except Exception as e:
        logger.warning(f"GDELT fetch failed: {e}")
        return ""


def fetch_polymarket_data(query_text: str) -> str:
    try:
        data = None
        for n_words in [5, 3]:
            keywords = " ".join(query_text.split()[:n_words])
            data = _safe_get(
                "https://gamma-api.polymarket.com/markets",
                params={"_limit": "5", "closed": "false", "order": "volume24hr",
                        "ascending": "false", "title_contains": keywords},
            )
            if data and isinstance(data, list) and len(data):
                break
        if not data or not isinstance(data, list):
            return ""
        lines = []
        for m in data[:5]:
            title      = m.get("question", m.get("title", "Unknown"))
            prices_raw = m.get("outcomePrices", "")
            outcomes   = m.get("outcomes", "")
            vol        = m.get("volume24hr", 0)
            try:
                prices   = json.loads(prices_raw) if isinstance(prices_raw, str) else prices_raw
                outcomes = json.loads(outcomes) if isinstance(outcomes, str) else outcomes
                price_str = ", ".join(f"{o}: {float(p)*100:.0f}%" for o, p in zip(outcomes, prices))
            except Exception:
                price_str = str(prices_raw)
            lines.append(f"  - {title} → {price_str} (24h vol: ${float(vol):,.0f})")
        return "**Prediction Markets (Polymarket):**\n" + "\n".join(lines)
    except Exception as e:
        logger.warning(f"Polymarket fetch failed: {e}")
        return ""


def fetch_metaculus_community(question: MetaculusQuestion) -> str:
    try:
        cp = question.community_prediction
        if cp is not None:
            val = f"{cp*100:.1f}%" if isinstance(cp, float) else str(cp)
            return f"**Metaculus Community Prediction:** {val}  ← STRONG PRIOR from many calibrated forecasters"
    except Exception:
        pass
    return ""


def fetch_reddit_context(query: str, subreddits: list = None) -> str:
    try:
        import requests
        subreddits = subreddits or ["worldnews", "geopolitics", "ukpolitics", "economics"]
        results = []
        for sub in subreddits[:3]:
            r = requests.get(
                f"https://www.reddit.com/r/{sub}/search.json",
                params={"q": query[:100], "sort": "relevance", "limit": 5, "t": "month"},
                headers={"User-Agent": "DanDanDonkeyBot/1.0"},
                timeout=10,
            )
            if r.status_code == 200:
                for p in r.json().get("data", {}).get("children", [])[:3]:
                    d = p["data"]
                    results.append(f"- [{d['subreddit']}] {d['title']} (score: {d['score']})")
        return ("## Reddit Discussion\n" + "\n".join(results)) if results else ""
    except Exception:
        return ""


def fetch_fred_data(query_text: str) -> str:
    api_key = os.environ.get("FRED_API_KEY", "")
    if not api_key:
        return ""
    try:
        data = _safe_get(
            "https://api.stlouisfed.org/fred/series/search",
            params={"search_text": query_text[:60], "api_key": api_key,
                    "file_type": "json", "limit": "3",
                    "order_by": "popularity", "sort_order": "desc"},
        )
        if not data or "seriess" not in data:
            return ""
        lines = []
        for series in data["seriess"][:3]:
            sid = series.get("id", "")
            obs = _safe_get(
                "https://api.stlouisfed.org/fred/series/observations",
                params={"series_id": sid, "api_key": api_key,
                        "file_type": "json", "sort_order": "desc", "limit": "1"},
            )
            if obs and obs.get("observations"):
                o = obs["observations"][0]
                lines.append(f"  - {series.get('title','')} ({sid}): {o.get('value','N/A')} ({o.get('date','?')})")
        return ("**Economic Data (FRED):**\n" + "\n".join(lines)) if lines else ""
    except Exception as e:
        logger.warning(f"FRED fetch failed: {e}")
        return ""


def fetch_world_bank_data(query_text: str) -> str:
    keyword_to_indicators = {
        "gdp": "NY.GDP.MKTP.CD",         "population": "SP.POP.TOTL",
        "inflation": "FP.CPI.TOTL.ZG",   "unemployment": "SL.UEM.TOTL.ZS",
        "poverty": "SI.POV.DDAY",         "life expectancy": "SP.DYN.LE00.IN",
        "co2": "EN.ATM.CO2E.PC",          "emissions": "EN.ATM.CO2E.PC",
        "trade": "NE.TRD.GNFS.ZS",        "debt": "GC.DOD.TOTL.GD.ZS",
        "education": "SE.ADT.LITR.ZS",    "mortality": "SP.DYN.IMRT.IN",
        "energy": "EG.USE.PCAP.KG.OE",    "renewable": "EG.FEC.RNEW.ZS",
    }
    matched = [(k, v) for k, v in keyword_to_indicators.items() if k in query_text.lower()]
    if not matched:
        return ""
    try:
        lines = []
        for keyword, indicator in matched[:2]:
            data = _safe_get(
                f"https://api.worldbank.org/v2/country/WLD/indicator/{indicator}",
                params={"format": "json", "per_page": "1", "mrv": "1"},
            )
            if data and isinstance(data, list) and len(data) > 1 and data[1]:
                rec = data[1][0]
                if rec.get("value") is not None:
                    lines.append(
                        f"  - {rec.get('indicator',{}).get('value',keyword)}: "
                        f"{rec['value']:,.2f} ({rec.get('date','?')})"
                    )
        return ("**Global Data (World Bank):**\n" + "\n".join(lines)) if lines else ""
    except Exception as e:
        logger.warning(f"World Bank fetch failed: {e}")
        return ""


def fetch_wikipedia_context(query_text: str) -> str:
    try:
        keywords = " ".join(query_text.split()[:6])
        search = _safe_get(
            "https://en.wikipedia.org/w/api.php",
            params={"action": "query", "list": "search", "srsearch": keywords,
                    "srlimit": "1", "format": "json"},
        )
        if not search or not search.get("query", {}).get("search"):
            return ""
        title = search["query"]["search"][0]["title"]
        extract_data = _safe_get(
            "https://en.wikipedia.org/w/api.php",
            params={"action": "query", "titles": title, "prop": "extracts",
                    "exintro": "true", "explaintext": "true", "format": "json"},
        )
        if not extract_data:
            return ""
        for _, page in extract_data.get("query", {}).get("pages", {}).items():
            extract = page.get("extract", "")
            if extract:
                return f"**Background (Wikipedia – {title}):**\n  {extract[:500]}{'...' if len(extract)>500 else ''}"
    except Exception as e:
        logger.warning(f"Wikipedia fetch failed: {e}")
    return ""


def fetch_who_health_data(query_text: str) -> str:
    health_kws = [
        "disease", "health", "pandemic", "virus", "vaccine", "mortality",
        "covid", "infection", "epidemic", "WHO", "measles", "malaria",
        "tuberculosis", "hiv", "flu", "influenza", "bird flu", "mpox",
        "ebola", "cholera", "polio", "death rate", "life expectancy",
    ]
    if not any(kw in query_text.lower() for kw in health_kws):
        return ""
    try:
        keywords = " ".join(query_text.split()[:4])
        data = _safe_get(
            "https://ghoapi.azureedge.net/api/Indicator",
            params={"$filter": f"contains(IndicatorName,'{keywords}')"},
        )
        if not data or not data.get("value"):
            return ""
        lines = []
        for ind in data["value"][:2]:
            code, name = ind.get("IndicatorCode", ""), ind.get("IndicatorName", "")
            obs = _safe_get(
                f"https://ghoapi.azureedge.net/api/{code}",
                params={"$filter": "SpatialDim eq 'GLOBAL'", "$top": "1", "$orderby": "TimeDim desc"},
            )
            if obs and obs.get("value"):
                o = obs["value"][0]
                lines.append(f"  - {name}: {o.get('NumericValue','N/A')} ({o.get('TimeDim','?')})")
            else:
                lines.append(f"  - {name} ({code})")
        return ("**Global Health (WHO GHO):**\n" + "\n".join(lines)) if lines else ""
    except Exception as e:
        logger.warning(f"WHO GHO fetch failed: {e}")
        return ""


def fetch_media_trend(query_text: str) -> str:
    import requests
    try:
        resp = requests.get(
            "https://api.gdeltproject.org/api/v2/doc/doc",
            params={"query": query_text[:60].replace('"', ''),
                    "mode": "TimelineVol", "format": "json", "TIMESPAN": "6m"},
            timeout=12,
        )
        if resp.status_code != 200 or "json" not in resp.headers.get("Content-Type", ""):
            return ""
        series = resp.json().get("timeline", [{}])[0].get("data", [])
        if len(series) < 4:
            return ""
        recent = sum(d.get("value", 0) for d in series[-6:]) / max(len(series[-6:]), 1)
        older  = sum(d.get("value", 0) for d in series[:6])  / max(len(series[:6]),  1)
        if older > 0:
            change    = (recent - older) / older * 100
            direction = "increasing" if change > 10 else "decreasing" if change < -10 else "stable"
            return f"**Media Attention Trend:** {direction} ({change:+.0f}% over 6 months)"
    except Exception as e:
        logger.warning(f"Media trend fetch failed: {e}")
    return ""


def fetch_sec_filings(query_text: str) -> str:
    fin_kws = [
        "stock", "revenue", "earnings", "company", "shares", "market cap",
        "ipo", "nasdaq", "nyse", "profit", "loss", "quarterly", "10-k", "8-k",
    ]
    if not any(kw in query_text.lower() for kw in fin_kws):
        return ""
    try:
        keywords = " ".join(query_text.split()[:5])
        data = _safe_get(
            "https://efts.sec.gov/LATEST/search-index",
            params={"q": keywords, "forms": "10-K,8-K,10-Q",
                    "dateRange": "custom", "startdt": "2024-01-01"},
            timeout=10,
        )
        if not data:
            return ""
        hits = data.get("hits", {}).get("hits", [])
        if not hits:
            return ""
        lines = [
            f"  - {h.get('_source',{}).get('display_date_filed','?')}: "
            f"{h.get('_source',{}).get('entity_name','?')} — "
            f"{h.get('_source',{}).get('file_type','?')}"
            for h in hits[:3]
        ]
        return ("**SEC Filings (EDGAR):**\n" + "\n".join(lines)) if lines else ""
    except Exception as e:
        logger.warning(f"SEC filings fetch failed: {e}")
        return ""


###############################################################################
# MULTI-MODEL DEBATE ENGINE
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


async def _call_model(
    model_id: str,
    prompt: str,
    temperature: float = 0.3,
) -> str:
    try:
        llm = GeneralLlm(
            model=model_id,
            temperature=temperature,
            timeout=180,
            allowed_tries=2,
        )
        return await llm.invoke(prompt)
    except Exception as e:
        logger.warning(f"Model {model_id} failed: {e}")
        return ""


async def run_debate_ensemble(
    question_text: str,
    base_prompt: str,
    question_context: str = "",
) -> tuple[float, str, bool]:
    logger.info("=== DEBATE ENSEMBLE: Phase 1 — Independent structured forecasts ===")

    full_phase1_prompt = base_prompt + _PHASE1_STRUCTURE

    phase1_tasks = {
        name: _call_model(model_id, full_phase1_prompt)
        for name, model_id in ENSEMBLE_MODELS.items()
    }
    raw1 = await asyncio.gather(*phase1_tasks.values(), return_exceptions=True)
    phase1_results: Dict[str, str] = {
        name: (r if isinstance(r, str) else "")
        for name, r in zip(phase1_tasks.keys(), raw1)
    }

    phase1_probs: Dict[str, Optional[float]] = {}
    for name, text in phase1_results.items():
        prob = _parse_probability(text)
        phase1_probs[name] = prob
        logger.info(f"  Phase 1 – {name}: {f'{prob*100:.1f}%' if prob else 'PARSE FAIL'}")

    valid_p1 = {n: p for n, p in phase1_probs.items() if p is not None}
    if not valid_p1:
        logger.error("All Phase 1 calls failed — returning 0.5")
        return 0.5, "All Phase 1 model calls failed.", False

    p1_vals   = list(valid_p1.values())
    p1_min    = min(p1_vals)
    p1_max    = max(p1_vals)
    p1_spread = p1_max - p1_min

    p1_summary = (
        f"Phase 1 range: {p1_min*100:.0f}% – {p1_max*100:.0f}%  "
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
            name: _call_model(model_id, revision_prompt)
            for name, model_id in ENSEMBLE_MODELS.items()
        }
        raw2 = await asyncio.gather(*phase2_tasks.values(), return_exceptions=True)
        phase2_results = {
            name: (r if isinstance(r, str) else "")
            for name, r in zip(phase2_tasks.keys(), raw2)
        }

        for name, text in phase2_results.items():
            prob = _parse_probability(text)
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

    final_prob = extremize_log_odds(final_probs, alpha=EXTREMIZATION_ALPHA)
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
            prob = _parse_probability(text)
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


###############################################################################
# 3-MODEL ENSEMBLE FOR NUMERIC/DATE/MC
###############################################################################

ENSEMBLE_NUMERIC_MODELS = {
    "DeepSeek-R1":     "openrouter/deepseek/deepseek-r1",
    "Qwen3-235B":      "openrouter/qwen/qwen3-235b-a22b",
    "Llama4-Maverick": "openrouter/meta-llama/llama-4-maverick",
}


async def _run_numeric_ensemble(prompt: str) -> str:
    tasks = {
        name: _call_model(model_id, prompt)
        for name, model_id in ENSEMBLE_NUMERIC_MODELS.items()
    }
    raw = await asyncio.gather(*tasks.values(), return_exceptions=True)
    results = {
        name: (r if isinstance(r, str) else "")
        for name, r in zip(tasks.keys(), raw)
    }

    valid = {n: r for n, r in results.items() if r.strip()}
    if not valid:
        return ""
    if len(valid) == 1:
        return list(valid.values())[0]

    def _count_percentile_lines(text: str) -> int:
        return len(re.findall(r'[Pp]ercentile\s+\d+', text))

    scored = [(name, text, _count_percentile_lines(text)) for name, text in valid.items()]
    scored.sort(key=lambda x: x[2], reverse=True)
    best_name, best_text, best_score = scored[0]
    logger.info(
        f"  Numeric ensemble: {len(valid)}/{len(ENSEMBLE_NUMERIC_MODELS)} responded. "
        f"Selected {best_name} (structure score: {best_score})"
    )
    return best_text


###############################################################################
# MAIN BOT CLASS
###############################################################################

class DanDanDonkeyBot(ForecastBot):

    _max_concurrent_questions = 1
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)
    _structure_output_validation_samples = 2

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            q_text = question.question_text

            asknews_raw = await fetch_asknews(q_text)
            asknews_data = asknews_raw[:_SRC_CAP_ASKNEWS] if asknews_raw else ""

            loop = asyncio.get_event_loop()
            def _gather_sync():
                raw = [
                    fetch_metaculus_community(question),
                    fetch_polymarket_data(q_text),
                    fetch_gdelt_articles(q_text),
                    fetch_fred_data(q_text),
                    fetch_world_bank_data(q_text),
                    fetch_who_health_data(q_text),
                    fetch_wikipedia_context(q_text),
                    fetch_media_trend(q_text),
                    fetch_sec_filings(q_text),
                    fetch_semantic_scholar(q_text),
                    fetch_arxiv(q_text),
                    fetch_reddit_context(q_text),
                ]
                return [s[:_SRC_CAP_DEFAULT] for s in raw if s]

            sync_sources = await loop.run_in_executor(None, _gather_sync)

            raw_external = "\n\n".join([asknews_data] + sync_sources)
            external_data = (
                raw_external[:_EXTERNAL_DATA_CAP] + "\n[...truncated for length]"
                if len(raw_external) > _EXTERNAL_DATA_CAP
                else raw_external
            ) or "(No external data retrieved.)"

            logger.info(f"  External data: {len(external_data)} chars from "
                        f"{1 + len(sync_sources)} sources")

            research_prompt = clean_indents(f"""
                You are a research assistant to a professional superforecaster.
                Produce a CONCISE structured brief — aim for 400–600 words total.

                ## Question
                {question.question_text}

                ## Resolution Criteria
                {question.resolution_criteria}

                {question.fine_print}

                ## External Data
                {external_data}

                ## Notes
                - Community Prediction is a STRONG PRIOR from calibrated forecasters — address it.
                - For economics questions, weight FRED/World Bank data heavily.

                ## Output (4 short sections, no padding)

                ### OUTSIDE VIEW
                Base rate, reference class, community prediction, analogous cases.

                ### INSIDE VIEW
                Specific recent evidence from the data above. Cite sources briefly.

                ### KEY UNCERTAINTIES
                2–3 biggest unknowns only.

                ### SYNTHESIS
                Qualitative range only (e.g. "likely Yes, ~60–75%"). No exact number.
            """)

            researcher = self.get_llm("researcher")
            if isinstance(researcher, GeneralLlm):
                research = await researcher.invoke(research_prompt)
            elif isinstance(researcher, str) and researcher.startswith("asknews/"):
                research = await AskNewsSearcher().call_preconfigured_version(
                    researcher, research_prompt
                )
            elif isinstance(researcher, str) and researcher.startswith("smart-searcher"):
                searcher = SmartSearcher(
                    model=researcher.removeprefix("smart-searcher/"),
                    temperature=0, num_searches_to_run=2,
                    num_sites_per_search=10, use_advanced_filters=False,
                )
                research = await searcher.invoke(research_prompt)
            elif not researcher or researcher in ("None", "no_research"):
                research = external_data
            else:
                research = await GeneralLlm(
                    model=researcher,
                    temperature=0.2, timeout=120, allowed_tries=2,
                    max_tokens=MAX_TOKENS_RESEARCH,
                ).invoke(research_prompt)

            logger.info(f"Research complete for {question.page_url}")
            return research

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:

        base_prompt = clean_indents(f"""
            You are a professional superforecaster.

            ## Question
            {question.question_text}

            ## Background
            {question.background_info}

            ## Resolution Criteria
            {question.resolution_criteria}

            {question.fine_print}

            ## Research Brief
            {research}

            ## Today's Date
            {datetime.now().strftime("%Y-%m-%d")}

            Before giving your probability, write:
            (a) Time remaining until resolution.
            (b) Status quo — what happens if nothing changes?
            (c) OUTSIDE VIEW — base rate. What % of similar situations historically resolved Yes?
            (d) INSIDE VIEW — what specific current evidence adjusts this up or down?
            (e) Brief No scenario.
            (f) Brief Yes scenario.
            (g) Metaculus community prediction (if in research) — state it, agree/disagree, WHY.
                Unless you have strong specific contrary evidence, stay within ~10pp of it.

            {self._get_conditional_disclaimer_if_necessary(question)}

            End with: "Probability: ZZ%"
        """)

        question_context = clean_indents(f"""
            Background: {question.background_info[:500]}
            Resolution: {question.resolution_criteria[:300]}
        """)

        ensemble_prob, debate_reasoning, phase2_ran = await run_debate_ensemble(
            question_text=question.question_text,
            base_prompt=base_prompt,
            question_context=question_context,
        )

        community_pred = _extract_community_float(question)
        community_weight = _get_community_weight(question.question_text)
        is_econ = community_weight > COMMUNITY_WEIGHT_DEFAULT

        final_prob = community_anchored_blend(
            ensemble_prob, community_pred, community_weight,
            apply_post_blend_extremization=is_econ,
        )

        full_reasoning = (
            debate_reasoning
            + f"\n\n### Community Blend\n"
            + f"Ensemble: {ensemble_prob*100:.1f}%  |  "
            + f"Community: {f'{community_pred*100:.1f}%' if community_pred else 'N/A'}  |  "
            + f"Final (weight={community_weight}"
            + f"{', post-extremized' if is_econ else ''}): **{final_prob*100:.1f}%**"
        )

        _log_prediction(
            question_url=getattr(question, "page_url", ""),
            question_text=question.question_text,
            prediction=final_prob,
            community_pred=community_pred,
            ensemble_pred=ensemble_prob,
            phase2_ran=phase2_ran,
            question_type="binary",
        )

        logger.info(f"Final prediction for {question.page_url}: {final_prob:.3f}")
        return ReasonedPrediction(prediction_value=final_prob, reasoning=full_reasoning)

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        prompt = clean_indents(f"""
            You are a professional forecaster.

            Question: {question.question_text}
            Options: {question.options}
            Background: {question.background_info}
            {question.resolution_criteria}
            {question.fine_print}
            Research: {research}
            Today: {datetime.now().strftime("%Y-%m-%d")}

            (a) Time left. (b) Status quo. (c) OUTSIDE VIEW (community/markets — address these).
            (d) INSIDE VIEW. (e) Unexpected outcome scenario.

            {self._get_conditional_disclaimer_if_necessary(question)}
            Anchor on base rates. Leave moderate probability on most options.

            Final probabilities:
            Option_A: Probability_A
            Option_B: Probability_B
            ...
        """)
        return await self._multiple_choice_prompt_to_forecast(question, prompt)

    async def _multiple_choice_prompt_to_forecast(self, question, prompt):
        parsing_instructions = clean_indents(f"""
            Option names must be one of: {question.options}
            Remove any "Option" prefix not part of the actual option name.
            Include 0% options.
        """)
        reasoning = await _run_numeric_ensemble(prompt)
        if not reasoning:
            reasoning = await self.get_llm("default", "llm").invoke(prompt)
        predicted_option_list: PredictedOptionList = await structure_output(
            text_to_structure=reasoning, output_type=PredictedOptionList,
            model=self.get_llm("parser", "llm"),
            num_validation_samples=self._structure_output_validation_samples,
            additional_instructions=parsing_instructions,
        )
        return ReasonedPrediction(prediction_value=predicted_option_list, reasoning=reasoning)

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        upper_msg, lower_msg = self._create_upper_and_lower_bound_messages(question)
        prompt = clean_indents(f"""
            You are a professional forecaster.

            Question: {question.question_text}
            Background: {question.background_info}
            {question.resolution_criteria}
            {question.fine_print}
            Units: {question.unit_of_measure or "infer from context"}
            Research: {research}
            Today: {datetime.now().strftime("%Y-%m-%d")}
            {lower_msg}
            {upper_msg}

            (a) Time left. (b) OUTSIDE VIEW (base rates, community). (c) INSIDE VIEW.
            (d) Expert/market expectations. (e) Low scenario. (f) High scenario.

            {self._get_conditional_disclaimer_if_necessary(question)}
            Use wide 10/90 intervals. No scientific notation.

            Percentile 10: XX
            Percentile 20: XX
            Percentile 40: XX
            Percentile 60: XX
            Percentile 80: XX
            Percentile 90: XX
        """)
        return await self._numeric_prompt_to_forecast(question, prompt)

    async def _numeric_prompt_to_forecast(self, question, prompt):
        reasoning = await _run_numeric_ensemble(prompt)
        if not reasoning:
            reasoning = await self.get_llm("default", "llm").invoke(prompt)
        parsing_instructions = clean_indents(f"""
            Numeric forecast for: "{question.question_text}".
            Units: {question.unit_of_measure}. Range: {question.lower_bound} to {question.upper_bound}.
            No scientific notation.
        """)
        percentile_list: list[Percentile] = await structure_output(
            reasoning, list[Percentile],
            model=self.get_llm("parser", "llm"),
            additional_instructions=parsing_instructions,
            num_validation_samples=self._structure_output_validation_samples,
        )
        prediction = NumericDistribution.from_question(percentile_list, question)
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    async def _run_forecast_on_date(
        self, question: DateQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        upper_msg, lower_msg = self._create_upper_and_lower_bound_messages(question)
        prompt = clean_indents(f"""
            You are a professional forecaster.

            Question: {question.question_text}
            Background: {question.background_info}
            {question.resolution_criteria}
            {question.fine_print}
            Research: {research}
            Today: {datetime.now().strftime("%Y-%m-%d")}
            {lower_msg}
            {upper_msg}

            Format dates YYYY-MM-DD, earliest first. Wide confidence intervals.

            (a) Time left. (b) OUTSIDE VIEW. (c) INSIDE VIEW.
            (d) Expert expectations. (e) Early scenario. (f) Late scenario.

            Percentile 10: YYYY-MM-DD
            Percentile 20: YYYY-MM-DD
            Percentile 40: YYYY-MM-DD
            Percentile 60: YYYY-MM-DD
            Percentile 80: YYYY-MM-DD
            Percentile 90: YYYY-MM-DD
        """)
        return await self._date_prompt_to_forecast(question, prompt)

    async def _date_prompt_to_forecast(self, question, prompt):
        reasoning = await _run_numeric_ensemble(prompt)
        if not reasoning:
            reasoning = await self.get_llm("default", "llm").invoke(prompt)
        parsing_instructions = clean_indents(f"""
            Date forecast for: "{question.question_text}".
            Range: {question.lower_bound} to {question.upper_bound}.
            Valid datetime strings, midnight UTC if no time given.
        """)
        date_percentile_list: list[DatePercentile] = await structure_output(
            reasoning, list[DatePercentile],
            model=self.get_llm("parser", "llm"),
            additional_instructions=parsing_instructions,
            num_validation_samples=self._structure_output_validation_samples,
        )
        percentile_list = [
            Percentile(percentile=p.percentile, value=p.value.timestamp())
            for p in date_percentile_list
        ]
        prediction = NumericDistribution.from_question(percentile_list, question)
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    async def _run_forecast_on_conditional(
        self, question: ConditionalQuestion, research: str
    ):
        parent_info, full_research = await self._get_question_prediction_info(question.parent, research, "parent")
        child_info,  full_research = await self._get_question_prediction_info(question.child, full_research, "child")
        yes_info,    full_research = await self._get_question_prediction_info(question.question_yes, full_research, "yes")
        no_info,     full_research = await self._get_question_prediction_info(question.question_no, full_research, "no")
        full_reasoning = clean_indents(f"""
            ## Parent: {parent_info.reasoning}
            ## Child: {child_info.reasoning}
            ## Yes: {yes_info.reasoning}
            ## No: {no_info.reasoning}
        """)
        return ReasonedPrediction(
            reasoning=full_reasoning,
            prediction_value=ConditionalPrediction(
                parent=parent_info.prediction_value, child=child_info.prediction_value,
                prediction_yes=yes_info.prediction_value, prediction_no=no_info.prediction_value,
            ),
        )

    async def _get_question_prediction_info(self, question, research, question_type):
        from forecasting_tools.data_models.data_organizer import DataOrganizer
        previous_forecasts = question.previous_forecasts
        if (question_type in ["parent", "child"] and previous_forecasts
                and question_type not in self.force_reforecast_in_conditional):
            pf = previous_forecasts[-1]
            if pf.timestamp_end is None or pf.timestamp_end > datetime.now(timezone.utc):
                pretty = DataOrganizer.get_readable_prediction(pf)
                return (
                    ReasonedPrediction(prediction_value=PredictionAffirmed(),
                                       reasoning=f"Reaffirmed at {pretty}."),
                    research,
                )
        info = await self._make_prediction(question, research)
        full_research = self._add_reasoning_to_research(research, info, question_type)
        return info, full_research

    def _add_reasoning_to_research(self, research, reasoning, question_type):
        from forecasting_tools.data_models.data_organizer import DataOrganizer
        qt = question_type.title()
        return clean_indents(f"""
            {research}
            ---
            ## {qt} Question Info
            Previous forecast: {DataOrganizer.get_readable_prediction(reasoning.prediction_value)}
            Reasoning: ```{reasoning.reasoning}```
            Do NOT re-forecast the {qt} question.
        """)

    def _create_upper_and_lower_bound_messages(self, question):
        if isinstance(question, NumericQuestion):
            upper = question.nominal_upper_bound if question.nominal_upper_bound is not None else question.upper_bound
            lower = question.nominal_lower_bound if question.nominal_lower_bound is not None else question.lower_bound
            unit  = question.unit_of_measure
        elif isinstance(question, DateQuestion):
            upper = question.upper_bound.date().isoformat()
            lower = question.lower_bound.date().isoformat()
            unit  = ""
        else:
            raise ValueError()
        upper_msg = (f"Creator thinks likely not higher than {upper} {unit}."
                     if question.open_upper_bound else f"Cannot be higher than {upper} {unit}.")
        lower_msg = (f"Creator thinks likely not lower than {lower} {unit}."
                     if question.open_lower_bound else f"Cannot be lower than {lower} {unit}.")
        return upper_msg, lower_msg

    def _get_conditional_disclaimer_if_necessary(self, question):
        if question.conditional_type not in ["yes", "no"]:
            return ""
        return "You are forecasting the CHILD question given the parent resolved. Never re-forecast the parent."


###############################################################################
# SUMMARY REPORT
###############################################################################

def save_summary_report(
    forecast_reports: list,
    output_path: str = "reports/summary.md",
) -> None:
    import re as _re
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    rows = []
    for report in forecast_reports:
        if isinstance(report, Exception):
            continue
        try:
            question     = report.question
            q_short      = question.question_text[:80] + ("..." if len(question.question_text) > 80 else "")
            url          = getattr(question, "page_url", "")
            community_raw= _extract_community_float(question)
            community_str= f"{community_raw*100:.1f}%" if community_raw is not None else "N/A"

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
                    p1_probs += [float(m) for m in _re.findall(r'(\d+\.?\d*)%', line)]
                if "Phase 2" in line and "%" in line and "Skipped" not in line:
                    p2_probs += [float(m) for m in _re.findall(r'(\d+\.?\d*)%', line)]

            spread_match = _re.search(r'spread[:\s]+(\d+)pp', reasoning)
            spread_str = f"{spread_match.group(1)}pp" if spread_match else "N/A"

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

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    logging.getLogger("LiteLLM").propagate = False

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
                model="openrouter/deepseek/deepseek-chat",
                temperature=0.2, timeout=120, allowed_tries=2,
                max_tokens=MAX_TOKENS_RESEARCH,
            ),
            "default": GeneralLlm(
                model="openrouter/deepseek/deepseek-r1",
                temperature=0.3, timeout=240, allowed_tries=2,
                max_tokens=MAX_TOKENS_FORECAST,
            ),
            "summarizer": "openrouter/deepseek/deepseek-chat",
            "parser":     "openrouter/deepseek/deepseek-chat",
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
        EXAMPLE_QUESTIONS = [
            "https://www.metaculus.com/questions/25523/nigel-farage-uk-pm-before-jan-1-2035/",
        ]
        danbot.skip_previously_forecasted_questions = False
        questions = [client.get_question_by_url(url) for url in EXAMPLE_QUESTIONS]
        forecast_reports = asyncio.run(danbot.forecast_questions(questions, return_exceptions=True))

    danbot.log_report_summary(forecast_reports)
    save_summary_report(forecast_reports)
