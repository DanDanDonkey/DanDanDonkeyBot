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

###############################################################################
# ENSEMBLE HELPERS
###############################################################################

ENSEMBLE_MODELS = {
    "DeepSeek-R1":      "openrouter/deepseek/deepseek-r1",
    "DeepSeek-V3":      "openrouter/deepseek/deepseek-chat",
    "Mistral-Small":    "openrouter/mistralai/mistral-small-3.1-24b-instruct:free",
    "Claude-Sonnet":    "anthropic/claude-sonnet-4-6",
}

EXTREMIZATION_ALPHA = 1.2

COMMUNITY_WEIGHT = 0.40

def _parse_probability(text: str) -> Optional[float]:
    matches = re.findall(r'[Pp]robability[:\s]+(\d+\.?\d*)\s*%', text)
    if matches:
        return max(0.01, min(0.99, float(matches[-1]) / 100))
    matches = re.findall(r'\b(\d+\.?\d*)\s*%', text)
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

def community_anchored_blend(model_pred: float, community_pred: Optional[float]) -> float:
    if community_pred is None:
        return model_pred
    blended = (1 - COMMUNITY_WEIGHT) * model_pred + COMMUNITY_WEIGHT * community_pred
    logger.info(
        f"  Community blend: ensemble={model_pred:.3f}, community={community_pred:.3f} "
        f"→ {blended:.3f} (weight={COMMUNITY_WEIGHT})"
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
# FREE DATA SOURCE HELPERS
###############################################################################

def _safe_get(url: str, timeout: int = 12, params: dict = None) -> dict | list | None:
    import requests
    try:
        resp = requests.get(url, params=params, timeout=timeout)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        logger.warning(f"HTTP request failed for {url[:80]}: {e}")
    return None

def fetch_gdelt_articles(query_text: str, max_articles: int = 8) -> str:
    import requests
    try:
        clean_query = query_text[:80].replace('"', '').replace("'", "")
        params = {"query": clean_query, "mode": "ArtList", "maxrecords": str(max_articles),
                  "format": "json", "sort": "DateDesc"}
        resp = requests.get("https://api.gdeltproject.org/api/v2/doc/doc", params=params, timeout=15)
        if resp.status_code != 200:
            return ""
        if "json" not in resp.headers.get("Content-Type", "") and "javascript" not in resp.headers.get("Content-Type", ""):
            return ""
        data = resp.json()
        articles = data.get("articles", [])
        if not articles:
            return ""
        lines = [f"  - {a.get('title','No title')} ({a.get('domain','?')}, {a.get('seendate','?')[:10]})"
                 for a in articles[:max_articles]]
        return "**Recent News (GDELT):**\n" + "\n".join(lines)
    except Exception as e:
        logger.warning(f"GDELT fetch failed: {e}")
        return ""

def fetch_polymarket_data(query_text: str) -> str:
    try:
        for n_words in [5, 3]:
            keywords = " ".join(query_text.split()[:n_words])
            data = _safe_get("https://gamma-api.polymarket.com/markets",
                             params={"_limit": "5", "closed": "false", "order": "volume24hr",
                                     "ascending": "false", "title_contains": keywords})
            if data and isinstance(data, list) and len(data):
                break
        if not data or not isinstance(data, list):
            return ""
        lines = []
        for m in data[:5]:
            title = m.get("question", m.get("title", "Unknown"))
            prices_raw = m.get("outcomePrices", "")
            outcomes = m.get("outcomes", "")
            vol = m.get("volume24hr", 0)
            try:
                prices = json.loads(prices_raw) if isinstance(prices_raw, str) else prices_raw
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

def fetch_web_search(query: str, max_results: int = 5) -> str:
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return ""
        out = "## Web Search Results\n"
        for r in results:
            out += f"- **{r['title']}**: {r['body']}\n  Source: {r['href']}\n"
        return out
    except Exception as e:
        return f"(Web search unavailable: {e})"

def fetch_reddit_context(query: str, subreddits: list = None) -> str:
    try:
        import requests
        subreddits = subreddits or ["worldnews", "geopolitics", "ukpolitics", "economics"]
        results = []
        for sub in subreddits[:3]:
            r = requests.get(f"https://www.reddit.com/r/{sub}/search.json",
                             params={"q": query[:100], "sort": "relevance", "limit": 5, "t": "month"},
                             headers={"User-Agent": "DanDanDonkeyBot/1.0"}, timeout=10)
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
        data = _safe_get("https://api.stlouisfed.org/fred/series/search",
                         params={"search_text": query_text[:60], "api_key": api_key,
                                 "file_type": "json", "limit": "3",
                                 "order_by": "popularity", "sort_order": "desc"})
        if not data or "seriess" not in data:
            return ""
        lines = []
        for series in data["seriess"][:3]:
            sid = series.get("id", "")
            obs = _safe_get("https://api.stlouisfed.org/fred/series/observations",
                            params={"series_id": sid, "api_key": api_key,
                                    "file_type": "json", "sort_order": "desc", "limit": "1"})
            if obs and obs.get("observations"):
                o = obs["observations"][0]
                lines.append(f"  - {series.get('title','')} ({sid}): {o.get('value','N/A')} ({o.get('date','?')})")
        return ("**Economic Data (FRED):**\n" + "\n".join(lines)) if lines else ""
    except Exception as e:
        logger.warning(f"FRED fetch failed: {e}")
        return ""

def fetch_world_bank_data(query_text: str) -> str:
    keyword_to_indicators = {
        "gdp": "NY.GDP.MKTP.CD", "population": "SP.POP.TOTL",
        "inflation": "FP.CPI.TOTL.ZG", "unemployment": "SL.UEM.TOTL.ZS",
        "poverty": "SI.POV.DDAY", "life expectancy": "SP.DYN.LE00.IN",
        "co2": "EN.ATM.CO2E.PC", "emissions": "EN.ATM.CO2E.PC",
        "trade": "NE.TRD.GNFS.ZS", "debt": "GC.DOD.TOTL.GD.ZS",
        "education": "SE.ADT.LITR.ZS", "mortality": "SP.DYN.IMRT.IN",
        "energy": "EG.USE.PCAP.KG.OE", "renewable": "EG.FEC.RNEW.ZS",
    }
    matched = [(k, v) for k, v in keyword_to_indicators.items() if k in query_text.lower()]
    if not matched:
        return ""
    try:
        lines = []
        for keyword, indicator in matched[:2]:
            data = _safe_get(f"https://api.worldbank.org/v2/country/WLD/indicator/{indicator}",
                             params={"format": "json", "per_page": "1", "mrv": "1"})
            if data and isinstance(data, list) and len(data) > 1 and data[1]:
                rec = data[1][0]
                if rec.get("value") is not None:
                    lines.append(f"  - {rec.get('indicator',{}).get('value',keyword)}: {rec['value']:,.2f} ({rec.get('date','?')})")
        return ("**Global Data (World Bank):**\n" + "\n".join(lines)) if lines else ""
    except Exception as e:
        logger.warning(f"World Bank fetch failed: {e}")
        return ""

def fetch_wikipedia_context(query_text: str) -> str:
    try:
        keywords = " ".join(query_text.split()[:6])
        search = _safe_get("https://en.wikipedia.org/w/api.php",
                           params={"action": "query", "list": "search", "srsearch": keywords,
                                   "srlimit": "1", "format": "json"})
        if not search or not search.get("query", {}).get("search"):
            return ""
        title = search["query"]["search"][0]["title"]
        extract_data = _safe_get("https://en.wikipedia.org/w/api.php",
                                 params={"action": "query", "titles": title, "prop": "extracts",
                                         "exintro": "true", "explaintext": "true", "format": "json"})
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
    health_kws = ["disease", "health", "pandemic", "virus", "vaccine", "mortality",
                  "covid", "infection", "epidemic", "WHO", "measles", "malaria",
                  "tuberculosis", "hiv", "flu", "influenza", "bird flu", "mpox",
                  "ebola", "cholera", "polio", "death rate", "life expectancy"]
    if not any(kw in query_text.lower() for kw in health_kws):
        return ""
    try:
        keywords = " ".join(query_text.split()[:4])
        data = _safe_get("https://ghoapi.azureedge.net/api/Indicator",
                         params={"$filter": f"contains(IndicatorName,'{keywords}')"})
        if not data or not data.get("value"):
            return ""
        lines = []
        for ind in data["value"][:2]:
            code, name = ind.get("IndicatorCode", ""), ind.get("IndicatorName", "")
            obs = _safe_get(f"https://ghoapi.azureedge.net/api/{code}",
                            params={"$filter": "SpatialDim eq 'GLOBAL'", "$top": "1", "$orderby": "TimeDim desc"})
            if obs and obs.get("value"):
                o = obs["value"][0]
                lines.append(f"  - {name}: {o.get('NumericValue','N/A')} ({o.get('TimeDim','?')})")
            else:
                lines.append(f"  - {name} ({code})")
        return ("**Global Health (WHO GHO):**\n" + "\n".join(lines)) if lines else ""
    except Exception as e:
        logger.warning(f"WHO GHO fetch failed: {e}")
        return ""

def fetch_acled_conflict_data(query_text: str) -> str:
    acled_key = os.environ.get("ACLED_API_KEY", "")
    acled_email = os.environ.get("ACLED_EMAIL", "")
    if not acled_key or not acled_email:
        return ""
    conflict_kws = ["war", "conflict", "military", "attack", "violence", "troops",
                    "invasion", "battle", "ceasefire", "coup", "protest", "riot",
                    "terrorism", "rebel", "armed"]
    if not any(kw in query_text.lower() for kw in conflict_kws):
        return ""
    try:
        data = _safe_get("https://api.acleddata.com/acled/read",
                         params={"key": acled_key, "email": acled_email, "limit": "5",
                                 "fields": "event_date|country|event_type|fatalities",
                                 "order": "event_date:desc"})
        if not data or not data.get("data"):
            return ""
        lines = [f"  - {ev.get('event_date','?')}: {ev.get('event_type','?')} in {ev.get('country','?')} ({ev.get('fatalities','?')} fatalities)"
                 for ev in data["data"][:5]]
        return "**Recent Conflict (ACLED):**\n" + "\n".join(lines)
    except Exception as e:
        logger.warning(f"ACLED fetch failed: {e}")
        return ""

def fetch_media_trend(query_text: str) -> str:
    import requests
    try:
        resp = requests.get("https://api.gdeltproject.org/api/v2/doc/doc",
                            params={"query": query_text[:60].replace('"', ''), "mode": "TimelineVol",
                                    "format": "json", "TIMESPAN": "6m"}, timeout=12)
        if resp.status_code != 200 or "json" not in resp.headers.get("Content-Type", ""):
            return ""
        series = resp.json().get("timeline", [{}])[0].get("data", [])
        if len(series) < 4:
            return ""
        recent = sum(d.get("value", 0) for d in series[-6:]) / max(len(series[-6:]), 1)
        older = sum(d.get("value", 0) for d in series[:6]) / max(len(series[:6]), 1)
        if older > 0:
            change = (recent - older) / older * 100
            direction = "increasing" if change > 10 else "decreasing" if change < -10 else "stable"
            return f"**Media Attention Trend:** {direction} ({change:+.0f}% over 6 months)"
    except Exception as e:
        logger.warning(f"Media trend fetch failed: {e}")
    return ""

def fetch_sec_company_info(query_text: str) -> str:
    fin_kws = ["stock", "revenue", "earnings", "company", "shares", "market cap",
               "ipo", "nasdaq", "nyse", "profit", "loss", "quarterly", "10-k", "8-k"]
    if not any(kw in query_text.lower() for kw in fin_kws):
        return ""
    try:
        words = query_text.split()
        company_query = " ".join(w for w in words[:8] if w and w[0].isupper())[:40] or " ".join(words[:4])
        data = _safe_get("https://efts.sec.gov/LATEST/search-index",
                         params={"q": company_query, "forms": "10-K,8-K",
                                 "dateRange": "custom", "startdt": "2024-01-01"}, timeout=10)
        if not data:
            return ""
        hits = data.get("hits", {}).get("hits", [])
        if not hits:
            return ""
        lines = [f"  - {h.get('_source',{}).get('display_date_filed','?')}: "
                 f"{h.get('_source',{}).get('entity_name','?')} — "
                 f"{h.get('_source',{}).get('file_type','?')}" for h in hits[:3]]
        return ("**SEC Filings (EDGAR):**\n" + "\n".join(lines)) if lines else ""
    except Exception as e:
        logger.warning(f"SEC company info fetch failed: {e}")
        return ""

def fetch_sec_filings(query_text: str) -> str:
    fin_kws = ["stock", "revenue", "earnings", "company", "shares", "market cap",
               "ipo", "nasdaq", "nyse", "profit", "loss"]
    if not any(kw in query_text.lower() for kw in fin_kws):
        return ""
    try:
        keywords = " ".join(query_text.split()[:5])
        data = _safe_get("https://efts.sec.gov/LATEST/search-index",
                         params={"q": keywords, "forms": "8-K,10-Q",
                                 "dateRange": "custom", "startdt": "2024-06-01"}, timeout=10)
        if not data:
            return ""
        hits = data.get("hits", {}).get("hits", [])
        if not hits:
            return ""
        lines = [f"  - {h.get('_source',{}).get('display_date_filed','?')}: "
                 f"{h.get('_source',{}).get('entity_name','?')} "
                 f"{h.get('_source',{}).get('file_type','?')}" for h in hits[:3]]
        return ("**Recent SEC Filings:**\n" + "\n".join(lines)) if lines else ""
    except Exception as e:
        logger.warning(f"SEC filings fetch failed: {e}")
        return ""

###############################################################################
# MULTI-MODEL DEBATE ENGINE
###############################################################################

async def _call_model(model_id: str, prompt: str, temperature: float = 0.3) -> str:
    try:
        llm = GeneralLlm(model=model_id, temperature=temperature, timeout=120, allowed_tries=2)
        return await llm.invoke(prompt)
    except Exception as e:
        logger.warning(f"Model {model_id} failed: {e}")
        return ""

async def run_debate_ensemble(
    question_text: str,
    base_prompt: str,
    question_context: str = "",
) -> tuple[float, str]:
    """
    Two-round multi-model debate for binary questions.

    Round 1: Each model independently forecasts from the same research brief.
    Round 2: Each model sees ALL other models' Round 1 reasoning + probabilities,
             then produces a revised forecast.
    Final:   Revised probabilities are aggregated via log-odds extremization (alpha=1.2).

    Returns: (final_probability_decimal, combined_reasoning_string)
    """
    logger.info("=== DEBATE ENSEMBLE: Round 1 (independent forecasts) ===")

    round1_tasks = {
        name: _call_model(model_id, base_prompt)
        for name, model_id in ENSEMBLE_MODELS.items()
    }
    results = await asyncio.gather(*round1_tasks.values(), return_exceptions=True)
    round1_results: Dict[str, str] = {
        name: (r if isinstance(r, str) else "")
        for name, r in zip(round1_tasks.keys(), results)
    }

    round1_probs: Dict[str, Optional[float]] = {}
    for name, text in round1_results.items():
        prob = _parse_probability(text)
        round1_probs[name] = prob
        logger.info(f"  Round 1 – {name}: {f'{prob*100:.1f}%' if prob else 'PARSE FAIL'}")

    debate_context = "\n\n".join(
        f"### {name} (Round 1 forecast: {f'{p*100:.1f}%' if p else 'unknown'})\n{text[:1500]}"
        for (name, text), p in zip(round1_results.items(), round1_probs.values())
    )

    logger.info("=== DEBATE ENSEMBLE: Round 2 (revisions after seeing peers) ===")

    revision_prompt_template = clean_indents(f"""
        You are one of four independent forecasters working on this question:

        {question_text}

        {question_context}

        Below are the Round 1 forecasts and reasoning from ALL four forecasters
        (including your own). Study them carefully. Where you disagree with others,
        explain why. Where their arguments are compelling, update your view.

        ## All Round 1 Forecasts
        {debate_context}

        ## Your Task
        Write a SHORT revised analysis (3-5 paragraphs):
        1. Which forecasters do you agree with most, and why?
        2. What is the strongest argument AGAINST your Round 1 position?
        3. Any evidence or base rate you think others missed?
        4. Your REVISED probability, accounting for the group reasoning.

        End with: "Revised Probability: ZZ%"
    """)

    round2_tasks = {
        name: _call_model(model_id, revision_prompt_template)
        for name, model_id in ENSEMBLE_MODELS.items()
    }
    results = await asyncio.gather(*round2_tasks.values(), return_exceptions=True)
    round2_results: Dict[str, str] = {
        name: (r if isinstance(r, str) else "")
        for name, r in zip(round2_tasks.keys(), results)
    }

    round2_probs: List[float] = []
    for name, text in round2_results.items():
        prob = _parse_probability(text)
        if prob is not None:
            round2_probs.append(prob)
            logger.info(f"  Round 2 – {name}: {prob*100:.1f}%")
        else:
            fallback = round1_probs.get(name)
            if fallback is not None:
                round2_probs.append(fallback)
                logger.info(f"  Round 2 – {name}: PARSE FAIL, using Round 1 fallback {fallback*100:.1f}%")

    if not round2_probs:
        round2_probs = [p for p in round1_probs.values() if p is not None]

    if not round2_probs:
        logger.error("All model calls failed, returning 0.5")
        return 0.5, "All model calls failed."

    final_prob = extremize_log_odds(round2_probs, alpha=EXTREMIZATION_ALPHA)
    logger.info(f"  Ensemble final (alpha={EXTREMIZATION_ALPHA}): {final_prob*100:.1f}%  "
                f"(from {[f'{p*100:.1f}%' for p in round2_probs]})")

    combined_reasoning = "## Multi-Model Debate Results\n\n"
    combined_reasoning += "### Round 1 Independent Forecasts\n"
    for name, prob in round1_probs.items():
        combined_reasoning += f"- **{name}**: {f'{prob*100:.1f}%' if prob else 'failed'}\n"
    combined_reasoning += "\n### Round 2 Revised Forecasts (after peer review)\n"
    for name, text in round2_results.items():
        prob = _parse_probability(text)
        combined_reasoning += f"- **{name}**: {f'{prob*100:.1f}%' if prob else 'failed'}\n"
        combined_reasoning += f"  _{text[:400].strip()}_\n\n"
    combined_reasoning += (
        f"\n### Ensemble Result\n"
        f"Raw revised probabilities: {[f'{p*100:.1f}%' for p in round2_probs]}\n"
        f"Log-odds extremization (α={EXTREMIZATION_ALPHA}): **{final_prob*100:.1f}%**\n"
    )

    return final_prob, combined_reasoning

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
            sources = [
                fetch_metaculus_community(question),
                fetch_polymarket_data(q_text),
                fetch_web_search(q_text),
                fetch_reddit_context(q_text),
                fetch_gdelt_articles(q_text),
                fetch_fred_data(q_text),
                fetch_world_bank_data(q_text),
                fetch_who_health_data(q_text),
                fetch_acled_conflict_data(q_text),
                fetch_wikipedia_context(q_text),
                fetch_media_trend(q_text),
                fetch_sec_company_info(q_text),
                fetch_sec_filings(q_text),
            ]
            external_data = "\n\n".join(s for s in sources if s) or "(No external data.)"

            research_prompt = clean_indents(f"""
                You are a research assistant to a superforecaster. Produce a structured
                research brief covering both outside view and inside view.

                ## Question
                {question.question_text}

                ## Resolution Criteria
                {question.resolution_criteria}

                {question.fine_print}

                ## External Data
                {external_data}

                ## Important: Community Prediction
                If Metaculus Community Prediction appears above, treat it as a STRONG PRIOR
                representing hundreds of calibrated forecasters. Your synthesis must address
                whether and why you agree or disagree.

                ## Output Sections (required)

                ### OUTSIDE VIEW
                - Base rate / reference class
                - Prediction market prices
                - Metaculus community prediction (MUST address)
                - Analogous past situations

                ### INSIDE VIEW
                - Recent specific developments
                - Key causal factors
                - Relevant data from above sources
                - Upcoming deadlines / events

                ### KEY UNCERTAINTIES
                - 2-3 biggest unknowns

                ### SYNTHESIS
                - Overall assessment (no exact probability, just qualitative range)
                - How much you weight community vs. your own analysis

                Be concise. Cite data. Do NOT fabricate.
            """)

            researcher = self.get_llm("researcher")
            if isinstance(researcher, GeneralLlm):
                research = await researcher.invoke(research_prompt)
            elif isinstance(researcher, str) and researcher.startswith("asknews/"):
                research = await AskNewsSearcher().call_preconfigured_version(researcher, research_prompt)
            elif isinstance(researcher, str) and researcher.startswith("smart-searcher"):
                searcher = SmartSearcher(model=researcher.removeprefix("smart-searcher/"),
                                         temperature=0, num_searches_to_run=2,
                                         num_sites_per_search=10, use_advanced_filters=False)
                research = await searcher.invoke(research_prompt)
            elif not researcher or researcher in ("None", "no_research"):
                research = external_data
            else:
                research = await self.get_llm("researcher", "llm").invoke(research_prompt)

            logger.info(f"Research done for {question.page_url}")
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
            (c) OUTSIDE VIEW base rate — what % of similar situations resolved Yes historically?
            (d) INSIDE VIEW — what specific current evidence shifts the probability?
            (e) Brief No scenario.
            (f) Brief Yes scenario.
            (g) Metaculus community prediction (if in research above) — state it, agree/disagree, WHY.
                Unless you have strong specific evidence, stay within ~10pp of it.

            {self._get_conditional_disclaimer_if_necessary(question)}

            End with: "Probability: ZZ%"
        """)

        question_context = clean_indents(f"""
            Background: {question.background_info[:500]}
            Resolution: {question.resolution_criteria[:300]}
        """)

        ensemble_prob, debate_reasoning = await run_debate_ensemble(
            question_text=question.question_text,
            base_prompt=base_prompt,
            question_context=question_context,
        )

        community_pred = _extract_community_float(question)
        final_prob = community_anchored_blend(ensemble_prob, community_pred)

        full_reasoning = (
            debate_reasoning
            + f"\n\n### Community Blend\n"
            + f"Ensemble: {ensemble_prob*100:.1f}%  |  "
            + f"Community: {f'{community_pred*100:.1f}%' if community_pred else 'N/A'}  |  "
            + f"Final (weight={COMMUNITY_WEIGHT}): **{final_prob*100:.1f}%**"
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

            Final probabilities:
            Option_A: Probability_A
            Option_B: Probability_B
            ...
        """)
        return await self._multiple_choice_prompt_to_forecast(question, prompt)

    async def _multiple_choice_prompt_to_forecast(self, question, prompt):
        parsing_instructions = clean_indents(f"""
            Option names must be one of: {question.options}
            Remove any "Option" prefix. Include 0% options.
        """)
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        predicted_option_list: PredictedOptionList = await structure_output(
            text_to_structure=reasoning, output_type=PredictedOptionList,
            model=self.get_llm("parser", "llm"),
            num_validation_samples=self._structure_output_validation_samples,
            additional_instructions=parsing_instructions,
        )
        return ReasonedPrediction(prediction_value=predicted_option_list, reasoning=reasoning)

    async def _run_forecast_on_numeric(self, question: NumericQuestion, research: str) -> ReasonedPrediction[NumericDistribution]:
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

    async def _run_forecast_on_date(self, question: DateQuestion, research: str) -> ReasonedPrediction[NumericDistribution]:
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

    def _create_upper_and_lower_bound_messages(self, question):
        if isinstance(question, NumericQuestion):
            upper = question.nominal_upper_bound if question.nominal_upper_bound is not None else question.upper_bound
            lower = question.nominal_lower_bound if question.nominal_lower_bound is not None else question.lower_bound
            unit = question.unit_of_measure
        elif isinstance(question, DateQuestion):
            upper = question.upper_bound.date().isoformat()
            lower = question.lower_bound.date().isoformat()
            unit = ""
        else:
            raise ValueError()
        upper_msg = (f"Creator thinks likely not higher than {upper} {unit}."
                     if question.open_upper_bound else f"Cannot be higher than {upper} {unit}.")
        lower_msg = (f"Creator thinks likely not lower than {lower} {unit}."
                     if question.open_lower_bound else f"Cannot be lower than {lower} {unit}.")
        return upper_msg, lower_msg

    async def _run_forecast_on_conditional(self, question: ConditionalQuestion, research: str):
        parent_info, full_research = await self._get_question_prediction_info(question.parent, research, "parent")
        child_info, full_research = await self._get_question_prediction_info(question.child, full_research, "child")
        yes_info, full_research = await self._get_question_prediction_info(question.question_yes, full_research, "yes")
        no_info, full_research = await self._get_question_prediction_info(question.question_no, full_research, "no")
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
                return ReasonedPrediction(prediction_value=PredictionAffirmed(),
                                          reasoning=f"Reaffirmed at {pretty}."), research
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

    def _get_conditional_disclaimer_if_necessary(self, question):
        if question.conditional_type not in ["yes", "no"]:
            return ""
        return "You are forecasting the CHILD question given the parent resolved. Never re-forecast the parent."

def save_summary_report(forecast_reports: list, output_path: str = "reports/summary.md") -> None:
    import os

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    rows = []
    for report in forecast_reports:
        if isinstance(report, Exception):
            continue
        try:
            question = report.question
            question_text = question.question_text[:80] + ("..." if len(question.question_text) > 80 else "")
            url = getattr(question, "page_url", "")

            community_raw = _extract_community_float(question)
            community_str = f"{community_raw*100:.1f}%" if community_raw is not None else "N/A"

            final_pred = None
            reasoning = ""
            for forecast_report in getattr(report, "prediction_reports", []):
                pred = getattr(forecast_report, "prediction", None)
                if isinstance(pred, float):
                    final_pred = pred
                reasoning = getattr(forecast_report, "reasoning", "") or ""

            if final_pred is None:
                continue

            final_str = f"{final_pred*100:.1f}%"

            delta_str = "N/A"
            if community_raw is not None:
                delta = (final_pred - community_raw) * 100
                delta_str = f"{delta:+.1f}pp"

            r1_probs = []
            r2_probs = []
            for line in reasoning.splitlines():
                if "Round 1" in line and "%" in line:
                    import re
                    matches = re.findall(r'(\d+\.?\d*)%', line)
                    r1_probs += [float(m) for m in matches]
                if "Round 2" in line and "%" in line:
                    import re
                    matches = re.findall(r'(\d+\.?\d*)%', line)
                    r2_probs += [float(m) for m in matches]

            r1_str = f"{sum(r1_probs)/len(r1_probs):.1f}%" if r1_probs else "N/A"
            r2_str = f"{sum(r2_probs)/len(r2_probs):.1f}%" if r2_probs else "N/A"

            rows.append({
                "question": question_text,
                "url": url,
                "community": community_str,
                "r1_avg": r1_str,
                "r2_avg": r2_str,
                "final": final_str,
                "delta": delta_str,
            })
        except Exception as e:
            logger.warning(f"Could not parse report for summary: {e}")
            continue

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"# DanDanDonkeyBot Forecast Summary",
        f"*Generated: {now}*",
        f"*Questions forecasted: {len(rows)}*",
        "",
        "| # | Question | Community | R1 Avg | R2 Avg | Final | Δ Community |",
        "|---|----------|-----------|--------|--------|-------|-------------|",
    ]
    for i, r in enumerate(rows, 1):
        q_link = f"[{r['question']}]({r['url']})" if r['url'] else r['question']
        lines.append(
            f"| {i} | {q_link} | {r['community']} | {r['r1_avg']} | {r['r2_avg']} | {r['final']} | {r['delta']} |"
        )

    lines += [
        "",
        "## Divergence from Community (>5pp)",
        "",
    ]
    diverged = [r for r in rows if r['delta'] not in ("N/A",) and abs(float(r['delta'].replace("pp","").replace("+",""))) > 5]
    if diverged:
        for r in diverged:
            lines.append(f"- **{r['delta']}** — {r['question']}")
    else:
        lines.append("- None — ensemble stayed within 5pp of community on all questions.")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Summary saved to {output_path}")
    logger.info(f"Summary report saved to {output_path}")

def save_summary_report(forecast_reports: list, output_path: str = "reports/summary.md") -> None:
    import os
    import re as _re

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    rows = []
    for report in forecast_reports:
        if isinstance(report, Exception):
            continue
        try:
            question = report.question
            q_text = question.question_text
            q_short = q_text[:80] + ("..." if len(q_text) > 80 else "")
            url = getattr(question, "page_url", "")
            community_raw = _extract_community_float(question)
            community_str = f"{community_raw*100:.1f}%" if community_raw is not None else "N/A"

            final_pred = None
            reasoning = ""
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
                delta = (final_pred - community_raw) * 100
                delta_str = f"{delta:+.1f}pp"

            r1_probs, r2_probs = [], []
            for line in reasoning.split("\n"):
                if "Round 1" in line and "%" in line:
                    r1_probs += [float(m) for m in _re.findall(r'(\d+\.?\d*)%', line)]
                if "Round 2" in line and "%" in line:
                    r2_probs += [float(m) for m in _re.findall(r'(\d+\.?\d*)%', line)]

            r1_str = f"{sum(r1_probs)/len(r1_probs):.1f}%" if r1_probs else "N/A"
            r2_str = f"{sum(r2_probs)/len(r2_probs):.1f}%" if r2_probs else "N/A"

            rows.append({
                "question": q_short, "url": url,
                "community": community_str, "r1_avg": r1_str,
                "r2_avg": r2_str, "final": final_str, "delta": delta_str,
            })
        except Exception as e:
            logger.warning(f"Could not parse report for summary: {e}")

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    md = [
        "# DanDanDonkeyBot Forecast Summary",
        f"*Generated: {now} | Questions: {len(rows)}*",
        "",
        "| # | Question | Community | R1 Avg | R2 Avg | Final | Δ Community |",
        "|---|----------|-----------|--------|--------|-------|-------------|",
    ]
    for i, r in enumerate(rows, 1):
        q_cell = f"[{r['question']}]({r['url']})" if r['url'] else r['question']
        md.append(f"| {i} | {q_cell} | {r['community']} | {r['r1_avg']} | {r['r2_avg']} | {r['final']} | {r['delta']} |")

    md += ["", "## Diverged from Community (>5pp)", ""]
    diverged = [
        r for r in rows
        if r['delta'] != "N/A" and abs(float(r['delta'].replace("pp", "").replace("+", ""))) > 5
    ]
    if diverged:
        for r in diverged:
            md.append(f"- **{r['delta']}** — {r['question']}")
    else:
        md.append("- None")

    with open(output_path, "w") as f:
        f.write("\n".join(md))
    logger.info(f"Summary report saved to {output_path}")


###############################################################################
# ENTRYPOINT
###############################################################################

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    logging.getLogger("LiteLLM").propagate = False

    parser = argparse.ArgumentParser(description="Run DanDanDonkeyBot v4")
    parser.add_argument("--mode", type=str,
                        choices=["tournament", "metaculus_cup", "test_questions"],
                        default="tournament")
    args = parser.parse_args()
    run_mode: Literal["tournament", "metaculus_cup", "test_questions"] = args.mode

    danbot = DanDanDonkeyBot(
        research_reports_per_question=1,
        predictions_per_research_report=1,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=False,
        folder_to_save_reports_to="reports",
        skip_previously_forecasted_questions=True,
        extra_metadata_in_explanation=True,
        llms={
            "researcher": GeneralLlm(
                model="anthropic/claude-sonnet-4-6",
                temperature=0.2, timeout=300, allowed_tries=2
            ),
            "default": GeneralLlm(
                model="openrouter/deepseek/deepseek-r1",
                temperature=0.3, timeout=300, allowed_tries=2
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
            "https://www.metaculus.com/questions/578/human-extinction-by-2100/",
            "https://www.metaculus.com/questions/22427/number-of-new-leading-ai-labs/",
            "https://www.metaculus.com/questions/25523/nigel-farage-uk-pm-before-jan-1-2035/",
        ]
        danbot.skip_previously_forecasted_questions = False
        questions = [client.get_question_by_url(url) for url in EXAMPLE_QUESTIONS]
        forecast_reports = asyncio.run(danbot.forecast_questions(questions, return_exceptions=True))

    danbot.log_report_summary(forecast_reports)
    save_summary_report(forecast_reports)
