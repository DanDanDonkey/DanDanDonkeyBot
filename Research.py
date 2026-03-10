import asyncio
import logging
import os

from forecasting_tools import (
    AskNewsSearcher,
    GeneralLlm,
    MetaculusQuestion,
    SmartSearcher,
    clean_indents,
)

from llm_calls import get_researcher_llm

logger = logging.getLogger(__name__)

_SRC_CAP_ASKNEWS   = 2500
_SRC_CAP_DEFAULT   = 600
_EXTERNAL_DATA_CAP = 6000


###############################################################################
# DATA SOURCE FETCHERS
###############################################################################

def _safe_get(url: str, timeout: int = 12, params: dict = None):
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
            logger.warning("AskNews: no credentials found in env.")
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
    import json as _json
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
                prices   = _json.loads(prices_raw) if isinstance(prices_raw, str) else prices_raw
                outcomes = _json.loads(outcomes) if isinstance(outcomes, str) else outcomes
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
    import requests
    try:
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
# RESEARCH ORCHESTRATOR
###############################################################################

RESEARCH_PROMPT = """
You are a research assistant to a professional superforecaster.
Produce a CONCISE structured brief — aim for 400–600 words total.

## Question
{question_text}

## Resolution Criteria
{resolution_criteria}

{fine_print}

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
"""


async def run_research(question: MetaculusQuestion, researcher_llm=None) -> str:
    q_text = question.question_text

    asknews_raw  = await fetch_asknews(q_text)
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

    logger.info(f"  External data: {len(external_data)} chars from {1 + len(sync_sources)} sources")

    research_prompt = RESEARCH_PROMPT.format(
        question_text=question.question_text,
        resolution_criteria=question.resolution_criteria,
        fine_print=question.fine_print or "",
        external_data=external_data,
    )

    if researcher_llm is None:
        researcher_llm = get_researcher_llm()

    if isinstance(researcher_llm, GeneralLlm):
        research = await researcher_llm.invoke(research_prompt)
    elif isinstance(researcher_llm, str) and researcher_llm.startswith("asknews/"):
        research = await AskNewsSearcher().call_preconfigured_version(researcher_llm, research_prompt)
    elif isinstance(researcher_llm, str) and researcher_llm.startswith("smart-searcher"):
        searcher = SmartSearcher(
            model=researcher_llm.removeprefix("smart-searcher/"),
            temperature=0, num_searches_to_run=2,
            num_sites_per_search=10, use_advanced_filters=False,
        )
        research = await searcher.invoke(research_prompt)
    elif not researcher_llm or researcher_llm in ("None", "no_research"):
        research = external_data
    else:
        research = await get_researcher_llm().invoke(research_prompt)

    logger.info(f"Research complete for {question.page_url}")
    return research
