import argparse
import asyncio
import logging
from datetime import datetime, timezone
import dotenv
from typing import Literal
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
    PredictionTypes,
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
        params = {
            "query": clean_query,
            "mode": "ArtList",
            "maxrecords": str(max_articles),
            "format": "json",
            "sort": "DateDesc",
        }
        resp = requests.get("https://api.gdeltproject.org/api/v2/doc/doc", params=params, timeout=15)
        if resp.status_code != 200:
            return ""
        content_type = resp.headers.get("Content-Type", "")
        if "json" not in content_type and "javascript" not in content_type:
            return ""
        data = resp.json()
        articles = data.get("articles", [])
        if not articles:
            return ""
        lines = []
        for a in articles[:max_articles]:
            title = a.get("title", "No title")
            source = a.get("domain", "Unknown")
            date = a.get("seendate", "Unknown")[:10]
            lines.append(f"  - {title} ({source}, {date})")
        return "**Recent News (GDELT):**\n" + "\n".join(lines)
    except Exception as e:
        logger.warning(f"GDELT fetch failed: {e}")
        return ""


def fetch_polymarket_data(query_text: str) -> str:
    try:
        keywords = " ".join(query_text.split()[:5])
        data = _safe_get(
            "https://gamma-api.polymarket.com/markets",
            params={"_limit": "5", "closed": "false", "order": "volume24hr",
                    "ascending": "false", "title_contains": keywords},
        )
        if not data or not isinstance(data, list) or len(data) == 0:
            keywords = " ".join(query_text.split()[:3])
            data = _safe_get(
                "https://gamma-api.polymarket.com/markets",
                params={"_limit": "5", "closed": "false", "order": "volume24hr",
                        "ascending": "false", "title_contains": keywords},
            )
        if not data or not isinstance(data, list) or len(data) == 0:
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
            lines.append(f"  - {title} -> {price_str} (24h vol: ${float(vol):,.0f})")
        return "**Prediction Markets (Polymarket):**\n" + "\n".join(lines)
    except Exception as e:
        logger.warning(f"Polymarket fetch failed: {e}")
        return ""


def fetch_metaculus_community(question: MetaculusQuestion) -> str:
    try:
        cp = question.community_prediction
        if cp is not None:
            if isinstance(cp, float):
                return f"**Metaculus Community Prediction:** {cp*100:.1f}%"
            else:
                return f"**Metaculus Community Prediction:** {cp}"
    except Exception:
        pass
    return ""


def fetch_fred_data(query_text: str) -> str:
    api_key = os.environ.get("FRED_API_KEY", "")
    if not api_key:
        return ""
    try:
        data = _safe_get(
            "https://api.stlouisfed.org/fred/series/search",
            params={"search_text": query_text[:60], "api_key": api_key,
                    "file_type": "json", "limit": "3", "order_by": "popularity", "sort_order": "desc"},
        )
        if not data or "seriess" not in data:
            return ""
        lines = []
        for series in data["seriess"][:3]:
            series_id = series.get("id", "")
            title = series.get("title", "")
            obs_data = _safe_get(
                "https://api.stlouisfed.org/fred/series/observations",
                params={"series_id": series_id, "api_key": api_key,
                        "file_type": "json", "sort_order": "desc", "limit": "1"},
            )
            if obs_data and "observations" in obs_data and obs_data["observations"]:
                obs = obs_data["observations"][0]
                lines.append(f"  - {title} ({series_id}): {obs.get('value', 'N/A')} (as of {obs.get('date', 'N/A')})")
        return ("**Economic Data (FRED):**\n" + "\n".join(lines)) if lines else ""
    except Exception as e:
        logger.warning(f"FRED fetch failed: {e}")
        return ""


def fetch_world_bank_data(query_text: str) -> str:
    try:
        keyword_to_indicators = {
            "gdp": "NY.GDP.MKTP.CD", "population": "SP.POP.TOTL",
            "inflation": "FP.CPI.TOTL.ZG", "unemployment": "SL.UEM.TOTL.ZS",
            "poverty": "SI.POV.DDAY", "life expectancy": "SP.DYN.LE00.IN",
            "co2": "EN.ATM.CO2E.PC", "emissions": "EN.ATM.CO2E.PC",
            "trade": "NE.TRD.GNFS.ZS", "debt": "GC.DOD.TOTL.GD.ZS",
            "education": "SE.ADT.LITR.ZS", "mortality": "SP.DYN.IMRT.IN",
            "energy": "EG.USE.PCAP.KG.OE", "renewable": "EG.FEC.RNEW.ZS",
        }
        query_lower = query_text.lower()
        matched = [(k, v) for k, v in keyword_to_indicators.items() if k in query_lower]
        if not matched:
            return ""
        lines = []
        for keyword, indicator in matched[:2]:
            data = _safe_get(
                f"https://api.worldbank.org/v2/country/WLD/indicator/{indicator}",
                params={"format": "json", "per_page": "1", "mrv": "1"},
            )
            if data and isinstance(data, list) and len(data) > 1 and data[1]:
                rec = data[1][0]
                value = rec.get("value")
                if value is not None:
                    ind_name = rec.get("indicator", {}).get("value", keyword)
                    lines.append(f"  - {ind_name}: {value:,.2f} ({rec.get('date', '?')})")
        return ("**Global Data (World Bank):**\n" + "\n".join(lines)) if lines else ""
    except Exception as e:
        logger.warning(f"World Bank fetch failed: {e}")
        return ""


def fetch_wikipedia_context(query_text: str) -> str:
    try:
        keywords = " ".join(query_text.split()[:6])
        search_data = _safe_get(
            "https://en.wikipedia.org/w/api.php",
            params={"action": "query", "list": "search", "srsearch": keywords,
                    "srlimit": "1", "format": "json"},
        )
        if not search_data or "query" not in search_data:
            return ""
        results = search_data["query"].get("search", [])
        if not results:
            return ""
        page_title = results[0]["title"]
        extract_data = _safe_get(
            "https://en.wikipedia.org/w/api.php",
            params={"action": "query", "titles": page_title, "prop": "extracts",
                    "exintro": "true", "explaintext": "true", "format": "json"},
        )
        if not extract_data or "query" not in extract_data:
            return ""
        pages = extract_data["query"].get("pages", {})
        for page_id, page_data in pages.items():
            extract = page_data.get("extract", "")
            if extract:
                extract = extract[:500] + ("..." if len(extract) > 500 else "")
                return f"**Background (Wikipedia - {page_title}):**\n  {extract}"
        return ""
    except Exception as e:
        logger.warning(f"Wikipedia fetch failed: {e}")
        return ""


def fetch_who_health_data(query_text: str) -> str:
    health_keywords = ["disease", "health", "pandemic", "virus", "vaccine", "mortality",
                       "covid", "infection", "epidemic", "WHO", "measles", "malaria",
                       "tuberculosis", "hiv", "flu", "influenza", "bird flu", "mpox",
                       "ebola", "cholera", "polio", "death rate", "life expectancy"]
    if not any(kw in query_text.lower() for kw in health_keywords):
        return ""
    try:
        keywords = " ".join(query_text.split()[:4])
        data = _safe_get(
            "https://ghoapi.azureedge.net/api/Indicator",
            params={"$filter": f"contains(IndicatorName,'{keywords}')"},
        )
        if not data or "value" not in data or not data["value"]:
            return ""
        lines = []
        for ind in data["value"][:2]:
            code = ind.get("IndicatorCode", "")
            name = ind.get("IndicatorName", "")
            obs_data = _safe_get(
                f"https://ghoapi.azureedge.net/api/{code}",
                params={"$filter": "SpatialDim eq 'GLOBAL'", "$top": "1", "$orderby": "TimeDim desc"},
            )
            if obs_data and "value" in obs_data and obs_data["value"]:
                obs = obs_data["value"][0]
                lines.append(f"  - {name}: {obs.get('NumericValue', 'N/A')} ({obs.get('TimeDim', '?')})")
            else:
                lines.append(f"  - {name} (indicator: {code})")
        return ("**Global Health (WHO GHO):**\n" + "\n".join(lines)) if lines else ""
    except Exception as e:
        logger.warning(f"WHO GHO fetch failed: {e}")
        return ""


def fetch_acled_conflict_data(query_text: str) -> str:
    acled_key = os.environ.get("ACLED_API_KEY", "")
    acled_email = os.environ.get("ACLED_EMAIL", "")
    if not acled_key or not acled_email:
        return ""
    conflict_keywords = ["war", "conflict", "military", "attack", "violence", "troops",
                         "invasion", "battle", "ceasefire", "peace", "coup", "protest",
                         "riot", "insurgent", "terrorism", "rebel", "armed"]
    if not any(kw in query_text.lower() for kw in conflict_keywords):
        return ""
    try:
        data = _safe_get(
            "https://api.acleddata.com/acled/read",
            params={"key": acled_key, "email": acled_email, "limit": "5",
                    "fields": "event_date|country|event_type|fatalities",
                    "order": "event_date:desc"},
        )
        if not data or "data" not in data or not data["data"]:
            return ""
        lines = []
        for ev in data["data"][:5]:
            lines.append(f"  - {ev.get('event_date','?')}: {ev.get('event_type','?')} in {ev.get('country','?')} ({ev.get('fatalities','?')} fatalities)")
        return "**Recent Conflict (ACLED):**\n" + "\n".join(lines)
    except Exception as e:
        logger.warning(f"ACLED fetch failed: {e}")
        return ""


def fetch_media_trend(query_text: str) -> str:
    import requests
    try:
        params = {"query": query_text[:60].replace('"',''), "mode": "TimelineVol",
                  "format": "json", "TIMESPAN": "6m"}
        resp = requests.get("https://api.gdeltproject.org/api/v2/doc/doc", params=params, timeout=12)
        if resp.status_code != 200 or "json" not in resp.headers.get("Content-Type", ""):
            return ""
        data = resp.json()
        timeline = data.get("timeline", [])
        if not timeline or not timeline[0].get("data"):
            return ""
        series = timeline[0]["data"]
        if len(series) < 4:
            return ""
        recent_avg = sum(d.get("value", 0) for d in series[-6:]) / max(len(series[-6:]), 1)
        older_avg = sum(d.get("value", 0) for d in series[:6]) / max(len(series[:6]), 1)
        if older_avg > 0:
            change = ((recent_avg - older_avg) / older_avg) * 100
            direction = "increasing" if change > 10 else "decreasing" if change < -10 else "stable"
            return f"**Media Attention Trend:** {direction} ({change:+.0f}% over 6 months)"
        return ""
    except Exception as e:
        logger.warning(f"Media trend fetch failed: {e}")
        return ""


###############################################################################
# MAIN BOT CLASS
###############################################################################

class DanDanDonkeyBot(ForecastBot):
    """
    DanDanDonkeyBot v2 - with inside/outside view research and multi-source data.
    Free data sources: GDELT, Polymarket, Metaculus community, FRED, World Bank,
    Wikipedia, WHO GHO, ACLED, GDELT media trends.
    """

    _max_concurrent_questions = 1
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)
    _structure_output_validation_samples = 2

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            q_text = question.question_text

            # Gather all free data sources
            sources = [
                fetch_metaculus_community(question),
                fetch_polymarket_data(q_text),
                fetch_gdelt_articles(q_text),
                fetch_fred_data(q_text),
                fetch_world_bank_data(q_text),
                fetch_who_health_data(q_text),
                fetch_acled_conflict_data(q_text),
                fetch_wikipedia_context(q_text),
                fetch_media_trend(q_text),
            ]
            external_data = "\n\n".join(s for s in sources if s)
            if not external_data:
                external_data = "(No external data sources returned results.)"

            # Inside/Outside view structured research prompt
            research_prompt = clean_indents(
                f"""
                You are a research assistant to a superforecaster. Produce a structured
                research brief covering BOTH the outside view and inside view.

                ## Question
                {question.question_text}

                ## Resolution Criteria
                {question.resolution_criteria}

                {question.fine_print}

                ## External Data Gathered
                {external_data}

                ## Your Task
                Produce a brief with EXACTLY these sections:

                ### OUTSIDE VIEW (Base rates & reference classes)
                - Historical base rate for events like this
                - Reference class this question belongs to
                - Prediction market prices (if data above)
                - Metaculus community prediction (if data above)
                - Analogous past situations

                ### INSIDE VIEW (Specific current evidence)
                - Recent developments affecting this question
                - Key causal factors driving the outcome
                - Relevant data points from sources above
                - Upcoming events or deadlines that matter

                ### KEY UNCERTAINTIES
                - 2-3 biggest unknowns that could swing the outcome
                - Where outside view and inside view disagree

                ### SYNTHESIS
                - Brief overall assessment integrating both views
                - Do NOT give a probability

                Be concise. Cite specific data from the external data above.
                """
            )

            researcher = self.get_llm("researcher")
            if isinstance(researcher, GeneralLlm):
                research = await researcher.invoke(research_prompt)
            elif isinstance(researcher, str) and (
                researcher.startswith("asknews/")
            ):
                research = await AskNewsSearcher().call_preconfigured_version(
                    researcher, research_prompt
                )
            elif isinstance(researcher, str) and researcher.startswith("smart-searcher"):
                model_name = researcher.removeprefix("smart-searcher/")
                searcher = SmartSearcher(
                    model=model_name, temperature=0,
                    num_searches_to_run=2, num_sites_per_search=10,
                    use_advanced_filters=False,
                )
                research = await searcher.invoke(research_prompt)
            elif not researcher or researcher == "None" or researcher == "no_research":
                research = external_data
            else:
                research = await self.get_llm("researcher", "llm").invoke(research_prompt)

            logger.info(f"Research for {question.page_url}:\n{research[:500]}...")
            return research

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Question background:
            {question.background_info}

            Resolution criteria (not yet satisfied):
            {question.resolution_criteria}

            {question.fine_print}

            Your research assistant's structured brief:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Before answering you write:
            (a) The time left until the outcome is known.
            (b) The status quo outcome if nothing changed.
            (c) The OUTSIDE VIEW base rate -- what % of similar situations historically resolved Yes?
            (d) The INSIDE VIEW adjustment -- what specific current evidence shifts the probability?
            (e) A brief No scenario.
            (f) A brief Yes scenario.
            (g) If prediction market or community forecasts are available, note them and whether you agree.

            Rationale guidelines:
            - Anchor on the outside view base rate, then adjust with inside view evidence.
            - Put extra weight on the status quo.
            - Respect prediction market and community forecasts as informative priors.
            {self._get_conditional_disclaimer_if_necessary(question)}

            The last thing you write is your final answer as: "Probability: ZZ%", 0-100
            """
        )
        return await self._binary_prompt_to_forecast(question, prompt)

    async def _binary_prompt_to_forecast(self, question: BinaryQuestion, prompt: str) -> ReasonedPrediction[float]:
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        logger.info(f"Reasoning for {question.page_url}: {reasoning}")
        binary_prediction: BinaryPrediction = await structure_output(
            reasoning, BinaryPrediction,
            model=self.get_llm("parser", "llm"),
            num_validation_samples=self._structure_output_validation_samples,
        )
        decimal_pred = max(0.01, min(0.99, binary_prediction.prediction_in_decimal))
        logger.info(f"Forecasted {question.page_url}: {decimal_pred}")
        return ReasonedPrediction(prediction_value=decimal_pred, reasoning=reasoning)

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            The options are: {question.options}

            Background:
            {question.background_info}

            {question.resolution_criteria}
            {question.fine_print}

            Your research assistant's structured brief:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Before answering you write:
            (a) Time left until outcome is known.
            (b) Status quo outcome if nothing changed.
            (c) OUTSIDE VIEW -- base rates, prediction markets, community forecasts.
            (d) INSIDE VIEW -- specific current evidence favoring certain options.
            (e) An unexpected outcome scenario.

            {self._get_conditional_disclaimer_if_necessary(question)}
            Remember: (1) anchor on base rates, (2) weight status quo, (3) leave moderate probability on most options.

            Final probabilities for options {question.options} as:
            Option_A: Probability_A
            Option_B: Probability_B
            ...
            Option_N: Probability_N
            """
        )
        return await self._multiple_choice_prompt_to_forecast(question, prompt)

    async def _multiple_choice_prompt_to_forecast(self, question: MultipleChoiceQuestion, prompt: str) -> ReasonedPrediction[PredictedOptionList]:
        parsing_instructions = clean_indents(
            f"""
            Make sure all option names are one of: {question.options}
            Remove any "Option" prefix not part of the actual option names.
            Do not skip 0% options -- include them with 0% probability.
            """
        )
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        logger.info(f"Reasoning for {question.page_url}: {reasoning}")
        predicted_option_list: PredictedOptionList = await structure_output(
            text_to_structure=reasoning, output_type=PredictedOptionList,
            model=self.get_llm("parser", "llm"),
            num_validation_samples=self._structure_output_validation_samples,
            additional_instructions=parsing_instructions,
        )
        logger.info(f"Forecasted {question.page_url}: {predicted_option_list}")
        return ReasonedPrediction(prediction_value=predicted_option_list, reasoning=reasoning)

    async def _run_forecast_on_numeric(self, question: NumericQuestion, research: str) -> ReasonedPrediction[NumericDistribution]:
        upper_msg, lower_msg = self._create_upper_and_lower_bound_messages(question)
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Background: {question.background_info}
            {question.resolution_criteria}
            {question.fine_print}
            Units: {question.unit_of_measure if question.unit_of_measure else "Not stated (infer)"}

            Research brief:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.
            {lower_msg}
            {upper_msg}

            Formatting: Use correct units. No scientific notation. Percentile 10 < 20 < 40 < 60 < 80 < 90.

            Before answering:
            (a) Time left.
            (b) OUTSIDE VIEW -- historical base rates.
            (c) INSIDE VIEW -- current specific evidence.
            (d) Expert/market expectations.
            (e) Low outcome scenario.
            (f) High outcome scenario.

            {self._get_conditional_disclaimer_if_necessary(question)}
            Set wide 90/10 confidence intervals for unknown unknowns.

            Final answer:
            "
            Percentile 10: XX
            Percentile 20: XX
            Percentile 40: XX
            Percentile 60: XX
            Percentile 80: XX
            Percentile 90: XX
            "
            """
        )
        return await self._numeric_prompt_to_forecast(question, prompt)

    async def _numeric_prompt_to_forecast(self, question: NumericQuestion, prompt: str) -> ReasonedPrediction[NumericDistribution]:
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        logger.info(f"Reasoning for {question.page_url}: {reasoning}")
        parsing_instructions = clean_indents(
            f"""
            Parsing a numeric forecast for: "{question.question_text}".
            Units: {question.unit_of_measure}. Example range: {question.lower_bound} to {question.upper_bound}.
            Parse values in correct units. Convert scientific notation to regular numbers.
            If percentiles not explicitly given, indicate that.
            """
        )
        percentile_list: list[Percentile] = await structure_output(
            reasoning, list[Percentile],
            model=self.get_llm("parser", "llm"),
            additional_instructions=parsing_instructions,
            num_validation_samples=self._structure_output_validation_samples,
        )
        prediction = NumericDistribution.from_question(percentile_list, question)
        logger.info(f"Forecasted {question.page_url}: {prediction.declared_percentiles}")
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    async def _run_forecast_on_date(self, question: DateQuestion, research: str) -> ReasonedPrediction[NumericDistribution]:
        upper_msg, lower_msg = self._create_upper_and_lower_bound_messages(question)
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Question: {question.question_text}
            Background: {question.background_info}
            {question.resolution_criteria}
            {question.fine_print}

            Research brief:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.
            {lower_msg}
            {upper_msg}

            Format: YYYY-MM-DD. Chronological order (earliest first).

            Before answering:
            (a) Time left. (b) OUTSIDE VIEW. (c) INSIDE VIEW.
            (d) Expert expectations. (e) Early scenario. (f) Late scenario.

            {self._get_conditional_disclaimer_if_necessary(question)}
            Set wide confidence intervals.

            Final answer:
            "
            Percentile 10: YYYY-MM-DD
            Percentile 20: YYYY-MM-DD
            Percentile 40: YYYY-MM-DD
            Percentile 60: YYYY-MM-DD
            Percentile 80: YYYY-MM-DD
            Percentile 90: YYYY-MM-DD
            "
            """
        )
        return await self._date_prompt_to_forecast(question, prompt)

    async def _date_prompt_to_forecast(self, question: DateQuestion, prompt: str) -> ReasonedPrediction[NumericDistribution]:
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        logger.info(f"Reasoning for {question.page_url}: {reasoning}")
        parsing_instructions = clean_indents(
            f"""
            Parsing a date forecast for: "{question.question_text}".
            Range: {question.lower_bound} to {question.upper_bound}.
            Format dates as valid datetime strings. Assume midnight UTC if no hour given.
            If percentiles not explicitly given, indicate that.
            """
        )
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
        logger.info(f"Forecasted {question.page_url}: {prediction.declared_percentiles}")
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    def _create_upper_and_lower_bound_messages(self, question: NumericQuestion | DateQuestion) -> tuple[str, str]:
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
        upper_msg = (f"The question creator thinks the number is likely not higher than {upper} {unit}."
                     if question.open_upper_bound else f"The outcome can not be higher than {upper} {unit}.")
        lower_msg = (f"The question creator thinks the number is likely not lower than {lower} {unit}."
                     if question.open_lower_bound else f"The outcome can not be lower than {lower} {unit}.")
        return upper_msg, lower_msg

    async def _run_forecast_on_conditional(self, question: ConditionalQuestion, research: str) -> ReasonedPrediction[ConditionalPrediction]:
        parent_info, full_research = await self._get_question_prediction_info(question.parent, research, "parent")
        child_info, full_research = await self._get_question_prediction_info(question.child, research, "child")
        yes_info, full_research = await self._get_question_prediction_info(question.question_yes, full_research, "yes")
        no_info, full_research = await self._get_question_prediction_info(question.question_no, full_research, "no")
        full_reasoning = clean_indents(f"""
            ## Parent Question Reasoning
            {parent_info.reasoning}
            ## Child Question Reasoning
            {child_info.reasoning}
            ## Yes Question Reasoning
            {yes_info.reasoning}
            ## No Question Reasoning
            {no_info.reasoning}
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
                return (ReasonedPrediction(prediction_value=PredictionAffirmed(),
                        reasoning=f"Reaffirmed at {pretty}."), research)
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
        return clean_indents("""
            You are forecasting the CHILD question given the parent's resolution.
            Never re-forecast the parent question.
        """)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    logging.getLogger("LiteLLM").propagate = False

    parser = argparse.ArgumentParser(description="Run DanDanDonkeyBot v2")
    parser.add_argument("--mode", type=str, choices=["tournament", "metaculus_cup", "test_questions"], default="tournament")
    args = parser.parse_args()
    run_mode: Literal["tournament", "metaculus_cup", "test_questions"] = args.mode

    danbot = DanDanDonkeyBot(
        research_reports_per_question=1,
        predictions_per_research_report=5,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=False,
        folder_to_save_reports_to="reports",
        skip_previously_forecasted_questions=True,
        extra_metadata_in_explanation=True,
        llms={
            "default": GeneralLlm(model="openrouter/deepseek/deepseek-r1", temperature=0.3, timeout=300, allowed_tries=2),
            "summarizer": "openrouter/deepseek/deepseek-chat",
            "researcher": GeneralLlm(model="openrouter/deepseek/deepseek-r1", temperature=0.3, timeout=300, allowed_tries=2),
            "parser": "openrouter/deepseek/deepseek-chat",
        },
    )

    client = MetaculusClient()
    if run_mode == "tournament":
        reports1 = asyncio.run(danbot.forecast_on_tournament(client.CURRENT_AI_COMPETITION_ID, return_exceptions=True))
        reports2 = asyncio.run(danbot.forecast_on_tournament(client.CURRENT_MINIBENCH_ID, return_exceptions=True))
        forecast_reports = reports1 + reports2
    elif run_mode == "metaculus_cup":
        danbot.skip_previously_forecasted_questions = False
        forecast_reports = asyncio.run(danbot.forecast_on_tournament(client.CURRENT_METACULUS_CUP_ID, return_exceptions=True))
    elif run_mode == "test_questions":
        EXAMPLE_QUESTIONS = [
            "https://www.metaculus.com/questions/578/human-extinction-by-2100/",
            "https://www.metaculus.com/questions/22427/number-of-new-leading-ai-labs/",
        ]
        danbot.skip_previously_forecasted_questions = False
        questions = [client.get_question_by_url(url) for url in EXAMPLE_QUESTIONS]
        forecast_reports = asyncio.run(danbot.forecast_questions(questions, return_exceptions=True))
    danbot.log_report_summary(forecast_reports)
