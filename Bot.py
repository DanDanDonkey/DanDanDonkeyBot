import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

from forecasting_tools import (
    BinaryQuestion,
    ForecastBot,
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
    PredictedOptionList,
    ReasonedPrediction,
    clean_indents,
    structure_output,
)

from ensemble import (
    community_anchored_blend,
    get_community_weight,
    run_debate_ensemble,
    COMMUNITY_WEIGHT_DEFAULT,
)
from llm_calls import get_parser_llm, run_numeric_ensemble
from postmortem import log_prediction
from research import run_research

logger = logging.getLogger(__name__)


def _extract_community_float(question: MetaculusQuestion) -> Optional[float]:
    try:
        cp = question.community_prediction
        if isinstance(cp, float):
            return max(0.02, min(0.98, cp))
    except Exception:
        pass
    return None


class DanDanDonkeyBot(ForecastBot):

    _max_concurrent_questions = 1
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)
    _structure_output_validation_samples = 2

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            researcher_llm = self.get_llm("researcher")
            return await run_research(question, researcher_llm=researcher_llm)

    # -------------------------------------------------------------------------
    # BINARY
    # -------------------------------------------------------------------------

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

        community_pred   = _extract_community_float(question)
        community_weight = get_community_weight(question.question_text)
        is_econ          = community_weight > COMMUNITY_WEIGHT_DEFAULT

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

        log_prediction(
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

    # -------------------------------------------------------------------------
    # MULTIPLE CHOICE
    # -------------------------------------------------------------------------

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
        reasoning = await run_numeric_ensemble(prompt)
        if not reasoning:
            reasoning = await self.get_llm("default", "llm").invoke(prompt)
        predicted_option_list: PredictedOptionList = await structure_output(
            text_to_structure=reasoning, output_type=PredictedOptionList,
            model=self.get_llm("parser", "llm"),
            num_validation_samples=self._structure_output_validation_samples,
            additional_instructions=parsing_instructions,
        )
        return ReasonedPrediction(prediction_value=predicted_option_list, reasoning=reasoning)

    # -------------------------------------------------------------------------
    # NUMERIC
    # -------------------------------------------------------------------------

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
        reasoning = await run_numeric_ensemble(prompt)
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

    # -------------------------------------------------------------------------
    # DATE
    # -------------------------------------------------------------------------

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
        reasoning = await run_numeric_ensemble(prompt)
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

    # -------------------------------------------------------------------------
    # CONDITIONAL
    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # HELPERS
    # -------------------------------------------------------------------------

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
