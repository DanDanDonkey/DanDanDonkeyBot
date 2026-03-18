import asyncio
import logging
import os
from typing import Optional

from forecasting_tools import GeneralLlm

logger = logging.getLogger(__name__)

METACULUS_TOKEN = os.environ.get("METACULUS_TOKEN", "")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

###############################################################################
# ENSEMBLE MODEL REGISTRY
###############################################################################

ENSEMBLE_MODELS = {
    "DeepSeek-R1":     "openrouter/deepseek/deepseek-r1",
    "DeepSeek-V3":     "openrouter/meta-llama/llama-3.3-70b-instruct",
    "Qwen3-235B":      "openrouter/qwen/qwen3-235b-a22b",
    "Llama4-Maverick": "openrouter/meta-llama/llama-4-maverick",
    "Mistral-Small":   "openrouter/mistralai/mistral-small-3.1-24b-instruct",
}

ENSEMBLE_NUMERIC_MODELS = {
    "DeepSeek-R1":     "openrouter/deepseek/deepseek-r1",
    "Qwen3-235B":      "openrouter/qwen/qwen3-235b-a22b",
    "Llama4-Maverick": "openrouter/meta-llama/llama-4-maverick",
}

MAX_TOKENS_RESEARCH = 1500
MAX_TOKENS_FORECAST = 1500


###############################################################################
# SINGLE MODEL CALL
###############################################################################

async def call_model(
    model_id: str,
    prompt: str,
    temperature: float = 0.3,
    max_tokens: int = MAX_TOKENS_FORECAST,
) -> str:
    try:
        llm = GeneralLlm(
            model=model_id,
            temperature=temperature,
            timeout=180,
            allowed_tries=2,
            max_tokens=max_tokens,
        )
        return await llm.invoke(prompt)
    except Exception as e:
        logger.warning(f"Model {model_id} failed: {e}")
        return ""


###############################################################################
# NUMERIC ENSEMBLE (best-of-3 by structure score)
###############################################################################

async def run_numeric_ensemble(prompt: str) -> str:
    import re
    tasks = {
        name: call_model(model_id, prompt)
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

    scored = sorted(
        [(name, text, _count_percentile_lines(text)) for name, text in valid.items()],
        key=lambda x: x[2], reverse=True,
    )
    best_name, best_text, best_score = scored[0]
    logger.info(
        f"  Numeric ensemble: {len(valid)}/{len(ENSEMBLE_NUMERIC_MODELS)} responded. "
        f"Selected {best_name} (structure score: {best_score})"
    )
    return best_text


###############################################################################
# PARSER / SUMMARIZER LLM (lightweight, used by structure_output)
###############################################################################

def get_parser_llm() -> GeneralLlm:
    return GeneralLlm(
        model="openrouter/deepseek/deepseek-r1",
        temperature=0.1,
        timeout=120,
        allowed_tries=2,
    )


def get_researcher_llm() -> GeneralLlm:
    return GeneralLlm(
        model="openrouter/deepseek/deepseek-r1",
        temperature=0.2,
        timeout=120,
        allowed_tries=2,
        max_tokens=MAX_TOKENS_RESEARCH,
    )
