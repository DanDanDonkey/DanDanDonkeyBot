import asyncio
import logging
import os
import re
from typing import Optional

from forecasting_tools import GeneralLlm

logger = logging.getLogger(__name__)

METACULUS_TOKEN = os.environ.get("METACULUS_TOKEN", "")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

###############################################################################
# CONSTANTS
###############################################################################

EXTREMIZATION_ALPHA      = 1.2
COMMUNITY_WEIGHT_DEFAULT = 0.40
COMMUNITY_WEIGHT_ECON    = 0.55
DEBATE_SPREAD_THRESHOLD  = 0.15

MAX_TOKENS_PHASE1    = 2000
MAX_TOKENS_PHASE2    = 1500
MAX_TOKENS_RESEARCH  = 1500
MAX_TOKENS_FORECAST  = 1500

_SRC_CAP_ASKNEWS   = 2500
_SRC_CAP_DEFAULT   = 600
_EXTERNAL_DATA_CAP = 6000

###############################################################################
# ENSEMBLE MODEL REGISTRY
###############################################################################

ENSEMBLE_MODELS = {
    "Claude-Sonnet":   "openrouter/anthropic/claude-sonnet-4.6",
    "GPT-4o":          "openrouter/openai/gpt-4o",
    "Llama4-Maverick": "openrouter/meta-llama/llama-4-maverick",
    "DeepSeek-R1":     "openrouter/deepseek/deepseek-r1",
    "Grok-4-Fast":     "openrouter/x-ai/grok-4-fast",
}

ENSEMBLE_NUMERIC_MODELS = {
    "Claude-Sonnet":   "openrouter/anthropic/claude-sonnet-4.6",
    "DeepSeek-R1":     "openrouter/deepseek/deepseek-r1",
    "Llama4-Maverick": "openrouter/meta-llama/llama-4-maverick",
}

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
    best_name, best_text, _ = scored[0]
    logger.info(f"Numeric ensemble: {len(valid)}/{len(ENSEMBLE_NUMERIC_MODELS)} responded. Best: {best_name}")
    return best_text
