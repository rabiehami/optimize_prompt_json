"""LLM API call wrappers using litellm."""

import asyncio
import json
import logging
from datetime import datetime, timezone
from uuid import uuid4

import litellm

from optimize_prompt_json import (
    PROMPT_TYPE_BASELINE_EXTRACTION,
    PROMPT_TYPE_JSON_EXTRACTION,
    PROMPT_TYPE_JSON_GENERATION,
)
from optimize_prompt_json.database import LLMResponse, get_session
from optimize_prompt_json.json_utils import extract_json_from_response

litellm.drop_params = True
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

_JSON_PROMPT_TYPES = {
    PROMPT_TYPE_JSON_GENERATION,
    PROMPT_TYPE_JSON_EXTRACTION,
    PROMPT_TYPE_BASELINE_EXTRACTION,
}


def _safe_completion_cost(response, model):
    """Return litellm completion cost, or 0.0 on failure."""
    try:
        return litellm.completion_cost(response, model=model)
    except Exception:
        return 0.0


async def ask_model(rate_limit_delay=0.0, **kwargs):
    """Make an async LLM completion request with retry and JSON-mode support."""
    MAX_RETRIES = 5
    INITIAL_BACKOFF = 1.0

    llm_model = kwargs.get("llm_model", "")
    api_key = kwargs.get("api_key", "")

    if rate_limit_delay > 0:
        await asyncio.sleep(rate_limit_delay)

    use_json_mode = kwargs.get("prompt_type") in _JSON_PROMPT_TYPES
    response_format = {"type": "json_object"} if use_json_mode else None

    prompt_content = kwargs["prompt"]
    messages = [{"role": "user", "content": prompt_content}]
    if use_json_mode and "json" not in prompt_content.lower():
        messages = [{"role": "system", "content": "Respond with valid JSON."}] + messages

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = await litellm.acompletion(
                model=llm_model,
                messages=messages,
                api_key=api_key,
                temperature=kwargs["llm_temperature"],
                response_format=response_format,
            )
            break
        except litellm.exceptions.RateLimitError as e:
            if attempt == MAX_RETRIES:
                raise
            backoff = INITIAL_BACKOFF * (2 ** (attempt - 1))
            logger.warning(f"Rate limit (attempt {attempt}/{MAX_RETRIES}), retrying in {backoff:.1f}s: {e}")
            await asyncio.sleep(backoff)
        except litellm.exceptions.BadRequestError as e:
            if "json_validate_failed" in str(e):
                logger.warning("JSON mode failed — retrying without JSON mode.")
                response = await litellm.acompletion(
                    model=llm_model,
                    messages=messages,
                    api_key=api_key,
                    temperature=kwargs["llm_temperature"],
                    response_format=None,
                )
                break
            raise

    full_response = {
        "prompt_type": kwargs.get("prompt_type"),
        "prompt": kwargs.get("prompt"),
        "content": response.choices[0].message.content,
        "completion_tokens": response.usage.completion_tokens,
        "prompt_tokens": response.usage.prompt_tokens,
        "total_tokens": response.usage.total_tokens,
        "created": datetime.fromtimestamp(response.created, timezone.utc),
        "json": None,
        "price": _safe_completion_cost(response, llm_model),
    }
    for k, v in kwargs.items():
        if k not in ("prompt_type", "prompt", "api_key"):
            full_response[k] = v

    logger.info(f"ask_model: {full_response}")
    return full_response


async def parallel_requests(prompts, meta, rate_limit_delay=0.0):
    """Execute multiple LLM requests in parallel and store results in DB."""
    tasks = [
        ask_model(prompt=p, rate_limit_delay=rate_limit_delay, **m)
        for p, m in zip(prompts, meta)
    ]
    results = await asyncio.gather(*tasks, return_exceptions=False)

    session = get_session()
    for r in results:
        if r["prompt_type"] == PROMPT_TYPE_JSON_GENERATION:
            try:
                r["json"] = json.dumps(extract_json_from_response(r["content"]), indent=2)
            except Exception:
                r["json"] = None
        session.add(LLMResponse(**r))
    session.commit()
    session.close()
    return results
