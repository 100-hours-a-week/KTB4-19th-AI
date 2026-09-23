from functools import lru_cache

from openai import (
    APIError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)

from zipsai.errors import (
    LlmRateLimitedError,
    LlmTimeoutError,
    LlmUnavailableError,
    LlmUpstreamError,
)
from zipsai.settings import get_settings


def generate_text(system_prompt: str, user_prompt: str) -> str:
    settings = get_settings()
    try:
        response = _get_client(
            settings.llm_api_key, settings.llm_base_url, settings.llm_timeout_seconds
        ).chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
    except RateLimitError as error:
        raise LlmRateLimitedError(
            "LLM rate limit exceeded", retry_after_seconds=_retry_after(error)
        ) from error
    except APITimeoutError as error:
        raise LlmTimeoutError("LLM request timed out") from error
    except APIStatusError as error:
        if error.status_code >= 500:
            raise LlmUpstreamError("LLM provider returned an upstream error") from error
        raise LlmUnavailableError("LLM request failed") from error
    except APIError as error:
        raise LlmUnavailableError("LLM request failed") from error

    if not response.choices:
        raise LlmUnavailableError("LLM returned an empty response")

    content = response.choices[0].message.content
    if not content:
        raise LlmUnavailableError("LLM returned an empty response")
    return content


def strip_json_code_fence(content: str) -> str:
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = stripped.removeprefix("```json").removeprefix("```").strip()
        stripped = stripped.removesuffix("```").strip()
    return stripped


@lru_cache
def _get_client(api_key: str, base_url: str | None, timeout_seconds: float) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_seconds)


def _retry_after(error: RateLimitError) -> int | None:
    header = error.response.headers.get("retry-after")
    try:
        return int(float(header)) if header is not None else None
    except ValueError:
        return None
