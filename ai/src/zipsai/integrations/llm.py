from functools import lru_cache

from openai import (
    APIError,
    APIStatusError,
    APITimeoutError,
    ContentFilterFinishReasonError,
    LengthFinishReasonError,
    OpenAI,
    RateLimitError,
)
from pydantic import BaseModel

from zipsai.errors import (
    LlmRateLimitedError,
    LlmTimeoutError,
    LlmUnavailableError,
    LlmUpstreamError,
)
from zipsai.settings import get_settings
from zipsai.tracing import get_tracing_client

_REQUIRE_STRUCTURED_OUTPUTS = {"provider": {"require_parameters": True}}


def generate_text(
    system_prompt: str,
    user_prompt: str,
    response_format: dict | None = None,
    *,
    usage_sink: dict[str, object] | None = None,
) -> str:
    settings = get_settings()
    try:
        client = _get_client(
            settings.llm_api_key, settings.llm_base_url, settings.llm_timeout_seconds
        )
        response = client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            **({"name": "generate-text"} if get_tracing_client() else {}),
            **(
                {
                    "response_format": response_format,
                    "extra_body": _REQUIRE_STRUCTURED_OUTPUTS,
                }
                if response_format
                else {}
            ),
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

    _record_usage(usage_sink, response.usage)

    if not response.choices:
        raise LlmUnavailableError("LLM returned an empty response")

    content = response.choices[0].message.content
    if not content:
        raise LlmUnavailableError("LLM returned an empty response")
    return content


def generate_structured[T: BaseModel](
    system_prompt: str,
    user_prompt: str,
    response_format: type[T],
    *,
    usage_sink: dict[str, object] | None = None,
) -> T | None:
    settings = get_settings()
    try:
        response = _get_client(
            settings.llm_api_key, settings.llm_base_url, settings.llm_timeout_seconds
        ).chat.completions.parse(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format=response_format,
            extra_body=_REQUIRE_STRUCTURED_OUTPUTS,
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
    except (LengthFinishReasonError, ContentFilterFinishReasonError):
        return None
    except APIError as error:
        raise LlmUnavailableError("LLM request failed") from error

    _record_usage(usage_sink, response.usage)

    if not response.choices:
        raise LlmUnavailableError("LLM returned an empty response")

    message = response.choices[0].message
    if message.refusal or message.parsed is None:
        return None
    return message.parsed


def _record_usage(sink: dict[str, object] | None, usage: object | None) -> None:
    if sink is None or usage is None:
        return
    sink["input_tokens"] = usage.prompt_tokens
    sink["output_tokens"] = usage.completion_tokens
    sink["total_tokens"] = usage.total_tokens
    details = getattr(usage, "completion_tokens_details", None)
    reasoning_tokens = getattr(details, "reasoning_tokens", None) if details else None
    if reasoning_tokens is not None:
        sink["reasoning_tokens"] = reasoning_tokens


def strip_json_code_fence(content: str) -> str:
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = stripped.removeprefix("```json").removeprefix("```").strip()
        stripped = stripped.removesuffix("```").strip()
    return stripped


@lru_cache
def _get_client(api_key: str, base_url: str | None, timeout_seconds: float) -> OpenAI:
    if get_tracing_client():
        from langfuse.openai import OpenAI as TracedOpenAI

        return TracedOpenAI(api_key=api_key, base_url=base_url, timeout=timeout_seconds)
    return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_seconds)


def _retry_after(error: RateLimitError) -> int | None:
    header = error.response.headers.get("retry-after")
    try:
        return int(float(header)) if header is not None else None
    except ValueError:
        return None
