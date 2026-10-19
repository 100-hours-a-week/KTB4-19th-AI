import asyncio
import logging
import re
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from functools import lru_cache
from urllib.parse import urlparse

from openai import (
    APIError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
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
from zipsai.metrics import observe_llm
from zipsai.observability import current_stage, note_llm_tokens
from zipsai.settings import get_settings
from zipsai.tracing import get_tracing_client

logger = logging.getLogger(__name__)

_REQUIRE_STRUCTURED_OUTPUTS = {"provider": {"require_parameters": True}}
_SECRET_HOST = re.compile(r"(^sk-|[A-Fa-f0-9]{32})")


def provider_host(base_url: str | None) -> str | None:
    """base_url의 호스트만 남긴다. 경로의 엔드포인트 아이디는 싣지 않는다."""
    if not base_url:
        return None
    host = urlparse(base_url).hostname
    if not host or _SECRET_HOST.search(host):
        return None if not host else "[REDACTED]"
    return host


def _hangul_ratio(text: str) -> float | None:
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return None
    hangul = sum(1 for char in letters if "\uac00" <= char <= "\ud7a3")
    return round(hangul / len(letters), 3)


def _token_map(usage: object | None) -> dict[str, int | None]:
    if usage is None:
        return {}
    details = getattr(usage, "completion_tokens_details", None)
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    return {
        "input": getattr(usage, "prompt_tokens", None),
        "output": getattr(usage, "completion_tokens", None),
        "cached": getattr(prompt_details, "cached_tokens", None) if prompt_details else None,
        "reasoning": getattr(details, "reasoning_tokens", None) if details else None,
    }


def _reason(value: object) -> str | None:
    if value is None:
        return None
    raw = getattr(value, "value", value)
    return str(raw)


@dataclass
class LlmCall:
    """호출 1회의 시간·토큰만 남긴다. 프롬프트와 응답 원문은 싣지 않는다."""

    purpose: str
    model: str
    base_url: str | None
    streaming: bool
    started: float
    structured: bool
    image_count: int = 0
    attempt: int = 1
    retry_reason: str | None = None
    ttft_ms: int | None = None
    usage: object | None = None
    done: bool = False

    def mark_ttft(self) -> None:
        if self.ttft_ms is None:
            self.ttft_ms = int((time.perf_counter() - self.started) * 1000)

    def finish(
        self,
        *,
        outcome: str,
        status_code: int | None = None,
        error_code: str | None = None,
        finish_reason: str | None = None,
        structured_output_ok: bool | None = None,
        output_text: str | None = None,
    ) -> None:
        if self.done:
            return
        self.done = True
        tokens = _token_map(self.usage)
        latency_ms = int((time.perf_counter() - self.started) * 1000)
        think_tag_present = None
        hangul_ratio = None
        if output_text is not None:
            think_tag_present = "<think>" in output_text.lower()
            hangul_ratio = _hangul_ratio(output_text)
        try:
            note_llm_tokens(tokens)
            observe_llm(
                purpose=self.purpose,
                model=self.model,
                outcome=outcome,
                duration_seconds=latency_ms / 1000,
                ttft_seconds=None if self.ttft_ms is None else self.ttft_ms / 1000,
                tokens=tokens,
                cost_usd=None,
                structured_output_ok=structured_output_ok,
            )
            logger.info(
                "llm_call",
                extra={
                    "purpose": self.purpose,
                    "outcome": outcome,
                    "provider_host": provider_host(self.base_url),
                    "model": self.model,
                    "attempt": self.attempt,
                    "streaming": self.streaming,
                    "latency_ms": latency_ms,
                    "ttft_ms": self.ttft_ms,
                    "input_tokens": tokens.get("input"),
                    "output_tokens": tokens.get("output"),
                    "cached_tokens": tokens.get("cached"),
                    "reasoning_tokens": tokens.get("reasoning"),
                    "finish_reason": finish_reason,
                    "status_code": status_code,
                    "error_code": error_code,
                    "structured_output_ok": structured_output_ok,
                    "retry_reason": self.retry_reason,
                    "image_count": self.image_count,
                    "image_pixels_total": None,
                    "cost_usd": None,
                    "think_tag_present": think_tag_present,
                    "hangul_ratio": hangul_ratio,
                    **(
                        {"usage_missing": True}
                        if self.streaming and self.usage is None
                        else {}
                    ),
                },
            )
        except Exception:  # noqa: BLE001 로그 실패가 답변을 막으면 안 된다.
            return


def start_llm_call(
    *,
    purpose: str | None,
    model: str,
    base_url: str | None,
    streaming: bool,
    structured: bool,
    image_count: int = 0,
    attempt: int = 1,
    retry_reason: str | None = None,
) -> LlmCall:
    return LlmCall(
        purpose=purpose or current_stage() or "unknown",
        model=model,
        base_url=base_url,
        streaming=streaming,
        started=time.perf_counter(),
        structured=structured,
        image_count=image_count,
        attempt=attempt,
        retry_reason=retry_reason,
    )


def generate_text(
    system_prompt: str,
    user_prompt: str,
    response_format: dict | None = None,
    *,
    usage_sink: dict[str, object] | None = None,
    purpose: str | None = None,
    attempt: int = 1,
    retry_reason: str | None = None,
) -> str:
    settings = get_settings()
    call = start_llm_call(
        purpose=purpose,
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        streaming=False,
        structured=response_format is not None,
        attempt=attempt,
        retry_reason=retry_reason,
    )
    outcome = "fail"
    status_code: int | None = None
    err_code: str | None = None
    finish_reason: str | None = None
    structured_ok: bool | None = True if response_format else None
    output_text: str | None = None
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
        call.usage = response.usage
        _record_usage(usage_sink, response.usage)
        content = response.choices[0].message.content if response.choices else None
        if not content:
            outcome = "fail"
            err_code = "DEPENDENCY_NOT_READY"
            raise LlmUnavailableError("LLM returned an empty response")
        output_text = content
        finish_reason = _reason(getattr(response.choices[0], "finish_reason", None))
        outcome = "ok"
        status_code = 200
        return content
    except RateLimitError as error:
        outcome, status_code, err_code = "fail", 429, "MODEL_RATE_LIMITED"
        raise LlmRateLimitedError(
            "LLM rate limit exceeded", retry_after_seconds=_retry_after(error)
        ) from error
    except APITimeoutError as error:
        outcome, status_code, err_code = "fail", None, "MODEL_TIMEOUT"
        raise LlmTimeoutError("LLM request timed out") from error
    except APIStatusError as error:
        outcome = "fail"
        status_code = error.status_code
        err_code = "MODEL_UPSTREAM_ERROR" if error.status_code >= 500 else "DEPENDENCY_NOT_READY"
        if error.status_code >= 500:
            raise LlmUpstreamError("LLM provider returned an upstream error") from error
        raise LlmUnavailableError("LLM request failed") from error
    except APIError as error:
        outcome, status_code, err_code = "fail", None, "DEPENDENCY_NOT_READY"
        raise LlmUnavailableError("LLM request failed") from error
    except Exception as error:
        outcome = "fail"
        if err_code is None:
            status_code = None
            err_code = type(error).__name__
        raise
    finally:
        call.finish(
            outcome=outcome,
            status_code=status_code,
            error_code=err_code,
            finish_reason=finish_reason,
            structured_output_ok=structured_ok,
            output_text=output_text,
        )


async def generate_text_stream(
    system_prompt: str,
    user_prompt: str,
    *,
    usage_sink: dict[str, object] | None = None,
    purpose: str | None = None,
    attempt: int = 1,
) -> AsyncIterator[str]:
    settings = get_settings()
    call = start_llm_call(
        purpose=purpose,
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        streaming=True,
        structured=False,
        attempt=attempt,
    )
    outcome = "fail"
    status_code: int | None = None
    err_code: str | None = None
    parts: list[str] = []
    try:
        client = _get_async_client(
            settings.llm_api_key, settings.llm_base_url, settings.llm_timeout_seconds
        )
        stream = await client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            stream=True,
            stream_options={"include_usage": True},
        )
        async for chunk in stream:
            usage = getattr(chunk, "usage", None)
            if usage is not None:
                call.usage = usage
                _record_usage(usage_sink, usage)
            if not chunk.choices:
                continue
            text = chunk.choices[0].delta.content
            if text:
                call.mark_ttft()
                parts.append(text)
                yield text
        outcome = "ok"
        status_code = 200
    except (asyncio.CancelledError, GeneratorExit):
        outcome = "cancelled"
        status_code = None
        raise
    except RateLimitError as error:
        outcome, status_code, err_code = "fail", 429, "MODEL_RATE_LIMITED"
        raise LlmRateLimitedError(
            "LLM rate limit exceeded", retry_after_seconds=_retry_after(error)
        ) from error
    except APITimeoutError as error:
        outcome, status_code, err_code = "fail", None, "MODEL_TIMEOUT"
        raise LlmTimeoutError("LLM request timed out") from error
    except APIStatusError as error:
        outcome = "fail"
        status_code = error.status_code
        err_code = "MODEL_UPSTREAM_ERROR" if error.status_code >= 500 else "DEPENDENCY_NOT_READY"
        if error.status_code >= 500:
            raise LlmUpstreamError("LLM provider returned an upstream error") from error
        raise LlmUnavailableError("LLM request failed") from error
    except APIError as error:
        outcome, status_code, err_code = "fail", None, "DEPENDENCY_NOT_READY"
        raise LlmUnavailableError("LLM request failed") from error
    except Exception as error:
        outcome = "fail"
        if err_code is None:
            status_code = None
            err_code = type(error).__name__
        raise
    finally:
        call.finish(
            outcome=outcome,
            status_code=status_code,
            error_code=err_code,
            output_text="".join(parts) or None,
        )


def generate_structured[T: BaseModel](
    system_prompt: str,
    user_prompt: str,
    response_format: type[T],
    *,
    model: str | None = None,
    usage_sink: dict[str, object] | None = None,
    purpose: str | None = None,
    attempt: int = 1,
    retry_reason: str | None = None,
) -> T | None:
    settings = get_settings()
    chosen = model if model is not None else settings.llm_model
    call = start_llm_call(
        purpose=purpose,
        model=chosen,
        base_url=settings.llm_base_url,
        streaming=False,
        structured=True,
        attempt=attempt,
        retry_reason=retry_reason,
    )
    outcome = "fail"
    status_code: int | None = None
    err_code: str | None = None
    finish_reason: str | None = None
    structured_ok: bool | None = True
    try:
        response = _get_client(
            settings.llm_api_key, settings.llm_base_url, settings.llm_timeout_seconds
        ).chat.completions.parse(
            model=chosen,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format=response_format,
            extra_body=_REQUIRE_STRUCTURED_OUTPUTS,
        )
        call.usage = response.usage
        _record_usage(usage_sink, response.usage)
        if not response.choices:
            outcome = "fail"
            err_code = "DEPENDENCY_NOT_READY"
            structured_ok = False
            raise LlmUnavailableError("LLM returned an empty response")
        message = response.choices[0].message
        finish_reason = _reason(getattr(response.choices[0], "finish_reason", None))
        if message.refusal or message.parsed is None:
            outcome = "fail"
            structured_ok = False
            err_code = "structured_output"
            return None
        outcome = "ok"
        status_code = 200
        return message.parsed
    except RateLimitError as error:
        outcome, status_code, err_code, structured_ok = "fail", 429, "MODEL_RATE_LIMITED", None
        raise LlmRateLimitedError(
            "LLM rate limit exceeded", retry_after_seconds=_retry_after(error)
        ) from error
    except APITimeoutError as error:
        outcome, status_code, err_code, structured_ok = "fail", None, "MODEL_TIMEOUT", None
        raise LlmTimeoutError("LLM request timed out") from error
    except APIStatusError as error:
        outcome = "fail"
        status_code = error.status_code
        structured_ok = None
        err_code = "MODEL_UPSTREAM_ERROR" if error.status_code >= 500 else "DEPENDENCY_NOT_READY"
        if error.status_code >= 500:
            raise LlmUpstreamError("LLM provider returned an upstream error") from error
        raise LlmUnavailableError("LLM request failed") from error
    except (LengthFinishReasonError, ContentFilterFinishReasonError) as error:
        outcome = "fail"
        structured_ok = False
        finish_reason = (
            "length" if isinstance(error, LengthFinishReasonError) else "content_filter"
        )
        err_code = type(error).__name__
        return None
    except APIError as error:
        outcome, status_code, err_code, structured_ok = "fail", None, "DEPENDENCY_NOT_READY", None
        raise LlmUnavailableError("LLM request failed") from error
    except Exception as error:
        outcome = "fail"
        if err_code is None:
            status_code = None
            err_code = type(error).__name__
            structured_ok = None
        raise
    finally:
        call.finish(
            outcome=outcome,
            status_code=status_code,
            error_code=err_code,
            finish_reason=finish_reason,
            structured_output_ok=structured_ok,
        )


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
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    cached_tokens = (
        getattr(prompt_details, "cached_tokens", None) if prompt_details else None
    )
    if cached_tokens is not None:
        sink["cached_tokens"] = cached_tokens


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


@lru_cache
def _get_async_client(
    api_key: str, base_url: str | None, timeout_seconds: float
) -> AsyncOpenAI:
    return AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout_seconds)


def _retry_after(error: RateLimitError) -> int | None:
    header = error.response.headers.get("retry-after")
    try:
        return int(float(header)) if header is not None else None
    except ValueError:
        return None
