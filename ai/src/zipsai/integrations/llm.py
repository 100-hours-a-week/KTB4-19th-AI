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

# OpenRouter는 같은 모델 슬러그도 여러 제공자로 라우팅한다. structured_outputs를
# 지원 안 하는 제공자로 넘어가면 스키마가 조용히 무시될 수 있어, 요청마다 강제한다.
# https://openrouter.ai/docs/guides/features/structured-outputs
_REQUIRE_STRUCTURED_OUTPUTS = {"provider": {"require_parameters": True}}


def generate_text(
    system_prompt: str,
    user_prompt: str,
    response_format: dict | None = None,
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
) -> T | None:
    """구조화 출력을 스키마 검증까지 마친 객체로 돌려준다.

    모델이 응답을 거부했거나(refusal) 길이 제한에 걸려 파싱할 내용이 없으면 None을
    돌려준다 — 전송 자체가 실패한 것과는 구분해, 호출자가 자신의 도메인 예외로
    승격할지 판단하게 한다.
    """
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

    if not response.choices:
        raise LlmUnavailableError("LLM returned an empty response")

    message = response.choices[0].message
    if message.refusal or message.parsed is None:
        return None
    return message.parsed


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
