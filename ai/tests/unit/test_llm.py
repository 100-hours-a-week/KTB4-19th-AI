from types import SimpleNamespace

import httpx
import pytest
from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

from zipsai.errors import (
    LlmRateLimitedError,
    LlmTimeoutError,
    LlmUnavailableError,
    LlmUpstreamError,
)
from zipsai.integrations import llm as llm_module
from zipsai.settings import Settings


class _FakeCompletions:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def create(self, **_kwargs: object) -> None:
        raise self.error


class _FakeClient:
    def __init__(self, error: Exception) -> None:
        self.chat = type("_Chat", (), {"completions": _FakeCompletions(error)})()


def _use_fake_client(monkeypatch: pytest.MonkeyPatch, error: Exception) -> None:
    monkeypatch.setattr(llm_module, "_get_client", lambda *a, **k: _FakeClient(error))
    monkeypatch.setattr(
        llm_module, "get_settings", lambda: Settings("key", None, "model", 30)
    )


def _request() -> httpx.Request:
    return httpx.Request("POST", "https://api.openai.com/v1/chat/completions")


def _response(
    status_code: int, headers: dict[str, str] | None = None
) -> httpx.Response:
    return httpx.Response(status_code, request=_request(), headers=headers or {})


def test_rate_limit_error_maps_to_llm_rate_limited_with_retry_after(monkeypatch):
    error = RateLimitError(
        "rate limited", response=_response(429, {"retry-after": "7"}), body=None
    )
    _use_fake_client(monkeypatch, error)

    with pytest.raises(LlmRateLimitedError) as exc_info:
        llm_module.generate_text("system", "user")

    assert exc_info.value.retry_after_seconds == 7


def test_timeout_error_maps_to_llm_timeout(monkeypatch):
    _use_fake_client(monkeypatch, APITimeoutError(_request()))

    with pytest.raises(LlmTimeoutError):
        llm_module.generate_text("system", "user")


def test_upstream_5xx_maps_to_llm_upstream_error(monkeypatch):
    error = APIStatusError("bad gateway", response=_response(502), body=None)
    _use_fake_client(monkeypatch, error)

    with pytest.raises(LlmUpstreamError):
        llm_module.generate_text("system", "user")


def test_client_side_status_error_maps_to_llm_unavailable(monkeypatch):
    error = APIStatusError("bad request", response=_response(400), body=None)
    _use_fake_client(monkeypatch, error)

    with pytest.raises(LlmUnavailableError):
        llm_module.generate_text("system", "user")


def test_connection_error_maps_to_llm_unavailable(monkeypatch):
    _use_fake_client(monkeypatch, APIConnectionError(request=_request()))

    with pytest.raises(LlmUnavailableError):
        llm_module.generate_text("system", "user")


def test_usage_sink_receives_token_counts(monkeypatch):
    usage = SimpleNamespace(prompt_tokens=12, completion_tokens=3, total_tokens=15)
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
        usage=usage,
    )
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **_kwargs: response)
        )
    )
    monkeypatch.setattr(llm_module, "_get_client", lambda *a, **k: fake_client)
    monkeypatch.setattr(
        llm_module, "get_settings", lambda: Settings("key", None, "model", 30)
    )

    sink: dict[str, object] = {}
    result = llm_module.generate_text("system", "user", usage_sink=sink)

    assert result == "ok"
    assert sink == {"input_tokens": 12, "output_tokens": 3, "total_tokens": 15}


def test_usage_sink_includes_reasoning_tokens_when_model_reports_them(monkeypatch):
    # reasoning 모델은 effort를 명시하지 않아도 completion_tokens 안에 보이지 않는
    # reasoning 분량을 섞어 과금한다 — 그 분량을 별도 필드로 복원할 수 있어야 한다.
    usage = SimpleNamespace(
        prompt_tokens=12,
        completion_tokens=50,
        total_tokens=62,
        completion_tokens_details=SimpleNamespace(reasoning_tokens=38),
    )
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
        usage=usage,
    )
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **_kwargs: response)
        )
    )
    monkeypatch.setattr(llm_module, "_get_client", lambda *a, **k: fake_client)
    monkeypatch.setattr(
        llm_module, "get_settings", lambda: Settings("key", None, "model", 30)
    )

    sink: dict[str, object] = {}
    llm_module.generate_text("system", "user", usage_sink=sink)

    assert sink == {
        "input_tokens": 12,
        "output_tokens": 50,
        "total_tokens": 62,
        "reasoning_tokens": 38,
    }


def test_usage_sink_left_untouched_without_sink(monkeypatch):
    usage = SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2)
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
        usage=usage,
    )
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **_kwargs: response)
        )
    )
    monkeypatch.setattr(llm_module, "_get_client", lambda *a, **k: fake_client)
    monkeypatch.setattr(
        llm_module, "get_settings", lambda: Settings("key", None, "model", 30)
    )

    # usage_sink를 안 넘기는 기존 호출부는 그대로 동작해야 한다.
    assert llm_module.generate_text("system", "user") == "ok"
