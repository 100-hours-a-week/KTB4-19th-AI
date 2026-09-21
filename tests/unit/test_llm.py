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
