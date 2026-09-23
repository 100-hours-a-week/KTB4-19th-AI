import json
from types import SimpleNamespace

import httpx
import pytest
from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

from zipsai.errors import (
    ImageAnalysisError,
    LlmRateLimitedError,
    LlmTimeoutError,
    LlmUnavailableError,
    LlmUpstreamError,
)
from zipsai.integrations import vlm as vlm_module
from zipsai.settings import Settings

_PROMPT = "analyze"
_IMAGE_URL = "https://zipsai-dev-uploads.s3.ap-northeast-2.amazonaws.com/leak.jpg?X-Amz-Signature=abc"


def _request() -> httpx.Request:
    return httpx.Request("GET", "https://api.openai.com/v1/chat/completions")


def _response(
    status_code: int, headers: dict[str, str] | None = None
) -> httpx.Response:
    return httpx.Response(status_code, request=_request(), headers=headers or {})


def _fake_completions_client(
    content: str, calls: list[dict[str, object]] | None = None
) -> SimpleNamespace:
    def create(**kwargs: object) -> SimpleNamespace:
        if calls is not None:
            calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )

    return SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )


def _fake_empty_client() -> SimpleNamespace:
    return SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **_: SimpleNamespace(choices=[]))
        )
    )


def _use_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        vlm_module, "get_settings", lambda: Settings("key", None, "text-model", 30)
    )


# --- analyze_images: response parsing -----------------------------------


def test_analyze_images_returns_typed_observations(monkeypatch):
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        vlm_module,
        "_get_client",
        lambda *_: _fake_completions_client(
            json.dumps(
                {"images": [{"summary": "세탁기 표시창이 켜져 있음", "ocr_text": "E1"}]}
            ),
            calls,
        ),
    )
    _use_settings(monkeypatch)

    result = vlm_module.analyze_images([_IMAGE_URL], _PROMPT)

    assert result.model_dump() == {
        "images": [
            {
                "url": _IMAGE_URL,
                "summary": "세탁기 표시창이 켜져 있음",
                "ocr_text": "E1",
            }
        ]
    }
    assert calls[0]["messages"][0] == {"role": "system", "content": _PROMPT}
    assert calls[0]["messages"][1]["content"][0] == {
        "type": "image_url",
        "image_url": {"url": _IMAGE_URL},
    }


def test_analyze_images_accepts_json_code_fence(monkeypatch):
    monkeypatch.setattr(
        vlm_module,
        "_get_client",
        lambda *_: _fake_completions_client(
            '```json\n{"images":[{"summary":"오류 코드가 보임","ocr_text":"E1"}]}\n```'
        ),
    )
    _use_settings(monkeypatch)

    result = vlm_module.analyze_images([_IMAGE_URL], _PROMPT)

    assert result.images[0].ocr_text == "E1"


def test_analyze_images_rejects_invalid_json(monkeypatch):
    monkeypatch.setattr(
        vlm_module, "_get_client", lambda *_: _fake_completions_client("not json")
    )
    _use_settings(monkeypatch)

    with pytest.raises(ImageAnalysisError):
        vlm_module.analyze_images([_IMAGE_URL], _PROMPT)


def test_analyze_images_rejects_empty_model_response(monkeypatch):
    monkeypatch.setattr(vlm_module, "_get_client", lambda *_: _fake_empty_client())
    _use_settings(monkeypatch)

    with pytest.raises(ImageAnalysisError):
        vlm_module.analyze_images([_IMAGE_URL], _PROMPT)


def test_analyze_images_rejects_observation_count_mismatch(monkeypatch):
    monkeypatch.setattr(
        vlm_module,
        "_get_client",
        lambda *_: _fake_completions_client(json.dumps({"images": []})),
    )
    _use_settings(monkeypatch)

    with pytest.raises(ImageAnalysisError):
        vlm_module.analyze_images([_IMAGE_URL], _PROMPT)


# --- analyze_images: OpenAI error mapping -------------------------------


class _FakeCompletions:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def create(self, **_kwargs: object) -> None:
        raise self.error


class _FakeClient:
    def __init__(self, error: Exception) -> None:
        self.chat = type("_Chat", (), {"completions": _FakeCompletions(error)})()


def _use_fake_completions_client(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    monkeypatch.setattr(vlm_module, "_get_client", lambda *a, **k: _FakeClient(error))
    _use_settings(monkeypatch)


def test_rate_limit_error_maps_to_llm_rate_limited(monkeypatch):
    error = RateLimitError("rate limited", response=_response(429), body=None)
    _use_fake_completions_client(monkeypatch, error)

    with pytest.raises(LlmRateLimitedError):
        vlm_module.analyze_images([_IMAGE_URL], _PROMPT)


def test_timeout_error_maps_to_llm_timeout(monkeypatch):
    _use_fake_completions_client(monkeypatch, APITimeoutError(_request()))

    with pytest.raises(LlmTimeoutError):
        vlm_module.analyze_images([_IMAGE_URL], _PROMPT)


def test_upstream_5xx_maps_to_llm_upstream_error(monkeypatch):
    error = APIStatusError("bad gateway", response=_response(502), body=None)
    _use_fake_completions_client(monkeypatch, error)

    with pytest.raises(LlmUpstreamError):
        vlm_module.analyze_images([_IMAGE_URL], _PROMPT)


def test_client_side_status_error_maps_to_llm_unavailable(monkeypatch):
    error = APIStatusError("bad request", response=_response(400), body=None)
    _use_fake_completions_client(monkeypatch, error)

    with pytest.raises(LlmUnavailableError):
        vlm_module.analyze_images([_IMAGE_URL], _PROMPT)


def test_connection_error_maps_to_llm_unavailable(monkeypatch):
    _use_fake_completions_client(monkeypatch, APIConnectionError(request=_request()))

    with pytest.raises(LlmUnavailableError):
        vlm_module.analyze_images([_IMAGE_URL], _PROMPT)
