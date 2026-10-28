import logging
import traceback
from types import SimpleNamespace

import httpx
import pytest
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    ContentFilterFinishReasonError,
    LengthFinishReasonError,
    RateLimitError,
)

from zipsai.contracts.converse import ImageAttachment, Route
from zipsai.errors import (
    ImageAnalysisError,
    LlmRateLimitedError,
    LlmTimeoutError,
    LlmUnavailableError,
    LlmUpstreamError,
)
from zipsai.integrations import vlm as vlm_module
from zipsai.integrations.vlm import _ModelIntentAndImages, _ModelObservation
from zipsai.settings import Settings

_SYSTEM_PROMPT = "classify and analyze"
_USER_PROMPT = "세탁기 밑으로 물이 새요"
_IMAGE = ImageAttachment(
    attachmentId=123,
    url="https://zipsai-dev-uploads.s3.ap-northeast-2.amazonaws.com/leak.jpg?signature=abc",
)


def _request() -> httpx.Request:
    return httpx.Request("GET", "https://api.openai.com/v1/chat/completions")


def _response(
    status_code: int, headers: dict[str, str] | None = None
) -> httpx.Response:
    return httpx.Response(status_code, request=_request(), headers=headers or {})


def _fake_parse_client(
    parsed: _ModelIntentAndImages | None,
    calls: list[dict[str, object]] | None = None,
    refusal: str | None = None,
) -> SimpleNamespace:
    def parse(**kwargs: object) -> SimpleNamespace:
        if calls is not None:
            calls.append(kwargs)
        message = SimpleNamespace(parsed=parsed, refusal=refusal)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    return SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(parse=parse))
    )


def _fake_empty_client() -> SimpleNamespace:
    return SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(parse=lambda **_: SimpleNamespace(choices=[]))
        )
    )


def _use_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        vlm_module, "get_settings", lambda: Settings("key", None, "text-model", 30)
    )


def _classify(images: list[ImageAttachment] | None = None):
    return vlm_module.classify_and_analyze(
        [_IMAGE] if images is None else images,
        system_prompt=_SYSTEM_PROMPT,
        user_prompt=_USER_PROMPT,
    )


def test_classify_and_analyze_returns_route_and_attachment_observations(monkeypatch):
    calls: list[dict[str, object]] = []
    parsed = _ModelIntentAndImages(
        route=Route.COMPLAINT,
        images=[_ModelObservation(summary="세탁기 아래 물이 고여 있음", ocr_text="E1")],
        contextualized_message="세탁기 아래 물이 새는데 어떻게 하나요?",
    )
    monkeypatch.setattr(
        vlm_module, "_get_client", lambda *_: _fake_parse_client(parsed, calls)
    )
    _use_settings(monkeypatch)

    route, analysis, contextualized_message = _classify()

    assert route is Route.COMPLAINT
    assert analysis.model_dump() == {
        "images": [
            {
                "attachment_id": 123,
                "summary": "세탁기 아래 물이 고여 있음",
                "ocr_text": "E1",
            }
        ]
    }
    assert contextualized_message == "세탁기 아래 물이 새는데 어떻게 하나요?"
    assert calls[0]["messages"][0] == {"role": "system", "content": _SYSTEM_PROMPT}
    assert calls[0]["messages"][1]["content"] == [
        {"type": "text", "text": _USER_PROMPT},
        {"type": "image_url", "image_url": {"url": _IMAGE.url}},
    ]
    assert calls[0]["response_format"] is _ModelIntentAndImages


def test_classify_and_analyze_supports_text_only_intent(monkeypatch):
    calls: list[dict[str, object]] = []
    parsed = _ModelIntentAndImages(
        route=Route.KNOWLEDGE, images=[], contextualized_message=None
    )
    monkeypatch.setattr(
        vlm_module, "_get_client", lambda *_: _fake_parse_client(parsed, calls)
    )
    _use_settings(monkeypatch)

    route, analysis, contextualized_message = _classify([])

    assert route is Route.KNOWLEDGE
    assert analysis.images == []
    assert contextualized_message is None
    assert calls[0]["messages"][1]["content"] == [
        {"type": "text", "text": _USER_PROMPT}
    ]


def test_pod_503_retries_image_analysis_on_openrouter(monkeypatch):
    pod = Settings("pod-key", "https://pod.example/v1", "pod-model", 30,
                   vlm_model="pod-model", provider="pod")
    backup = Settings("router-key", "https://router.example/v1", "router-text", 30,
                      vlm_model="router-vision", provider="openrouter")
    parsed = _ModelIntentAndImages(
        route=Route.COMPLAINT,
        images=[_ModelObservation(summary="누수", ocr_text=None)],
        contextualized_message=None,
    )
    calls = []

    def client(key, *_args):
        def parse(**kwargs):
            calls.append((key, kwargs["model"]))
            if key == "pod-key":
                raise APIStatusError("unavailable", response=_response(503), body=None)
            return _fake_parse_client(parsed).chat.completions.parse(**kwargs)

        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=parse)))

    monkeypatch.setattr(vlm_module, "get_settings", lambda: pod)
    monkeypatch.setattr(vlm_module, "_get_client", client)
    monkeypatch.setattr(
        "zipsai.integrations.llm.get_settings", lambda provider=None: backup
    )
    monkeypatch.setattr("zipsai.integrations.llm.mark_pod_unavailable", lambda *_: None)

    assert _classify()[0] is Route.COMPLAINT
    assert calls == [("pod-key", "pod-model"), ("router-key", "router-vision")]


def test_classify_and_analyze_ignores_hallucinated_images_when_none_attached(
    monkeypatch: pytest.MonkeyPatch,
):
    """VLM이 이미지 0장인데도 images 필드를 채워 돌려줄 때가 있다(할루시네이션).

    프로덕션에서 텍스트만 있는 평범한 턴이 이 때문에 500으로 떨어졌다
    (zipsai-ai-stage-failed 알람). 요청한 이미지가 없으면 모델이 뭘 돌려주든
    버리고 통과시킨다 — 실제 이미지 개수가 어긋난 경우는 별개로 계속 막는다.
    """
    parsed = _ModelIntentAndImages(
        route=Route.KNOWLEDGE,
        images=[_ModelObservation(summary="할루시네이션된 관측", ocr_text=None)],
        contextualized_message="쓰레기 배출일이 언제예요?",
    )
    monkeypatch.setattr(
        vlm_module, "_get_client", lambda *_: _fake_parse_client(parsed)
    )
    _use_settings(monkeypatch)

    route, analysis, contextualized_message = _classify([])

    assert route is Route.KNOWLEDGE
    assert analysis.images == []
    assert contextualized_message == "쓰레기 배출일이 언제예요?"


@pytest.mark.parametrize(
    "client",
    [
        lambda: _fake_parse_client(None, refusal="blocked"),
        lambda: _fake_parse_client(None),
        _fake_empty_client,
        lambda: _fake_parse_client(
            _ModelIntentAndImages(
                route=Route.CLARIFY, images=[], contextualized_message=None
            )
        ),
    ],
)
def test_classify_and_analyze_rejects_refusal_invalid_or_mismatched_response(
    monkeypatch: pytest.MonkeyPatch, client
):
    monkeypatch.setattr(vlm_module, "_get_client", lambda *_: client())
    _use_settings(monkeypatch)

    with pytest.raises(ImageAnalysisError):
        _classify()


class _FakeCompletions:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def parse(self, **_kwargs: object) -> None:
        raise self.error


class _FakeClient:
    def __init__(self, error: Exception) -> None:
        self.chat = type("_Chat", (), {"completions": _FakeCompletions(error)})()


def _use_fake_completions_client(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    monkeypatch.setattr(vlm_module, "_get_client", lambda *a, **k: _FakeClient(error))
    _use_settings(monkeypatch)


@pytest.mark.parametrize(
    ("error", "error_type"),
    [
        (
            RateLimitError("rate limited", response=_response(429), body=None),
            LlmRateLimitedError,
        ),
        (APITimeoutError(_request()), LlmTimeoutError),
        (
            APIStatusError("bad gateway", response=_response(502), body=None),
            LlmUpstreamError,
        ),
        (
            APIStatusError("bad request", response=_response(400), body=None),
            LlmUnavailableError,
        ),
        (APIConnectionError(request=_request()), LlmUnavailableError),
        (
            LengthFinishReasonError(
                completion=SimpleNamespace(
                    usage=SimpleNamespace(completion_tokens_details=None)
                )
            ),
            ImageAnalysisError,
        ),
        (ContentFilterFinishReasonError(), ImageAnalysisError),
    ],
)
def test_provider_errors_keep_their_existing_error_mapping(
    monkeypatch: pytest.MonkeyPatch, error: Exception, error_type: type[Exception]
):
    _use_fake_completions_client(monkeypatch, error)

    with pytest.raises(error_type):
        _classify()


def test_rate_limit_error_preserves_retry_after(monkeypatch):
    _use_fake_completions_client(
        monkeypatch,
        RateLimitError(
            "rate limited", response=_response(429, {"retry-after": "7"}), body=None
        ),
    )

    with pytest.raises(LlmRateLimitedError) as caught:
        _classify()

    assert caught.value.retry_after_seconds == 7


def test_provider_error_does_not_log_signed_image_url(monkeypatch):
    _use_fake_completions_client(
        monkeypatch,
        APIStatusError(
            "Failed to fetch https://example.com/image.jpg?X-Amz-Signature=secret",
            response=_response(400),
            body={
                "error": {
                    "message": "Failed to fetch https://example.com/image.jpg?X-Amz-Signature=secret"
                }
            },
        ),
    )

    with pytest.raises(LlmUnavailableError) as caught:
        _classify()

    rendered_error = "".join(traceback.format_exception(caught.value))
    assert caught.value.__suppress_context__
    assert "X-Amz-Signature" not in rendered_error
    assert "secret" not in rendered_error


def test_unknown_vlm_exception_is_logged_as_failure(monkeypatch, caplog):
    class BoomError(RuntimeError):
        pass

    def parse(**_kwargs: object) -> None:
        raise BoomError("주민 원문")

    monkeypatch.setattr(
        vlm_module,
        "_get_client",
        lambda *_: SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(parse=parse))
        ),
    )
    _use_settings(monkeypatch)

    with (
        caplog.at_level(logging.INFO, logger="zipsai.integrations.llm"),
        pytest.raises(BoomError),
    ):
        _classify()

    recorded = [record for record in caplog.records if record.message == "llm_call"]
    assert recorded[-1].outcome == "fail"
    assert recorded[-1].status_code is None
    assert recorded[-1].error_code == "BoomError"
    assert "주민 원문" not in recorded[-1].getMessage()
