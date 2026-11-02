from types import SimpleNamespace

import pytest

import zipsai.orchestration.intent as intent_module
from zipsai.contracts.converse import (
    ConverseRequest,
    ImageAnalysis,
    ImageObservation,
    Route,
)
from zipsai.errors import (
    ImageAnalysisError,
    IntentClassificationError,
    LlmRateLimitedError,
    LlmTimeoutError,
    LlmUnavailableError,
    LlmUpstreamError,
)
from zipsai.orchestration.intent import classify_intent
from zipsai.orchestration.state import AgentState


def _build_state(
    text: str | None,
    *,
    images: list[dict[str, object]] | None = None,
    history: list[dict[str, object]] | None = None,
    current_route: Route | None = None,
) -> AgentState:
    return {
        "request": ConverseRequest.model_validate(
            {
                "building_id": 1,
                "room_no": "301",
                "resident_id": "linda",
                "conversation_id": "conv-001",
                "turn_id": "turn-001",
                "trace_id": "trace-001",
                "current_route": current_route,
                "current_complaint_state": None,
                "message": {
                    "message_id": "msg-001",
                    "text": text,
                    "images": images or [],
                },
                "conversation_history": history or [],
                "complaint_draft": None,
            }
        ),
        "route": None,
        "complaint_state": None,
        "reply": None,
        "result": None,
        "image_analysis": None,
    }


def _decision(route: Route, contextualized_message: str | None = None):
    return SimpleNamespace(route=route, contextualized_message=contextualized_message)


def test_classify_intent_passes_image_observations_to_route_model(monkeypatch):
    analysis = ImageAnalysis(
        images=[ImageObservation(attachmentId=123, summary="세탁기 아래 물이 고임")]
    )
    monkeypatch.setattr(
        intent_module, "analyze_images", lambda *_args, **_kwargs: analysis
    )
    prompts = []

    def classify(system, user, schema, **_kwargs):
        prompts.append((system, user, schema))
        return _decision(Route.COMPLAINT, "세탁기 밑으로 물이 새요")

    monkeypatch.setattr(intent_module, "generate_structured", classify)
    result = classify_intent(
        _build_state(
            "물이 새요",
            images=[{"attachmentId": 123, "url": "https://example.com/leak.jpg"}],
        )
    )

    assert result["route"] is Route.COMPLAINT
    assert result["image_analysis"] is analysis
    assert result["contextualized_message"] == "세탁기 밑으로 물이 새요"
    assert "세탁기 아래 물이 고임" in prompts[0][1]
    assert "route" in prompts[0][2].model_fields
    assert "images" not in prompts[0][2].model_fields


def test_intent_prompt_keeps_observation_and_context_rules(monkeypatch):
    prompts = []
    monkeypatch.setattr(
        intent_module,
        "analyze_images",
        lambda *_args, **_kwargs: ImageAnalysis(images=[]),
    )

    def classify(system, user, *_args, **_kwargs):
        prompts.append((system, user))
        return _decision(Route.CLARIFY)

    monkeypatch.setattr(intent_module, "generate_structured", classify)
    classify_intent(_build_state("사진을 봐주세요"))
    system, user = prompts[0]
    assert "Do not answer, diagnose, or add facts" in system
    assert "The image alone cannot establish intent or location" in system
    assert "현재 이미지 관찰" in user


def test_text_only_classifies_without_vlm(monkeypatch):
    monkeypatch.setattr(
        intent_module,
        "analyze_images",
        lambda *_args, **_kwargs: pytest.fail("no image call"),
    )
    monkeypatch.setattr(
        intent_module,
        "generate_structured",
        lambda *_args, **_kwargs: _decision(
            Route.KNOWLEDGE, "쓰레기 배출일이 언제예요?"
        ),
    )

    result = classify_intent(_build_state("쓰레기 배출일이 언제예요?"))

    assert result == {
        "route": Route.KNOWLEDGE,
        "image_analysis": None,
        "contextualized_message": "쓰레기 배출일이 언제예요?",
    }


def test_image_only_success_passes_observation_to_route_model(monkeypatch):
    analysis = ImageAnalysis(
        images=[ImageObservation(attachmentId=123, summary="세탁기 아래에 물이 고임")]
    )
    monkeypatch.setattr(
        intent_module, "analyze_images", lambda *_args, **_kwargs: analysis
    )
    prompts = []

    def classify(_system, user, *_args, **_kwargs):
        prompts.append(user)
        return _decision(Route.CLARIFY)

    monkeypatch.setattr(intent_module, "generate_structured", classify)
    result = classify_intent(
        _build_state(
            None, images=[{"attachmentId": 123, "url": "https://example.com/leak.jpg"}]
        )
    )

    assert result["route"] is Route.CLARIFY
    assert result["image_analysis"] is analysis
    assert "세탁기 아래에 물이 고임" in prompts[0]


def test_classify_intent_resolves_history_reference(monkeypatch):
    def classify(_system, user, *_args, **_kwargs):
        assert "분리수거 언제해" in user
        return _decision(Route.KNOWLEDGE, "분리수거는 어디에 배출하나요?")

    monkeypatch.setattr(intent_module, "generate_structured", classify)
    history = [
        {"message_id": "m1", "role": "user", "text": "분리수거 언제해"},
        {"message_id": "m2", "role": "assistant", "text": "화요일 저녁에 배출합니다."},
    ]

    result = classify_intent(_build_state("그거 어디다해", history=history))

    assert result["contextualized_message"] == "분리수거는 어디에 배출하나요?"


def test_classify_intent_rejects_invalid_structured_response(monkeypatch):
    monkeypatch.setattr(
        intent_module, "generate_structured", lambda *_args, **_kwargs: None
    )
    with pytest.raises(IntentClassificationError):
        classify_intent(_build_state("화장실에서 물이 새요"))


def test_classify_intent_propagates_provider_failure(monkeypatch):
    monkeypatch.setattr(
        intent_module,
        "generate_structured",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            LlmUnavailableError("provider unavailable")
        ),
    )
    with pytest.raises(LlmUnavailableError):
        classify_intent(_build_state("화장실에서 물이 새요"))


@pytest.mark.parametrize(
    "error",
    [
        ImageAnalysisError("invalid response"),
        LlmRateLimitedError("rate limited"),
        LlmTimeoutError("timed out"),
        LlmUnavailableError("unavailable"),
        LlmUpstreamError("upstream error"),
    ],
)
def test_image_failure_with_text_falls_back_to_text_intent(monkeypatch, error):
    monkeypatch.setattr(
        intent_module,
        "analyze_images",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(error),
    )
    calls = []

    def classify(system, user, *_args, **_kwargs):
        calls.append((system, user))
        return _decision(Route.KNOWLEDGE)

    monkeypatch.setattr(intent_module, "generate_structured", classify)
    result = classify_intent(
        _build_state(
            "세탁기 어떻게 돌려?",
            images=[{"attachmentId": 123, "url": "https://example.com/washer.jpg"}],
        )
    )

    assert result["route"] is Route.KNOWLEDGE
    assert result["image_analysis"] is None
    assert result["image_analysis_failed"] is True
    assert "세탁기 어떻게 돌려?" in calls[0][1]
    assert "사진 내용을 추측하지 말 것" in calls[0][1]


@pytest.mark.parametrize(
    "error",
    [
        ImageAnalysisError("invalid response"),
        LlmRateLimitedError("rate limited"),
        LlmTimeoutError("timed out"),
        LlmUnavailableError("unavailable"),
        LlmUpstreamError("upstream error"),
    ],
)
def test_image_only_failure_routes_to_clarify_without_text_fallback(monkeypatch, error):
    monkeypatch.setattr(
        intent_module,
        "analyze_images",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(error),
    )
    monkeypatch.setattr(
        intent_module,
        "generate_structured",
        lambda *_args, **_kwargs: pytest.fail(
            "must not infer intent from failed image"
        ),
    )
    result = classify_intent(
        _build_state(
            None,
            images=[{"attachmentId": 123, "url": "https://example.com/washer.jpg"}],
        )
    )

    assert result["route"] is Route.CLARIFY
    assert result["image_analysis"] is None
    assert result["image_analysis_failed"] is True
