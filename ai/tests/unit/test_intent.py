from types import SimpleNamespace

import pytest

import zipsai.orchestration.intent as intent_module
from zipsai.contracts.converse import (
    ComplaintDraft,
    ComplaintState,
    ConverseRequest,
    HistoryTurn,
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
    return SimpleNamespace(
        planned_tools=[],
        route=route,
        contextualized_message=contextualized_message,
    )


def _plan_decision(
    planned_tools: list[str],
    route: Route,
    contextualized_message: str | None = None,
):
    return SimpleNamespace(
        planned_tools=planned_tools,
        route=route,
        contextualized_message=contextualized_message,
    )


@pytest.mark.parametrize(
    ("planned_tools", "route"),
    [
        (["process_complaint_turn"], Route.COMPLAINT),
        (["search_building_docs"], Route.KNOWLEDGE),
        (["search_public_web", "process_complaint_turn", "process_complaint_turn"], Route.KNOWLEDGE),
        ([], Route.KNOWLEDGE),
    ],
    ids=["complaint", "building-knowledge", "mixed", "no-tool"],
)
def test_plan_returns_distinct_tools_and_response_route_fallback(
    monkeypatch, planned_tools, route
):
    schemas = []
    monkeypatch.setattr(
        intent_module,
        "generate_structured",
        lambda _system, _user, schema, **_kwargs: (
            schemas.append(schema), _plan_decision(planned_tools, route)
        )[1],
    )

    result = classify_intent(_build_state("질문 또는 민원"))

    assert result["planned_tools"] == list(dict.fromkeys(planned_tools))
    assert result["route"] is route
    assert "planned_tools" in schemas[0].model_fields
    assert "route" in schemas[0].model_fields
    assert "arguments" not in schemas[0].model_fields


def test_plan_uses_active_complaint_as_soft_context_for_follow_up(monkeypatch):
    state = _build_state("욕실이요", current_route=Route.COMPLAINT)
    state["request"] = state["request"].model_copy(
        update={
            "current_complaint_state": ComplaintState.COLLECTING,
            "complaint_draft": ComplaintDraft(issue_type="plumbing", symptom="물이 새요"),
            "conversation_history": [
                HistoryTurn(message_id="a1", role="assistant", text="어디에서 생겼나요?")
            ],
        }
    )
    prompt_inputs = []
    monkeypatch.setattr(
        intent_module,
        "generate_structured",
        lambda system, user, *_args, **_kwargs: (
            prompt_inputs.append((system, user)),
            _plan_decision(["process_complaint_turn"], Route.COMPLAINT),
        )[1],
    )

    result = classify_intent(state)

    assert result["planned_tools"] == ["process_complaint_turn"]
    system, user = prompt_inputs[0]
    assert "only show whether this turn continues a complaint" in system
    assert "An active complaint alone does not require process_complaint_turn" in system
    assert "현재 민원 상태 힌트: collecting" in user
    assert '"symptom":"물이 새요"' in user
    assert "미수집 민원 항목 힌트: location" in user
    assert "assistant: 어디에서 생겼나요?" in user


def test_clear_new_question_does_not_inherit_complaint_tool_plan(monkeypatch):
    state = _build_state("주차 등록은 어떻게 하나요?", current_route=Route.COMPLAINT)
    state["request"] = state["request"].model_copy(
        update={
            "current_complaint_state": ComplaintState.COLLECTING,
            "complaint_draft": ComplaintDraft(issue_type="plumbing", symptom="물이 새요"),
        }
    )
    monkeypatch.setattr(
        intent_module,
        "generate_structured",
        lambda *_args, **_kwargs: _plan_decision(
            ["search_building_docs"], Route.KNOWLEDGE
        ),
    )

    result = classify_intent(state)

    assert result["planned_tools"] == ["search_building_docs"]
    assert result["route"] is Route.KNOWLEDGE


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


def test_intent_prompt_plans_tools_and_does_not_assign_order_or_arguments(monkeypatch):
    prompts = []

    def classify(system, user, *_args, **_kwargs):
        prompts.append((system, user))
        return _decision(Route.COMPLAINT)

    monkeypatch.setattr(intent_module, "generate_structured", classify)

    classify_intent(_build_state("누수 처리 기한을 알려주고 욕실 누수 민원도 작성해줘"))

    system, _ = prompts[0]
    assert "planned_tools = the distinct union" in system
    assert "Do not answer the resident or choose tool order or arguments" in system
    assert "route (used only when no tool runs)" in system
    assert "required_capabilities" not in system


def test_intent_prompt_uses_tool_contract_descriptions_and_prefers_building_docs(
    monkeypatch,
):
    from zipsai.orchestration.tool_contracts import TOOL_CONTRACTS

    prompts = []

    def classify(system, user, *_args, **_kwargs):
        prompts.append(system)
        return _decision(Route.KNOWLEDGE)

    monkeypatch.setattr(intent_module, "generate_structured", classify)

    classify_intent(_build_state("관리비 납부일 알려줘"))

    system = prompts[0]
    for contract in TOOL_CONTRACTS:
        assert f"- {contract.name}: {contract.description}" in system
    assert "If unsure between building and public information, choose search_building_docs" in system
    assert "a new building question during it still needs search_building_docs" in system


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
        "planned_tools": [],
        "image_analysis": None,
        "contextualized_message": "쓰레기 배출일이 언제예요?",
    }


def test_classify_intent_returns_empty_plan_when_model_selects_no_tools(monkeypatch):
    monkeypatch.setattr(
        intent_module,
        "generate_structured",
        lambda *_args, **_kwargs: _decision(
            Route.COMPLAINT,
            "누수 처리 기한을 알려주고 욕실 누수 민원도 작성해줘",
        ),
    )

    result = classify_intent(
        _build_state("누수 처리 기한을 알려주고 욕실 누수 민원도 작성해줘")
    )

    assert result["planned_tools"] == []


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


def test_intent_prompt_searches_again_for_follow_up_questions(monkeypatch):
    prompts = []

    def classify(system, user, *_args, **_kwargs):
        prompts.append(system)
        return _decision(Route.KNOWLEDGE)

    monkeypatch.setattr(intent_module, "generate_structured", classify)

    classify_intent(_build_state("음식물 쓰레기는?"))

    system = prompts[0]
    assert "A follow-up question still needs its search tool" in system
    assert "never answer from conversation history" in system
