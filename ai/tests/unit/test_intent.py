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


def _plan(*kinds: str, contextualized_message: str | None = None):
    return SimpleNamespace(
        requests=[SimpleNamespace(text="요청", kind=kind) for kind in kinds],
        contextualized_message=contextualized_message,
    )


@pytest.mark.parametrize(
    ("kinds", "planned_tools", "route"),
    [
        (["complaint"], ["process_complaint_turn"], Route.COMPLAINT),
        (["building_question"], ["search_building_docs"], Route.KNOWLEDGE),
        (
            ["public_question", "complaint", "complaint"],
            ["search_public_web", "process_complaint_turn"],
            Route.COMPLAINT,
        ),
        (["other"], [], Route.CLARIFY),
        ([], [], Route.CLARIFY),
    ],
    ids=["complaint", "building-question", "mixed", "other", "empty"],
)
def test_plan_derives_tools_and_route_from_request_kinds(monkeypatch, kinds, planned_tools, route):
    schemas = []
    monkeypatch.setattr(
        intent_module,
        "generate_structured",
        lambda _system, _user, schema, **_kwargs: (schemas.append(schema), _plan(*kinds))[1],
    )

    result = classify_intent(_build_state("질문 또는 민원"))

    assert result["planned_tools"] == planned_tools
    assert result["route"] is route
    # 모델은 이번 턴의 요청을 먼저 적고, 도구와 route는 코드가 정한다.
    assert list(schemas[0].model_fields) == ["requests", "contextualized_message"]


def test_question_during_complaint_searches_and_keeps_complaint_route(monkeypatch):
    state = _build_state("주차 등록은 어떻게 하나요?", current_route=Route.COMPLAINT)
    state["request"] = state["request"].model_copy(
        update={
            "current_complaint_state": ComplaintState.COLLECTING,
            "complaint_draft": ComplaintDraft(issue_type="plumbing", symptom="물이 새요"),
        }
    )
    monkeypatch.setattr(
        intent_module, "generate_structured", lambda *_a, **_kw: _plan("building_question")
    )

    result = classify_intent(state)

    assert result["planned_tools"] == ["search_building_docs"]
    assert result["route"] is Route.COMPLAINT


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
            _plan("complaint"),
        )[1],
    )

    result = classify_intent(state)

    assert result["planned_tools"] == ["process_complaint_turn"]
    system, user = prompt_inputs[0]
    assert "they are never requests themselves" in system
    assert "현재 민원 상태 힌트: collecting" in user
    # Plan은 route를 정하지 않으므로 이전 route로 쏠리지 않게 넘기지 않는다.
    assert "route" not in user
    assert '"symptom":"물이 새요"' in user
    assert "미수집 민원 항목 힌트: location" in user
    assert "assistant: 어디에서 생겼나요?" in user


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
        return _plan("complaint", contextualized_message="세탁기 밑으로 물이 새요")

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
    assert "requests" in prompts[0][2].model_fields
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
        return _plan()

    monkeypatch.setattr(intent_module, "generate_structured", classify)
    classify_intent(_build_state("사진을 봐주세요"))
    system, user = prompts[0]
    assert "Do not answer, diagnose, or add facts" in system
    assert "The image alone cannot establish intent or location" in system
    assert "현재 이미지 관찰" in user


def _system_prompt(monkeypatch) -> str:
    prompts = []
    monkeypatch.setattr(
        intent_module,
        "generate_structured",
        lambda system, *_a, **_kw: (prompts.append(system), _plan())[1],
    )
    classify_intent(_build_state("안녕하세요"))
    return prompts[0]


def test_intent_prompt_lists_requests_of_current_turn_only(monkeypatch):
    system = _system_prompt(monkeypatch)

    # 핵심 규칙: 이번 턴만 요청으로 나누고, 대화 상태는 요청을 해석하는 데만 쓴다.
    assert "split only the current turn into the requests it makes" in system
    assert "A reported problem and each question are separate requests, even within one sentence" in system
    assert "they are never requests themselves" in system
    assert "A question stays a question" in system
    # 도구·route는 코드가 정하므로 모델에게 고르게 하지 않는다.
    assert "planned_tools" not in system
    assert "route" not in system


@pytest.mark.parametrize(
    "definition",
    [
        "- complaint: reports a broken, leaking, noisy, dirty, or unsafe facility",
        "A broken facility is a complaint even when phrased as a question",
        "- building_question: asks about this building or complex",
        "use building_question if the answer could differ by building",
        "- public_question: asks living-related public information",
        "분리배출 방법",
        "- other: greetings, thanks, small talk, how to use this chat assistant itself",
        "A follow-up question is a new request",
        # 이 서비스에서 민원은 이 채팅으로 접수하는 시설 민원이다. 공공 민원(국민신문고)이 아니다.
        "In this service, 민원 means a facility complaint filed in this chat",
        # 질문 종류는 답이 어디서 오느냐로 나눈다: 건물 문서 / 공개 정보 / 이 assistant 자신.
        "answered from its documents",
    ],
)
def test_intent_prompt_defines_each_request_kind(monkeypatch, definition):
    assert definition in _system_prompt(monkeypatch)


@pytest.mark.parametrize(
    "patch_rule",
    [
        "An active complaint alone does not require process_complaint_turn",
        "a new building question during it still needs search_building_docs",
        "even if other questions came in between",
        "A new question during a complaint is not such a reply",
        "Missing complaint details do not make a facility report ambiguous",
        # 기존 민원과 관련 있는지는 민원 노드만 판단한다. Plan은 시설 문제인지만 본다.
        "ongoing complaint",
        # 효과 없던 예시 문장.
        "is a complaint and a building_question",
    ],
)
def test_intent_prompt_drops_exception_rules_covered_by_kinds(monkeypatch, patch_rule):
    # 진행 중 민원과 새 질문을 구분하던 예외 문장들은 kind 정의 하나로 대체됐다.
    assert patch_rule not in _system_prompt(monkeypatch)


def test_text_only_classifies_without_vlm(monkeypatch):
    monkeypatch.setattr(
        intent_module,
        "analyze_images",
        lambda *_args, **_kwargs: pytest.fail("no image call"),
    )
    monkeypatch.setattr(
        intent_module,
        "generate_structured",
        lambda *_args, **_kwargs: _plan("building_question", contextualized_message="쓰레기 배출일이 언제예요?"),
    )

    result = classify_intent(_build_state("쓰레기 배출일이 언제예요?"))

    assert result == {
        "route": Route.KNOWLEDGE,
        "planned_tools": ["search_building_docs"],
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
        return _plan()

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
        return _plan("building_question", contextualized_message="분리수거는 어디에 배출하나요?")

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
        return _plan("building_question")

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
