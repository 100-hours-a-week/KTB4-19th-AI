import pytest

import zipsai.knowledge.node as knowledge_node
import zipsai.orchestration.graph as graph_module
from zipsai.contracts.converse import (
    ComplaintState,
    ConverseRequest,
    ImageAnalysis,
    ImageObservation,
    Route,
    RouteResult,
)
from zipsai.errors import LlmUnavailableError
from zipsai.orchestration.graph import build_graph, select_next_node
from zipsai.orchestration.state import AgentState
from zipsai.settings import EMBEDDING_DIM


def _make_request(
    current_route: Route | None = None,
    current_complaint_state: ComplaintState | None = None,
) -> ConverseRequest:
    return ConverseRequest.model_validate(
        {
            "building_id": 1,
            "room_no": "301",
            "resident_id": "linda",
            "conversation_id": "conv-001",
            "turn_id": "turn-001",
            "trace_id": "trace-001",
            "current_route": current_route,
            "current_complaint_state": current_complaint_state,
            "message": {
                "message_id": "msg-001",
                "text": "도와주세요",
                "images": [],
            },
            "conversation_history": [],
            "complaint_draft": None,
        }
    )


def _make_state(route: Route | None) -> AgentState:
    return {
        "request": _make_request(),
        "route": route,
        "complaint_state": ComplaintState.COLLECTING,
        "reply": None,
    }


def _stub_classify_intent(state: AgentState) -> dict[str, Route]:
    return {"route": state["route"] or Route.CLARIFY}


@pytest.mark.parametrize(
    ("route", "expected_node"),
    [
        (Route.COMPLAINT, "complaint"),
        (Route.KNOWLEDGE, "knowledge"),
        (Route.CLARIFY, "clarify"),
        (None, "clarify"),
    ],
)
def test_select_next_node_returns_route_node(route: Route | None, expected_node: str):
    assert select_next_node(_make_state(route)) == expected_node


@pytest.mark.parametrize(
    ("route", "handler_name"),
    [
        (Route.COMPLAINT, "handle_complaint"),
        (Route.KNOWLEDGE, "handle_knowledge"),
    ],
)
def test_graph_invokes_feature_handler_for_selected_route(
    monkeypatch: pytest.MonkeyPatch,
    route: Route,
    handler_name: str,
):
    state = _make_state(route)
    handled_requests: list[ConverseRequest] = []

    def handler(request: ConverseRequest, **_kwargs: object) -> dict[str, object]:
        handled_requests.append(request)
        return {"reply": "ok", "result": RouteResult()}

    monkeypatch.setattr(graph_module, handler_name, handler)
    monkeypatch.setattr(graph_module, "classify_intent", _stub_classify_intent)

    build_graph().invoke(state)

    assert handled_requests == [state["request"]]


def _collecting_state() -> AgentState:
    request = _make_request(Route.COMPLAINT, ComplaintState.COLLECTING)
    return {
        "request": request,
        "route": None,
        "complaint_state": request.current_complaint_state,
        "reply": None,
        "result": RouteResult(),
    }


def test_graph_classifies_intent_even_when_complaint_in_progress(
    monkeypatch: pytest.MonkeyPatch,
):
    state = _collecting_state()
    classified: list[ConverseRequest] = []
    handled_requests: list[ConverseRequest] = []

    def handler(req: ConverseRequest, **_kwargs: object) -> dict[str, object]:
        handled_requests.append(req)
        return {
            "complaint_state": ComplaintState.COLLECTING,
            "reply": "ok",
            "result": RouteResult(),
        }

    def recording_classify_intent(inner: AgentState) -> dict[str, Route]:
        classified.append(inner["request"])
        return {"route": Route.COMPLAINT}

    monkeypatch.setattr(graph_module, "handle_complaint", handler)
    monkeypatch.setattr(graph_module, "classify_intent", recording_classify_intent)

    build_graph().invoke(state)

    # 수집 중에도 의도 분류를 거친다. 건너뛰면 다른 화제로 넘어갈 길이 없다.
    assert classified == [state["request"]]
    assert handled_requests == [state["request"]]


def test_graph_leaves_complaint_when_intent_reclassifies_mid_collection(
    monkeypatch: pytest.MonkeyPatch,
):
    state = _collecting_state()

    def unexpected_complaint(
        _req: ConverseRequest, **_kwargs: object
    ) -> dict[str, object]:
        raise AssertionError("complaint must not run once intent left the route")

    monkeypatch.setattr(graph_module, "handle_complaint", unexpected_complaint)
    monkeypatch.setattr(
        graph_module,
        "classify_intent",
        lambda _state: {"route": Route.KNOWLEDGE},
    )
    monkeypatch.setattr(
        graph_module,
        "handle_knowledge",
        lambda _req, **_kwargs: {
            "complaint_state": None,
            "reply": "22시까지 이용할 수 있습니다.",
            "result": RouteResult(),
        },
    )

    result = build_graph().invoke(state)

    assert result["route"] is Route.KNOWLEDGE
    assert result["complaint_state"] is None


def test_graph_finishes_on_clarify_route(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(graph_module, "classify_intent", _stub_classify_intent)

    result = build_graph().invoke(_make_state(None))

    assert result["route"] is Route.CLARIFY
    assert result["complaint_state"] is None
    assert result["reply"] is not None


@pytest.mark.parametrize("route", [Route.KNOWLEDGE, Route.CLARIFY])
def test_graph_clears_complaint_state_for_non_complaint_route(
    monkeypatch: pytest.MonkeyPatch,
    route: Route,
):
    monkeypatch.setattr(graph_module, "classify_intent", _stub_classify_intent)
    # 노드를 통째로 스텁으로 갈면 스텁이 넣은 None을 그대로 확인하게 된다.
    # 실제 노드를 태우고 바깥으로 나가는 호출만 막는다.
    monkeypatch.setattr(knowledge_node, "get_client", lambda: object())
    monkeypatch.setattr(knowledge_node, "query_encoder", lambda: object())
    monkeypatch.setattr(
        knowledge_node,
        "encode_question",
        lambda _question, *, encoder, trace_id: ([0.0] * EMBEDDING_DIM, {"7": 0.5}),
    )
    monkeypatch.setattr(knowledge_node, "search_chunks", lambda *_a, **_kw: [])

    result = build_graph().invoke(_make_state(route))

    assert result["complaint_state"] is None


def test_graph_propagates_intent_classification_failure(
    monkeypatch: pytest.MonkeyPatch,
):
    def failing_classify_intent(state: AgentState) -> dict[str, Route]:
        raise LlmUnavailableError("boom")

    monkeypatch.setattr(graph_module, "classify_intent", failing_classify_intent)

    with pytest.raises(LlmUnavailableError):
        build_graph().invoke(_make_state(None))


def test_graph_passes_image_analysis_to_selected_node(monkeypatch: pytest.MonkeyPatch):
    state = _make_state(Route.KNOWLEDGE)
    received: list[object] = []

    def classify(_state: AgentState) -> dict[str, object]:
        from zipsai.contracts.converse import ImageAnalysis, ImageObservation

        return {
            "route": Route.KNOWLEDGE,
            "image_analysis": ImageAnalysis(
                images=[ImageObservation(attachmentId=123, summary="세탁기")]
            ),
        }

    def knowledge(_request: ConverseRequest, *, image_analysis=None):
        received.append(image_analysis)
        return {"reply": "안내", "result": RouteResult()}

    monkeypatch.setattr(graph_module, "classify_intent", classify)
    monkeypatch.setattr(graph_module, "handle_knowledge", knowledge)

    result = build_graph().invoke(state)

    assert received[0].images[0].attachment_id == 123
    assert result["result"].image_analysis == received[0]


def test_graph_passes_intent_image_analysis_to_complaint(
    monkeypatch: pytest.MonkeyPatch,
):
    state = _make_state(Route.COMPLAINT)
    analysis = ImageAnalysis(
        images=[ImageObservation(attachmentId=123, summary="세탁기 아래 물이 고임")]
    )
    received: list[object] = []

    monkeypatch.setattr(
        graph_module,
        "classify_intent",
        lambda _state: {"route": Route.COMPLAINT, "image_analysis": analysis},
    )

    def complaint(_request: ConverseRequest, *, image_analysis=None):
        received.append(image_analysis)
        return {"reply": "확인", "result": RouteResult()}

    monkeypatch.setattr(graph_module, "handle_complaint", complaint)

    build_graph().invoke(state)

    assert received == [analysis]


@pytest.mark.parametrize("route", [Route.COMPLAINT, Route.KNOWLEDGE])
def test_graph_discloses_image_failure_while_answering_from_text(
    monkeypatch: pytest.MonkeyPatch, route: Route
):
    state = _make_state(route)
    monkeypatch.setattr(
        graph_module,
        "classify_intent",
        lambda _state: {
            "route": route,
            "image_analysis": None,
            "image_analysis_failed": True,
        },
    )
    handler_name = (
        "handle_complaint" if route is Route.COMPLAINT else "handle_knowledge"
    )
    monkeypatch.setattr(
        graph_module,
        handler_name,
        lambda *_args, **_kwargs: {
            "reply": "계속 안내합니다.",
            "result": RouteResult(),
        },
    )

    result = build_graph().invoke(state)

    assert result["reply"].startswith("사진을 확인하지 못했지만")
    assert result["reply"].endswith("계속 안내합니다.")


def test_graph_asks_to_retry_or_describe_when_image_only_analysis_fails(
    monkeypatch: pytest.MonkeyPatch,
):
    state = _make_state(Route.CLARIFY)
    state["request"] = state["request"].model_copy(
        update={
            "message": state["request"].message.model_copy(
                update={
                    "text": None,
                    "images": [
                        {"attachmentId": 123, "url": "https://example.com/image.jpg"}
                    ],
                }
            )
        }
    )
    monkeypatch.setattr(
        graph_module,
        "classify_intent",
        lambda _state: {
            "route": Route.CLARIFY,
            "image_analysis": None,
            "image_analysis_failed": True,
        },
    )

    result = build_graph().invoke(state)

    assert "사진을 확인하지 못했어요" in result["reply"]
    assert "다시 첨부" in result["reply"]
