import asyncio
import json
import logging
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY
from qdrant_client import models

import zipsai.api.converse as converse_module
import zipsai.complaint.node as complaint_node
import zipsai.knowledge.node as knowledge_node
import zipsai.orchestration.graph as graph_module
import zipsai.orchestration.intent as intent_module
from zipsai.contracts.converse import ConverseRequest, Route
from zipsai.errors import (
    ChunkTooLongError,
    ComplaintExtractionError,
    EmbeddingError,
    EmbeddingRequestInvalidError,
    LlmRateLimitedError,
    LlmTimeoutError,
    LlmUnavailableError,
    LlmUpstreamError,
    VectorStoreError,
)
from zipsai.integrations.llm import ToolCall, ToolDecision
from zipsai.main import app
from zipsai.observability import _ContextFilter, collect_timings, note_llm_cost
from zipsai.orchestration.tool_contracts import BuildingDocsResult
from zipsai.settings import EMBEDDING_DIM, Settings


class _FakeGraph:
    def __init__(
        self,
        result: dict[str, object] | Exception,
        custom_events: list[str] | None = None,
    ):
        self.result = result
        self.custom_events = custom_events or []
        self.received_state: dict[str, object] | None = None

    def invoke(self, state: dict[str, object]) -> dict[str, object]:
        self.received_state = state
        if isinstance(self.result, Exception):
            raise self.result
        return {**state, **self.result}

    async def astream(self, state: dict[str, object], *, stream_mode: list[str]):
        self.received_state = state
        for event in self.custom_events:
            yield "custom", event
        if isinstance(self.result, Exception):
            raise self.result
        yield "values", {**state, **self.result}


def _payload() -> dict[str, object]:
    return {
        "building_id": 1,
        "room_no": "301",
        "resident_id": "linda",
        "conversation_id": "conv-001",
        "turn_id": "turn-001",
        "trace_id": "trace-001",
        "current_route": None,
        "current_complaint_state": None,
        "message": {"message_id": "msg-001", "text": "네"},
        "conversation_history": [],
        "complaint_draft": None,
    }


def test_converse_invokes_graph_and_returns_ai_contract(monkeypatch):
    graph = _FakeGraph(
        {
            "route": Route.CLARIFY,
            "complaint_state": None,
            "reply": "민원/시설 문제인가요, 건물 정보 질문인가요?",
        }
    )
    monkeypatch.setattr(converse_module, "build_graph", lambda: graph, raising=False)
    monkeypatch.setattr(
        converse_module,
        "get_settings",
        lambda: Settings("key", None, "test-model", 30),
        raising=False,
    )

    response = TestClient(app).post("/api/v3/ai/converse", json=_payload())

    assert response.status_code == 200
    assert response.json() == {
        "code": "ai_response_success",
        "turn_id": "turn-001",
        "trace_id": "trace-001",
        "data": {
            "route": "clarify",
            "complaint_intent": None,
            "next_complaint_state": None,
            "reply": "민원/시설 문제인가요, 건물 정보 질문인가요?",
            "result": {
                "complaint_draft": None,
                "qa_card_draft": None,
                "missing_fields": [],
                "citations": [],
                "has_sufficient_evidence": None,
                "image_analysis": None,
            },
            "meta": {
                "model": "test-model",
                "timing_ms": response.json()["data"]["meta"]["timing_ms"],
            },
        },
    }
    assert graph.received_state is not None
    assert graph.received_state["complaint_state"] is None


def test_converse_times_out_returns_request_timeout(monkeypatch):
    class _SlowGraph:
        def invoke(self, state: dict[str, object]) -> dict[str, object]:
            time.sleep(0.2)
            return state

    monkeypatch.setattr(converse_module, "build_graph", lambda: _SlowGraph(), raising=False)
    monkeypatch.setattr(converse_module, "CONVERSE_TIMEOUT_SECONDS", 0.05)

    response = TestClient(app).post("/api/v3/ai/converse", json=_payload())

    assert response.status_code == 504
    assert response.json()["error"]["code"] == "REQUEST_TIMEOUT"


def test_converse_stream_times_out_emits_error_and_done(monkeypatch):
    class _SlowStreamGraph:
        async def astream(self, state: dict[str, object], *, stream_mode: list[str]):
            await asyncio.sleep(0.2)
            yield "values", {
                **state,
                "route": Route.CLARIFY,
                "complaint_state": None,
                "reply": "늦은 응답",
            }

    monkeypatch.setattr(
        converse_module, "build_graph", lambda streaming=False: _SlowStreamGraph()
    )
    monkeypatch.setattr(converse_module, "CONVERSE_TIMEOUT_SECONDS", 0.05)

    response = TestClient(app).post("/api/v3/ai/converse/stream", json=_payload())

    assert response.status_code == 200
    assert '"type": "error"' in response.text
    assert response.text.endswith('data: {"type": "done"}\n\n')


def test_converse_stream_returns_sse_status_reply_and_done(monkeypatch):
    graph = _FakeGraph(
        {
            "route": Route.KNOWLEDGE,
            "complaint_state": None,
            "reply": "건물 이용 안내입니다.",
        }
    )
    monkeypatch.setattr(
        converse_module, "build_graph", lambda streaming=False: graph
    )
    monkeypatch.setattr(
        converse_module,
        "get_settings",
        lambda: Settings("key", None, "test-model", 30),
    )

    response = TestClient(app).post("/api/v3/ai/converse/stream", json=_payload())

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    assert [event["type"] for event in events] == [
        "status",
        "delta",
        "complete",
        "done",
    ]
    complete = events[2]["response"]
    assert complete["code"] == "ai_response_success"
    assert complete["turn_id"] == "turn-001"
    assert complete["trace_id"] == "trace-001"
    assert complete["data"]["route"] == "knowledge"
    assert complete["data"]["reply"] == "건물 이용 안내입니다."
    assert complete["data"]["meta"]["model"] == "test-model"
    assert isinstance(complete["data"]["meta"]["timing_ms"], int)


def test_converse_stream_forwards_multiple_custom_deltas(monkeypatch):
    graph = _FakeGraph(
        {
            "route": Route.KNOWLEDGE,
            "complaint_state": None,
            "reply": "건물 이용 안내입니다.",
        },
        custom_events=["건물 이용", " 안내입니다."],
    )
    monkeypatch.setattr(
        converse_module, "build_graph", lambda streaming=False: graph
    )
    monkeypatch.setattr(
        converse_module,
        "get_settings",
        lambda: Settings("key", None, "test-model", 30),
    )

    response = TestClient(app).post("/api/v3/ai/converse/stream", json=_payload())

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["x-accel-buffering"] == "no"
    assert response.text.count('"type": "delta"') == 2
    assert '"type": "complete"' in response.text
    assert response.text.endswith('data: {"type": "done"}\n\n')


def test_converse_stream_terminates_after_error(monkeypatch):
    graph = _FakeGraph(LlmUnavailableError("LLM request failed"))
    monkeypatch.setattr(
        converse_module, "build_graph", lambda streaming=False: graph
    )

    response = TestClient(app).post("/api/v3/ai/converse/stream", json=_payload())

    assert response.status_code == 200
    assert '"type": "error"' in response.text
    assert response.text.endswith('data: {"type": "done"}\n\n')


def test_converse_stream_terminates_after_unexpected_error(monkeypatch):
    graph = _FakeGraph(RuntimeError("unexpected stream failure"), custom_events=["앞부분"])
    monkeypatch.setattr(
        converse_module, "build_graph", lambda streaming=False: graph
    )
    monkeypatch.setattr(
        converse_module,
        "get_settings",
        lambda: Settings("key", None, "test-model", 30),
    )

    response = TestClient(app).post("/api/v3/ai/converse/stream", json=_payload())

    assert response.status_code == 200
    assert '"type": "delta"' in response.text
    assert '"type": "error"' in response.text
    assert response.text.endswith('data: {"type": "done"}\n\n')


def _converse_count(route: str, outcome: str) -> float:
    return (
        REGISTRY.get_sample_value(
            "zipsai_converse_duration_seconds_count",
            {"route": route, "outcome": outcome},
        )
        or 0.0
    )


def _stream_done(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [
        record
        for record in caplog.records
        if record.getMessage() == "stream_request_done"
    ]


def test_stream_success_records_ok(monkeypatch, caplog):
    graph = _FakeGraph(
        {
            "route": Route.KNOWLEDGE,
            "complaint_state": None,
            "reply": "안내",
        }
    )
    monkeypatch.setattr(converse_module, "build_graph", lambda streaming=False: graph)
    monkeypatch.setattr(
        converse_module,
        "get_settings",
        lambda: Settings("key", None, "test-model", 30),
    )
    before = _converse_count("knowledge", "ok")

    with caplog.at_level(logging.INFO, logger="zipsai.api.converse"):
        response = TestClient(app).post("/api/v3/ai/converse/stream", json=_payload())

    assert response.status_code == 200
    recorded = _stream_done(caplog)
    assert recorded[-1].outcome == "ok"
    assert recorded[-1].intent_route == "knowledge"
    assert recorded[-1].error_code is None
    assert "네" not in recorded[-1].getMessage()
    assert _converse_count("knowledge", "ok") == before + 1


def test_both_converse_paths_bind_conversation_id(monkeypatch, caplog):
    graph = _FakeGraph(
        {
            "route": Route.CLARIFY,
            "complaint_state": None,
            "reply": "안내",
        }
    )
    monkeypatch.setattr(converse_module, "build_graph", lambda streaming=False: graph)
    monkeypatch.setattr(
        converse_module,
        "get_settings",
        lambda: Settings("key", None, "test-model", 30),
    )
    logger = logging.getLogger("zipsai.api.converse")
    context_filter = _ContextFilter()
    logger.addFilter(context_filter)
    try:
        with caplog.at_level(logging.INFO, logger="zipsai.api.converse"):
            assert (
                TestClient(app).post("/api/v3/ai/converse", json=_payload()).status_code
                == 200
            )
            assert (
                TestClient(app)
                .post("/api/v3/ai/converse/stream", json=_payload())
                .status_code
                == 200
            )
    finally:
        logger.removeFilter(context_filter)

    started = [
        record
        for record in caplog.records
        if record.getMessage() in {"request_started", "stream_request_started"}
    ]
    assert {record.getMessage() for record in started} == {
        "request_started",
        "stream_request_started",
    }
    assert all(record.conversation_id == "conv-001" for record in started)


def test_request_totals_sum_priced_calls(caplog: pytest.LogCaptureFixture) -> None:
    with (
        collect_timings() as timings,
        caplog.at_level(logging.INFO, logger="zipsai.api.converse"),
    ):
        note_llm_cost(0.25)
        note_llm_cost(0.75)
        converse_module._request_done(
            0.0, timings, {"in_flight": 1}, outcome="ok", intent_route="knowledge"
        )
        converse_module._stream_request_done(
            0.0, timings, 1, outcome="ok", intent_route="knowledge"
        )

    done = [
        record
        for record in caplog.records
        if record.getMessage() in {"request_done", "stream_request_done"}
    ]
    assert {record.getMessage() for record in done} == {
        "request_done",
        "stream_request_done",
    }
    assert all(record.cost_usd_total == 1.0 for record in done)


@pytest.mark.parametrize(
    ("result", "error_type"),
    [
        (LlmUnavailableError("LLM request failed"), "LlmUnavailableError"),
        (RuntimeError("unexpected stream failure"), "RuntimeError"),
    ],
)
def test_stream_failures_record_fail(monkeypatch, caplog, result, error_type):
    graph = _FakeGraph(result)
    monkeypatch.setattr(converse_module, "build_graph", lambda streaming=False: graph)
    monkeypatch.setattr(
        converse_module,
        "get_settings",
        lambda: Settings("key", None, "test-model", 30),
    )
    before = _converse_count("unknown", "fail")

    with caplog.at_level(logging.INFO, logger="zipsai.api.converse"):
        response = TestClient(app).post("/api/v3/ai/converse/stream", json=_payload())

    assert response.status_code == 200
    recorded = _stream_done(caplog)
    assert recorded[-1].outcome == "fail"
    assert recorded[-1].error_type == error_type
    assert recorded[-1].error_code
    assert recorded[-1].total_ms >= 0
    assert _converse_count("unknown", "fail") == before + 1


def test_stream_without_reply_records_fail(monkeypatch, caplog):
    graph = _FakeGraph(
        {"route": Route.KNOWLEDGE, "complaint_state": None, "reply": None}
    )
    monkeypatch.setattr(converse_module, "build_graph", lambda streaming=False: graph)
    monkeypatch.setattr(
        converse_module,
        "get_settings",
        lambda: Settings("key", None, "test-model", 30),
    )
    before = _converse_count("knowledge", "fail")

    with caplog.at_level(logging.INFO, logger="zipsai.api.converse"):
        response = TestClient(app).post("/api/v3/ai/converse/stream", json=_payload())

    assert '"type": "error"' in response.text
    recorded = _stream_done(caplog)
    assert recorded[-1].outcome == "fail"
    assert recorded[-1].error_code == "INTERNAL_SERVER_ERROR"
    assert recorded[-1].intent_route == "knowledge"
    assert _converse_count("knowledge", "fail") == before + 1


def test_closed_stream_records_cancelled(monkeypatch, caplog):
    monkeypatch.setattr(
        converse_module,
        "get_settings",
        lambda: Settings("key", None, "test-model", 30),
    )
    before = _converse_count("unknown", "cancelled")
    request = ConverseRequest.model_validate(_payload())

    async def close_early() -> None:
        stream = converse_module._stream_converse(request)
        await anext(stream)
        await stream.aclose()

    with caplog.at_level(logging.INFO, logger="zipsai.api.converse"):
        asyncio.run(close_early())

    recorded = _stream_done(caplog)
    assert recorded[-1].outcome == "cancelled"
    assert recorded[-1].error_type == "GeneratorExit"
    assert _converse_count("unknown", "cancelled") == before + 1


def test_request_done_metric_drops_unknown_outcomes():
    before = _converse_count("knowledge", "fail")
    converse_module._request_done(
        0,
        {},
        {"in_flight": 0},
        outcome="weird",
        status_code=500,
        intent_route="knowledge",
    )
    assert _converse_count("knowledge", "fail") == before + 1


@pytest.mark.parametrize(
    ("error", "status_code", "error_body"),
    [
        pytest.param(
            LlmUnavailableError("LLM request failed"),
            503,
            {
                "code": "DEPENDENCY_NOT_READY",
                "detail": "AI model is unavailable",
                "retryable": True,
            },
            id="unavailable",
        ),
        pytest.param(
            LlmRateLimitedError("LLM rate limit exceeded", retry_after_seconds=12),
            429,
            {
                "code": "MODEL_RATE_LIMITED",
                "detail": "AI model rate limit exceeded",
                "retryable": True,
                "retry_after_seconds": 12,
            },
            id="rate-limited",
        ),
        pytest.param(
            LlmTimeoutError("LLM request timed out"),
            504,
            {
                "code": "MODEL_TIMEOUT",
                "detail": "AI model did not respond in time",
                "retryable": True,
            },
            id="timeout",
        ),
        pytest.param(
            LlmUpstreamError("LLM provider returned an upstream error"),
            502,
            {
                "code": "MODEL_UPSTREAM_ERROR",
                "detail": "AI model provider returned an upstream error",
                "retryable": True,
            },
            id="upstream",
        ),
        pytest.param(
            ComplaintExtractionError("LLM returned an invalid complaint draft"),
            502,
            {
                "code": "MODEL_UPSTREAM_ERROR",
                "detail": "AI model returned an invalid complaint draft",
                "retryable": True,
            },
            id="complaint-extraction",
        ),
        # 위키 §9 — 의존 컨테이너가 죽으면 500이 아니라 503이 나가야 한다.
        pytest.param(
            EmbeddingError("Embedding request failed"),
            503,
            {
                "code": "DEPENDENCY_NOT_READY",
                "detail": "Embedding service is unavailable",
                "retryable": True,
            },
            id="embedding-down",
        ),
        pytest.param(
            ChunkTooLongError(0, 8001, 8000),
            503,
            {
                "code": "CHUNK_TOO_LONG",
                "detail": "Embedding service is unavailable",
                "retryable": True,
            },
            id="chunk-too-long",
        ),
        pytest.param(
            EmbeddingRequestInvalidError(422),
            503,
            {
                "code": "EMBEDDING_REQUEST_INVALID",
                "detail": "Embedding service is unavailable",
                "retryable": True,
            },
            id="embedding-request-invalid",
        ),
        pytest.param(
            VectorStoreError("Vector store query failed"),
            503,
            {
                "code": "DEPENDENCY_NOT_READY",
                "detail": "Vector store is unavailable",
                "retryable": True,
            },
            id="qdrant-down",
        ),
    ],
)
def test_converse_maps_llm_error_to_api_response(
    monkeypatch,
    error,
    status_code,
    error_body,
):
    monkeypatch.setattr(
        converse_module,
        "build_graph",
        lambda: _FakeGraph(error),
        raising=False,
    )
    monkeypatch.setattr(
        converse_module,
        "get_settings",
        lambda: Settings("key", None, "test-model", 30),
        raising=False,
    )

    response = TestClient(app).post("/api/v3/ai/converse", json=_payload())

    assert response.status_code == status_code
    assert response.json() == {
        "message": "ai_response_error",
        "error": error_body,
        "turn_id": "turn-001",
        "trace_id": "trace-001",
    }


def test_converse_returns_dependency_error_when_llm_settings_are_missing(monkeypatch):
    monkeypatch.setattr(
        converse_module,
        "build_graph",
        lambda: _FakeGraph(
            {
                "route": Route.CLARIFY,
                "complaint_state": None,
                "reply": "민원/시설 문제인가요, 건물 정보 질문인가요?",
            }
        ),
    )
    monkeypatch.setattr(
        converse_module,
        "get_settings",
        lambda: (_ for _ in ()).throw(LlmUnavailableError("missing settings")),
    )

    response = TestClient(app, raise_server_exceptions=False).post(
        "/api/v3/ai/converse", json=_payload()
    )

    assert response.status_code == 503
    assert response.json() == {
        "message": "ai_response_error",
        "error": {
            "code": "DEPENDENCY_NOT_READY",
            "detail": "AI model is unavailable",
            "retryable": True,
        },
        "turn_id": "turn-001",
        "trace_id": "trace-001",
    }


def test_converse_runs_clarify_flow_through_real_graph(monkeypatch):
    monkeypatch.setattr(
        graph_module,
        "classify_intent",
        lambda _: {"route": Route.CLARIFY},
    )
    monkeypatch.setattr(
        converse_module,
        "get_settings",
        lambda: Settings("key", None, "test-model", 30),
    )
    monkeypatch.setattr(
        graph_module,
        "generate_tool_decision",
        lambda *_args, **_kwargs: ToolDecision(
            SimpleNamespace(),
            "어떤 것을 도와드릴까요? 민원/시설 문제인가요, 건물 정보 질문인가요?",
            (),
        ),
    )

    response = TestClient(app).post("/api/v3/ai/converse", json=_payload())

    assert response.status_code == 200
    assert response.json()["data"]["route"] == "clarify"
    assert response.json()["data"]["reply"] == (
        "어떤 것을 도와드릴까요? 민원/시설 문제인가요, 건물 정보 질문인가요?"
    )


def test_converse_returns_collecting_reply_for_incomplete_complaint(monkeypatch):
    monkeypatch.setattr(
        graph_module,
        "classify_intent",
        lambda _: {"route": Route.COMPLAINT},
    )
    monkeypatch.setattr(
        converse_module,
        "get_settings",
        lambda: Settings("key", None, "test-model", 30),
    )
    monkeypatch.setattr(
        complaint_node,
        "generate_structured",
        lambda *, response_format, **_: (
            complaint_node._InterpretationOutput(
                complaint_switch="same",
                issue_type=None,
                location=None,
                symptom=None,
                occurred_at=None,
            )
            if response_format is complaint_node._InterpretationOutput
            else None
        ),
    )
    monkeypatch.setattr(
        graph_module,
        "generate_tool_decision",
        lambda *_args, **_kwargs: ToolDecision(
            SimpleNamespace(),
            None,
            (ToolCall("call-1", "process_complaint_turn", {}),),
        ),
    )
    payload = _payload()
    payload["current_route"] = "complaint"
    payload["current_complaint_state"] = "collecting"
    payload["message"] = {
        "message_id": "msg-001",
        "text": "민원 접수해주세요.",
    }

    response = TestClient(app).post("/api/v3/ai/converse", json=payload)

    assert response.status_code == 200
    assert response.json()["data"] == {
        "route": "complaint",
        "complaint_intent": None,
        "next_complaint_state": "collecting",
        "reply": "어디에서 생긴 문제인가요?",
        "result": {
            "complaint_draft": {
                "issue_type": None,
                "location": None,
                "symptom": None,
                "occurred_at": None,
                "attachmentIds": [],
            },
            "qa_card_draft": None,
            "missing_fields": ["location", "symptom"],
            "citations": [],
            "has_sufficient_evidence": None,
            "image_analysis": None,
        },
        "meta": {
            "model": "test-model",
            "timing_ms": response.json()["data"]["meta"]["timing_ms"],
        },
    }


def test_converse_returns_document_backed_reply_for_knowledge(monkeypatch):
    monkeypatch.setattr(
        graph_module,
        "classify_intent",
        lambda _: {"route": Route.KNOWLEDGE},
    )
    monkeypatch.setattr(
        converse_module,
        "get_settings",
        lambda: Settings("key", None, "test-model", 30),
    )
    monkeypatch.setattr(knowledge_node, "get_client", lambda: object())
    monkeypatch.setattr(knowledge_node, "query_encoder", lambda: object())
    monkeypatch.setattr(
        knowledge_node,
        "encode_question",
        lambda _question, *, encoder, trace_id: ([0.0] * EMBEDDING_DIM, {"7": 0.5}),
    )
    monkeypatch.setattr(
        knowledge_node,
        "search_chunks",
        lambda *_args, **_kwargs: [
            models.ScoredPoint(
                id=1,
                version=0,
                score=0.9,
                payload={"title": "세탁실 이용", "text": "세탁실은 22시까지입니다."},
            )
        ],
    )
    monkeypatch.setattr(
        knowledge_node,
        "generate_text",
        lambda **_kwargs: "22시까지 이용할 수 있습니다.",
    )
    decisions = iter(
        [
            ToolDecision(
                SimpleNamespace(),
                None,
                (ToolCall("call-1", "search_building_docs", {"query": "세탁실"}),),
            ),
            ToolDecision(SimpleNamespace(), "22시까지 이용할 수 있습니다.", ()),
        ]
    )
    monkeypatch.setattr(
        graph_module,
        "generate_tool_decision",
        lambda *_args, **_kwargs: next(decisions),
    )
    monkeypatch.setattr(
        graph_module,
        "search_building_docs",
        lambda *_args: BuildingDocsResult(status="no_results"),
    )

    response = TestClient(app).post("/api/v3/ai/converse", json=_payload())

    assert response.status_code == 200
    assert response.json()["data"]["route"] == "knowledge"
    assert response.json()["data"]["reply"] == "22시까지 이용할 수 있습니다."


def test_converse_consults_intent_even_when_complaint_state_in_progress(
    monkeypatch, caplog
):
    monkeypatch.setattr(
        intent_module,
        "generate_structured",
        lambda *_args, **_kwargs: intent_module._RouteDecision(
            route=Route.COMPLAINT, contextualized_message=None
        ),
    )
    monkeypatch.setattr(
        converse_module,
        "get_settings",
        lambda: Settings("key", None, "test-model", 30),
    )
    monkeypatch.setattr(
        complaint_node,
        "generate_structured",
        lambda *, response_format, **_: (
            complaint_node._InterpretationOutput(
                complaint_switch="same",
                issue_type=None,
                location=None,
                symptom=None,
                occurred_at=None,
            )
            if response_format is complaint_node._InterpretationOutput
            else None
        ),
    )
    monkeypatch.setattr(
        graph_module,
        "generate_tool_decision",
        lambda *_args, **_kwargs: ToolDecision(
            SimpleNamespace(),
            None,
            (ToolCall("call-1", "process_complaint_turn", {}),),
        ),
    )
    payload = _payload()
    payload["current_route"] = "complaint"
    payload["current_complaint_state"] = "collecting"

    # 운영에서는 configure_logging이 핸들러에 붙이는 필터를 여기서는 로거에 심는다.
    loggers = [
        logging.getLogger(name)
        for name in ("zipsai.orchestration.intent", "zipsai.complaint.node")
    ]
    context_filter = _ContextFilter()
    for target in loggers:
        target.addFilter(context_filter)
    try:
        with caplog.at_level(logging.INFO):
            response = TestClient(app, raise_server_exceptions=False).post(
                "/api/v3/ai/converse", json=payload
            )
    finally:
        for target in loggers:
            target.removeFilter(context_filter)

    assert response.status_code == 200
    assert response.json()["data"]["route"] == "complaint"
    stages = {
        record.stage: record for record in caplog.records if record.msg == "stage_done"
    }
    # 수집 중이라고 건너뛰지 않는다. 건너뛰면 intent 단계가 통째로 관측에서 빠진다.
    assert stages["intent"].outcome == "ok"
    assert stages["intent"].intent_route == "complaint"
    # 재분류가 돈 경로도 단계 로그가 경로를 담아야 turn_id 조회가 이어진다.
    assert {record.intent_route for record in stages.values()} == {"complaint"}


def test_converse_rejects_empty_message_before_graph_invocation(monkeypatch):
    monkeypatch.setattr(
        converse_module,
        "build_graph",
        lambda: _FakeGraph(
            {
                "route": Route.CLARIFY,
                "complaint_state": None,
                "reply": "should not be returned",
            }
        ),
    )
    payload = _payload()
    payload["message"] = {
        "message_id": "msg-001",
        "text": None,
    }

    response = TestClient(app).post("/api/v3/ai/converse", json=payload)

    assert response.status_code == 400
    assert response.json() == {
        "message": "ai_response_error",
        "error": {
            "code": "MISSING_REQUIRED_FIELD",
            "detail": "A message requires text or images",
            "retryable": False,
        },
        "turn_id": "turn-001",
        "trace_id": "trace-001",
    }


def test_converse_returns_standard_error_for_contract_violation():
    payload = _payload()
    payload["current_route"] = "knowledge"
    payload["current_complaint_state"] = "collecting"

    response = TestClient(app).post("/api/v3/ai/converse", json=payload)

    assert response.status_code == 422
    assert response.json() == {
        "message": "ai_response_error",
        "error": {
            "code": "VALIDATION_ERROR",
            "detail": "Request validation failed",
            "retryable": False,
        },
        "turn_id": "turn-001",
        "trace_id": "trace-001",
    }


def test_converse_returns_standard_error_for_missing_required_field():
    payload = _payload()
    del payload["building_id"]

    response = TestClient(app).post("/api/v3/ai/converse", json=payload)

    assert response.status_code == 400
    assert response.json() == {
        "message": "ai_response_error",
        "error": {
            "code": "MISSING_REQUIRED_FIELD",
            "detail": "Request validation failed",
            "retryable": False,
        },
        "turn_id": "turn-001",
        "trace_id": "trace-001",
    }


def test_converse_returns_generic_error_for_unexpected_exception(monkeypatch):
    monkeypatch.setattr(
        graph_module,
        "classify_intent",
        lambda _: (_ for _ in ()).throw(RuntimeError("completely unexpected bug")),
    )
    monkeypatch.setattr(
        converse_module,
        "get_settings",
        lambda: Settings("key", None, "test-model", 30),
        raising=False,
    )

    response = TestClient(app, raise_server_exceptions=False).post(
        "/api/v3/ai/converse", json=_payload()
    )

    assert response.status_code == 500
    assert response.json() == {
        "message": "ai_response_error",
        "error": {
            "code": "INTERNAL_SERVER_ERROR",
            "detail": "Unexpected server error",
            "retryable": False,
        },
        "turn_id": "turn-001",
        "trace_id": "trace-001",
    }


def test_converse_returns_bad_request_for_malformed_json():
    response = TestClient(app).post(
        "/api/v3/ai/converse",
        content="{",
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 400
    assert response.json() == {
        "message": "ai_response_error",
        "error": {
            "code": "MISSING_REQUIRED_FIELD",
            "detail": "Request validation failed",
            "retryable": False,
        },
    }
