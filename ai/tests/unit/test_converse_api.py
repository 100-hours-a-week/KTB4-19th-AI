import pytest
from fastapi.testclient import TestClient
from qdrant_client import models

import zipsai.api.converse as converse_module
import zipsai.knowledge.node as knowledge_node
import zipsai.orchestration.graph as graph_module
from zipsai.contracts.converse import Route
from zipsai.errors import (
    EmbeddingError,
    LlmRateLimitedError,
    LlmTimeoutError,
    LlmUnavailableError,
    LlmUpstreamError,
    VectorStoreError,
)
from zipsai.main import app
from zipsai.settings import EMBEDDING_DIM, Settings


class _FakeGraph:
    def __init__(self, result: dict[str, object] | Exception):
        self.result = result
        self.received_state: dict[str, object] | None = None

    def invoke(self, state: dict[str, object]) -> dict[str, object]:
        self.received_state = state
        if isinstance(self.result, Exception):
            raise self.result
        return {**state, **self.result}


def _payload() -> dict[str, object]:
    return {
        "building_id": 1,
        "room_no": "301",
        "resident_id": "linda",
        "conversation_id": "conv-001",
        "trace_id": "trace-001",
        "current_route": None,
        "current_complaint_state": None,
        "message": {"message_id": "msg-001", "text": "네", "image_urls": []},
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
        "trace_id": "trace-001",
        "data": {
            "route": "clarify",
            "complaint_intent": None,
            "next_complaint_state": None,
            "reply": "민원/시설 문제인가요, 건물 정보 질문인가요?",
            "result": {
                "draft_patch": None,
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
    payload = _payload()
    payload["current_route"] = "complaint"
    payload["current_complaint_state"] = "collecting"
    payload["message"] = {
        "message_id": "msg-001",
        "text": "민원 접수해주세요.",
        "image_urls": [],
    }

    response = TestClient(app).post("/api/v3/ai/converse", json=payload)

    assert response.status_code == 200
    assert response.json()["data"] == {
        "route": "complaint",
        "complaint_intent": None,
        "next_complaint_state": "collecting",
        "reply": "민원 접수를 위해 발생 위치와 불편 증상을 알려주세요.",
        "result": {
            "draft_patch": None,
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
        lambda _question, *, encoder: ([0.0] * EMBEDDING_DIM, {"7": 0.5}),
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

    response = TestClient(app).post("/api/v3/ai/converse", json=_payload())

    assert response.status_code == 200
    assert response.json()["data"]["route"] == "knowledge"
    assert response.json()["data"]["reply"] == "22시까지 이용할 수 있습니다."


def test_converse_stays_in_complaint_without_consulting_intent_when_state_in_progress(
    monkeypatch,
):
    def failing_classify_intent(_state: object) -> dict[str, object]:
        raise AssertionError("classify_intent must not run mid-complaint")

    monkeypatch.setattr(graph_module, "classify_intent", failing_classify_intent)
    monkeypatch.setattr(
        converse_module,
        "get_settings",
        lambda: Settings("key", None, "test-model", 30),
    )
    payload = _payload()
    payload["current_route"] = "complaint"
    payload["current_complaint_state"] = "collecting"

    response = TestClient(app, raise_server_exceptions=False).post(
        "/api/v3/ai/converse", json=payload
    )

    assert response.status_code == 200
    assert response.json()["data"]["route"] == "complaint"


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
        "image_urls": [],
    }

    response = TestClient(app).post("/api/v3/ai/converse", json=payload)

    assert response.status_code == 400
    assert response.json() == {
        "message": "ai_response_error",
        "error": {
            "code": "MISSING_REQUIRED_FIELD",
            "detail": "A message requires text or image_urls",
            "retryable": False,
        },
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
