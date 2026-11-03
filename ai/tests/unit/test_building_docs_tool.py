from qdrant_client import models

import zipsai.knowledge.tool as tool_module
from zipsai.contracts.converse import ConverseRequest
from zipsai.errors import EmbeddingError, VectorStoreError
from zipsai.orchestration.tool_contracts import (
    BuildingDocsArgs,
    BuildingDocsResult,
    ToolFailure,
)


def _request(building_id: int = 101) -> ConverseRequest:
    return ConverseRequest.model_validate(
        {
            "building_id": building_id,
            "room_no": "101",
            "resident_id": "resident-1",
            "conversation_id": "conversation-1",
            "turn_id": "turn-1",
            "trace_id": "trace-1",
            "current_route": None,
            "current_complaint_state": None,
            "message": {"message_id": "message-1", "text": "주차는 어떻게 하나요?"},
            "conversation_history": [],
            "complaint_draft": None,
        }
    )


def _chunk(payload: dict[str, object]) -> models.ScoredPoint:
    return models.ScoredPoint(id=1, version=0, score=0.9, payload=payload)


def _stub_encoder(monkeypatch):
    calls = []
    monkeypatch.setattr(tool_module, "query_encoder", lambda: "encoder")

    def encode(query, *, encoder, trace_id):
        calls.append((query, encoder, trace_id))
        return [0.1], {"1": 0.2}

    monkeypatch.setattr(tool_module, "encode_question", encode)
    monkeypatch.setattr(tool_module, "get_client", lambda: "client")
    return calls


def test_search_building_docs_uses_server_building_scope_and_returns_evidence(monkeypatch):
    calls = _stub_encoder(monkeypatch)
    search_calls = []
    monkeypatch.setattr(
        tool_module,
        "search_chunks",
        lambda vector, building_id, *, client: search_calls.append((vector, building_id, client))
        or [
            _chunk(
                {
                    "doc_id": "parking-guide",
                    "title": "주차 안내",
                    "text": "방문 차량은 관리실에 등록해야 합니다.",
                }
            )
        ],
    )

    result = tool_module.search_building_docs(BuildingDocsArgs(query="  방문 주차  "), _request())

    assert isinstance(result, BuildingDocsResult)
    assert result.status == "ok"
    assert calls == [("방문 주차", "encoder", "trace-1")]
    assert search_calls == [(([0.1], {"1": 0.2}), 101, "client")]
    assert result.evidence[0].text == "방문 차량은 관리실에 등록해야 합니다."
    assert result.evidence[0].citation.source_type == "building_document"
    assert result.evidence[0].citation.source_id == "parking-guide"


def test_search_building_docs_returns_no_results_without_evidence(monkeypatch):
    _stub_encoder(monkeypatch)
    monkeypatch.setattr(tool_module, "search_chunks", lambda *_args, **_kwargs: [])

    result = tool_module.search_building_docs(BuildingDocsArgs(query="없는 규정"), _request())

    assert result == BuildingDocsResult(status="no_results")


def test_search_building_docs_uses_only_safe_qa_answer(monkeypatch):
    _stub_encoder(monkeypatch)
    monkeypatch.setattr(
        tool_module,
        "search_chunks",
        lambda *_args, **_kwargs: [
            _chunk(
                {
                    "source_type": "qa_history",
                    "complaint_detail_id": 55,
                    "title": "과거 안내",
                    "text": "입주민의 개인 질문 원문",
                    "answer": "관리실에서 처리했습니다.",
                }
            )
        ],
    )

    result = tool_module.search_building_docs(BuildingDocsArgs(query="처리 상태"), _request())

    assert result.evidence[0].text == "관리실에서 처리했습니다."
    assert result.evidence[0].citation.source_type == "qa_history"
    assert result.evidence[0].citation.source_id == "55"


def test_search_building_docs_returns_unavailable_failure(monkeypatch):
    _stub_encoder(monkeypatch)
    monkeypatch.setattr(
        tool_module,
        "search_chunks",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(VectorStoreError("down")),
    )

    result = tool_module.search_building_docs(BuildingDocsArgs(query="주차"), _request())

    assert result == ToolFailure(code="unavailable", message="건물 문서를 검색할 수 없습니다.")


def test_search_building_docs_returns_unavailable_for_embedding_failure(monkeypatch):
    _stub_encoder(monkeypatch)
    monkeypatch.setattr(
        tool_module,
        "encode_question",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(EmbeddingError("down")),
    )

    result = tool_module.search_building_docs(BuildingDocsArgs(query="주차"), _request())

    assert result == ToolFailure(code="unavailable", message="건물 문서를 검색할 수 없습니다.")
