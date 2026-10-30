import pytest
from pydantic import ValidationError

from zipsai.contracts.converse import ComplaintState, RouteResult
from zipsai.orchestration.tool_contracts import (
    TOOL_CONTRACTS,
    BuildingDocsArgs,
    BuildingDocsResult,
    ComplaintTurnArgs,
    ComplaintTurnResult,
    SearchEvidence,
    SearchResult,
    ToolFailure,
)


def test_tool_contracts_expose_current_workflow_schemas():
    assert [contract.name for contract in TOOL_CONTRACTS] == [
        "search_building_docs",
        "process_complaint_turn",
    ]
    for contract in TOOL_CONTRACTS:
        tool = contract.model_spec()
        assert tool["type"] == "function"
        assert tool["function"]["name"] == contract.name
        assert tool["function"]["description"]
        assert tool["function"]["parameters"]["additionalProperties"] is False
    assert TOOL_CONTRACTS[-1].terminal is True
    assert all(not contract.terminal for contract in TOOL_CONTRACTS[:-1])


def test_search_args_accept_query_only():
    assert BuildingDocsArgs.model_validate({"query": "  분리수거 시간  "}).query == "분리수거 시간"
    for payload in ({"query": "  "}, {"query": "시설", "building_id": 2}):
        with pytest.raises(ValidationError):
            BuildingDocsArgs.model_validate(payload)


def test_complaint_args_reject_model_supplied_server_state():
    assert ComplaintTurnArgs.model_validate({}).model_dump() == {}
    with pytest.raises(ValidationError):
        ComplaintTurnArgs.model_validate({"complaint_draft": {}})


def test_search_result_distinguishes_evidence_no_results_and_error():
    evidence = SearchEvidence.model_validate(
        {
            "text": "분리수거는 화요일입니다.",
            "citation": {
                "source_type": "building_document",
                "source_id": "doc-1",
                "title": "분리수거 안내",
                "snippet": "분리수거는 화요일입니다.",
                "url": None,
            },
        }
    )
    assert SearchResult(status="ok", evidence=[evidence]).evidence == [evidence]
    assert SearchResult(status="no_results").evidence == []
    with pytest.raises(ValidationError):
        SearchResult(status="ok")
    with pytest.raises(ValidationError):
        SearchResult(status="no_results", evidence=[evidence])
    assert ToolFailure(code="timeout", message="검색 시간이 초과됐습니다.").status == "error"


def test_complaint_result_preserves_existing_reply_and_state_contract():
    result = ComplaintTurnResult(
        reply="위치를 알려주세요.",
        complaint_state=ComplaintState.COLLECTING,
        result=RouteResult(missing_fields=["location"]),
    )
    assert result.result.missing_fields == ["location"]
    with pytest.raises(ValidationError):
        ComplaintTurnResult.model_validate({"reply": "위치를 알려주세요."})


@pytest.mark.parametrize(
    ("source_type", "valid"),
    [
        ("building_document", True),
        ("qa_history", True),
        ("web", False),
    ],
)
def test_search_result_rejects_wrong_source(source_type, valid):
    payload = {
        "status": "ok",
        "evidence": [
            {
                "text": "근거 내용",
                "citation": {
                    "source_type": source_type,
                    "source_id": "source-1",
                    "title": "근거 제목",
                    "snippet": "근거 내용",
                    "url": None,
                },
            }
        ],
    }
    if valid:
        assert BuildingDocsResult.model_validate(payload).evidence
    else:
        with pytest.raises(ValidationError):
            BuildingDocsResult.model_validate(payload)
