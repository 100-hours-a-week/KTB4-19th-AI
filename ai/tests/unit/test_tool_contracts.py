import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from zipsai.contracts.comment import CommentItem
from zipsai.contracts.converse import (
    ComplaintDraft,
    ComplaintState,
    ConverseRequest,
    RouteResult,
)
from zipsai.orchestration.tool_contracts import (
    TOOL_CONTRACTS,
    BuildingDocsArgs,
    BuildingDocsResult,
    ComplaintTurnArgs,
    ComplaintTurnResult,
    SearchEvidence,
    SearchResult,
    ToolFailure,
    WebSearchArgs,
    WebSearchResult,
)


def test_tool_contracts_expose_current_workflow_schemas():
    assert [contract.name for contract in TOOL_CONTRACTS] == [
        "search_building_docs",
        "process_complaint_turn",
        "search_public_web",
    ]
    for contract in TOOL_CONTRACTS:
        tool = contract.model_spec()
        assert tool["type"] == "function"
        assert tool["function"]["name"] == contract.name
        assert tool["function"]["description"]
        assert tool["function"]["parameters"]["additionalProperties"] is False
    assert all(not hasattr(contract, "terminal") for contract in TOOL_CONTRACTS)


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
    "issue_type",
    [
        "plumbing",
        "heating_cooling",
        "interior_repair",
        "electrical",
        "appliance",
        "hygiene",
        "common_area",
        "noise",
        "other",
    ],
)
def test_complaint_draft_accepts_the_resident_maintenance_taxonomy(issue_type):
    assert ComplaintDraft(issue_type=issue_type).issue_type == issue_type


@pytest.mark.parametrize(
    "issue_type",
    ["water_supply", "drain", "leak", "heating", "electricity", "mold", "pest", "facility"],
)
def test_new_complaint_draft_rejects_retired_issue_types(issue_type):
    with pytest.raises(ValidationError):
        ComplaintDraft(issue_type=issue_type)


@pytest.mark.parametrize(
    "issue_type",
    [
        "plumbing",
        "heating_cooling",
        "interior_repair",
        "electrical",
        "appliance",
        "hygiene",
        "common_area",
        "noise",
        "other",
        "water_supply",
        "drain",
        "leak",
        "heating",
        "electricity",
        "mold",
        "pest",
        "facility",
    ],
)
def test_comment_ingestion_accepts_new_and_historical_issue_types(issue_type):
    comment = CommentItem.model_validate(
        {
            "complaint_detail_id": 1,
            "building_id": 1,
            "complaint_type": "COMPLAINT",
            "symptom": "욕실 천장에서 물이 떨어져요",
            "comment": "접수했습니다.",
            "location": "욕실",
            "issue_type": issue_type,
            "updated_at": "2026-11-06T10:00:00+09:00",
        }
    )

    assert comment.issue_type == issue_type


def test_symptom_changed_request_fixture_uses_current_taxonomy():
    payload = json.loads(
        (Path(__file__).parents[2] / "complaint_symptom_changed_payload.json").read_text()
    )

    request = ConverseRequest.model_validate(payload)

    assert request.complaint_draft.issue_type == "plumbing"


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


def test_web_search_args_accepts_only_a_nonempty_query():
    assert WebSearchArgs.model_validate({"query": "  오늘 날씨  "}).query == "오늘 날씨"
    for payload in ({"query": "  "}, {"query": "시설", "building_id": 2}):
        with pytest.raises(ValidationError):
            WebSearchArgs.model_validate(payload)


@pytest.mark.parametrize(
    ("source_type", "url", "valid"),
    [
        ("web", "https://example.com/a", True),
        ("web", None, False),
        ("building_document", "https://example.com/a", False),
    ],
)
def test_web_search_result_requires_public_web_citations(source_type, url, valid):
    payload = {
        "status": "ok",
        "evidence": [
            {
                "text": "근거 내용",
                "citation": {
                    "source_type": source_type,
                    "source_id": None,
                    "title": "근거 제목",
                    "snippet": "근거 내용",
                    "url": url,
                },
            }
        ],
    }
    if valid:
        assert WebSearchResult.model_validate(payload).evidence
    else:
        with pytest.raises(ValidationError):
            WebSearchResult.model_validate(payload)
