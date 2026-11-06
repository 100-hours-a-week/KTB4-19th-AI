import pytest

from zipsai.contracts.converse import ConverseRequest
from zipsai.errors import (
    WebSearchRateLimitedError,
    WebSearchTimeoutError,
    WebSearchUnavailableError,
    WebSearchUpstreamError,
)
from zipsai.orchestration.tool_contracts import (
    ToolFailure,
    WebSearchArgs,
    WebSearchResult,
)


def _request() -> ConverseRequest:
    return ConverseRequest.model_validate(
        {
            "building_id": 1,
            "room_no": "301",
            "resident_id": "linda",
            "conversation_id": "conv-001",
            "turn_id": "turn-001",
            "trace_id": "trace-001",
            "current_route": None,
            "current_complaint_state": None,
            "message": {"message_id": "msg-001", "text": "오늘 날씨 어때요?"},
            "conversation_history": [],
            "complaint_draft": None,
        }
    )


def test_web_evidence_drops_entries_without_url_or_content():
    from zipsai.knowledge.web import web_evidence

    raw = [
        {"title": "A", "url": "https://example.com/a", "content": "내용"},
        {"title": "B", "url": None, "content": "내용 없는 url"},
        {"title": "C", "url": "https://example.com/c", "content": ""},
        {"title": "D", "url": "https://example.com/a", "content": "중복 url"},
        {"title": "E", "url": "javascript:alert(1)", "content": "위험한 링크"},
        {"title": "F", "url": 7, "content": "잘못된 URL 타입"},
        {"title": "G", "url": "https://example.com/g", "content": 12},
    ]
    evidence = web_evidence(raw)
    assert [item.citation.url for item in evidence] == ["https://example.com/a"]
    assert evidence[0].citation.source_type == "web"


def test_web_evidence_truncates_excerpts_and_uses_url_when_title_is_missing():
    from zipsai.knowledge.web import SNIPPET_LENGTH, web_evidence

    evidence = web_evidence(
        [{"url": "https://example.com/a", "content": "가" * (SNIPPET_LENGTH + 20)}]
    )

    assert evidence[0].citation.title == "https://example.com/a"
    assert len(evidence[0].citation.snippet) == SNIPPET_LENGTH


def test_search_public_web_returns_no_results_when_tavily_has_nothing(monkeypatch):
    import zipsai.knowledge.web as web_module

    monkeypatch.setattr(web_module, "search", lambda _query: [])

    result = web_module.search_public_web(WebSearchArgs(query="아주 희귀한 질문"), _request())

    assert isinstance(result, WebSearchResult)
    assert result.status == "no_results"
    assert result.evidence == []


def test_search_public_web_returns_ok_with_citations(monkeypatch):
    import zipsai.knowledge.web as web_module

    monkeypatch.setattr(
        web_module,
        "search",
        lambda _query: [
            {
                "title": "일반 결로 예방법",
                "url": "https://example.com/mold",
                "content": "환기를 자주 하세요.",
            }
        ],
    )

    result = web_module.search_public_web(WebSearchArgs(query="결로 예방법"), _request())

    assert isinstance(result, WebSearchResult)
    assert result.status == "ok"
    assert result.evidence[0].citation.source_type == "web"
    assert result.evidence[0].citation.url == "https://example.com/mold"


@pytest.mark.parametrize(
    ("raised", "expected_code"),
    [
        (WebSearchTimeoutError("t"), "timeout"),
        (WebSearchRateLimitedError("r"), "rate_limited"),
        (WebSearchUnavailableError("u"), "unavailable"),
        (WebSearchUpstreamError("e"), "upstream_error"),
    ],
)
def test_search_public_web_maps_client_errors_to_tool_failure(monkeypatch, raised, expected_code):
    import zipsai.knowledge.web as web_module

    def _raise(_query):
        raise raised

    monkeypatch.setattr(web_module, "search", _raise)

    result = web_module.search_public_web(WebSearchArgs(query="결로 예방법"), _request())

    assert isinstance(result, ToolFailure)
    assert result.code == expected_code
