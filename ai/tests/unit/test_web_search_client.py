import json

import httpx
import pytest

from zipsai.errors import (
    WebSearchRateLimitedError,
    WebSearchTimeoutError,
    WebSearchUnavailableError,
    WebSearchUpstreamError,
)
from zipsai.integrations import web_search


def test_search_sends_bearer_authenticated_bounded_request(monkeypatch):
    monkeypatch.setattr(web_search, "TAVILY_API_KEY", "tvly-test-key")
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(
            200,
            json={"results": [{"title": "안내", "url": "https://example.com", "content": "내용"}]},
        )

    results = web_search.search("분리수거 일정", transport=httpx.MockTransport(handler))

    assert results[0]["title"] == "안내"
    assert str(seen[0].url) == "https://api.tavily.com/search"
    assert seen[0].headers["Authorization"] == "Bearer tvly-test-key"
    assert json.loads(seen[0].content) == {
        "query": "분리수거 일정",
        "search_depth": "basic",
        "max_results": 5,
        "chunks_per_source": 2,
    }


def test_search_fails_closed_without_api_key(monkeypatch):
    monkeypatch.setattr(web_search, "TAVILY_API_KEY", None)

    with pytest.raises(WebSearchUnavailableError):
        web_search.search("질문")


def test_search_maps_rate_limit_and_retry_after(monkeypatch):
    monkeypatch.setattr(web_search, "TAVILY_API_KEY", "key")
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(429, headers={"Retry-After": "3"})
    )

    with pytest.raises(WebSearchRateLimitedError) as exc:
        web_search.search("질문", transport=transport)

    assert exc.value.retry_after_seconds == 3


@pytest.mark.parametrize("status", [432, 433, 434])
def test_search_maps_quota_limits(status, monkeypatch):
    monkeypatch.setattr(web_search, "TAVILY_API_KEY", "key")

    with pytest.raises(WebSearchRateLimitedError):
        web_search.search(
            "질문", transport=httpx.MockTransport(lambda _request: httpx.Response(status))
        )


def test_search_maps_timeout(monkeypatch):
    monkeypatch.setattr(web_search, "TAVILY_API_KEY", "key")

    def timeout(_request):
        raise httpx.ReadTimeout("late")

    with pytest.raises(WebSearchTimeoutError):
        web_search.search("질문", transport=httpx.MockTransport(timeout))


@pytest.mark.parametrize(
    ("response", "exception"),
    [
        (httpx.Response(401), WebSearchUnavailableError),
        (httpx.Response(503), WebSearchUnavailableError),
        (httpx.Response(400), WebSearchUpstreamError),
        (httpx.Response(200, json={"bad": "shape"}), WebSearchUpstreamError),
        (httpx.Response(200, json={"results": {}}), WebSearchUpstreamError),
    ],
)
def test_search_maps_provider_errors(response, exception, monkeypatch):
    monkeypatch.setattr(web_search, "TAVILY_API_KEY", "key")

    with pytest.raises(exception):
        web_search.search(
            "질문", transport=httpx.MockTransport(lambda _request: response)
        )
