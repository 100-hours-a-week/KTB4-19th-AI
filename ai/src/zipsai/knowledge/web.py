from urllib.parse import urlsplit

from zipsai.contracts.converse import Citation, ConverseRequest
from zipsai.errors import (
    WebSearchRateLimitedError,
    WebSearchTimeoutError,
    WebSearchUnavailableError,
    WebSearchUpstreamError,
)
from zipsai.integrations.web_search import search
from zipsai.orchestration.tool_contracts import (
    SearchEvidence,
    ToolFailure,
    WebSearchArgs,
    WebSearchResult,
)

SNIPPET_LENGTH = 200
_UNAVAILABLE = "Web 검색을 사용할 수 없습니다."
_TIMEOUT = "Web 검색이 시간 초과됐습니다."
_RATE_LIMITED = "Web 검색 요청이 많아 잠시 후 다시 시도해야 합니다."
_UPSTREAM_ERROR = "Web 검색 결과를 처리하지 못했습니다."


def web_evidence(raw_results: list[dict[str, object]]) -> list[SearchEvidence]:
    """url·content가 없는 결과는 근거로 쓰지 않는다. url 기준으로 중복을 제거한다."""
    result: list[SearchEvidence] = []
    seen: set[str] = set()
    for item in raw_results:
        url = item.get("url")
        content = item.get("content")
        parsed_url = urlsplit(url) if isinstance(url, str) else None
        if (
            parsed_url is None
            or parsed_url.scheme not in {"http", "https"}
            or not parsed_url.netloc
            or not isinstance(content, str)
            or not content.strip()
        ):
            continue
        if url in seen:
            continue
        seen.add(url)
        snippet = content[:SNIPPET_LENGTH]
        result.append(
            SearchEvidence(
                text=snippet,
                citation=Citation(
                    source_type="web",
                    source_id=None,
                    title=item.get("title") if isinstance(item.get("title"), str) else url,
                    snippet=snippet,
                    url=url,
                ),
            )
        )
    return result


def search_public_web(
    args: WebSearchArgs, request: ConverseRequest
) -> WebSearchResult | ToolFailure:
    """공개 Web에서만 근거를 검색한다. request는 호출 패턴을 search_building_docs와
    맞추기 위해 받지만 건물 범위 검증에는 쓰지 않는다 — 공개 정보라 범위가 없다."""
    del request
    try:
        raw_results = search(args.query)
    except WebSearchTimeoutError:
        return ToolFailure(code="timeout", message=_TIMEOUT)
    except WebSearchRateLimitedError:
        return ToolFailure(code="rate_limited", message=_RATE_LIMITED)
    except WebSearchUnavailableError:
        return ToolFailure(code="unavailable", message=_UNAVAILABLE)
    except WebSearchUpstreamError:
        return ToolFailure(code="upstream_error", message=_UPSTREAM_ERROR)

    evidence = web_evidence(raw_results)
    if not evidence:
        return WebSearchResult(status="no_results")
    return WebSearchResult(status="ok", evidence=evidence)
