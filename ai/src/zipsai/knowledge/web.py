from urllib.parse import urlsplit

from zipsai.contracts.converse import Citation, ConverseRequest, Route
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
_PRIVATE_QUERY = "개인정보나 민원 상세는 공개 Web 검색에 보낼 수 없습니다."


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
    args: WebSearchArgs,
    request: ConverseRequest,
    *,
    private_context: tuple[str, ...] = (),
) -> WebSearchResult | ToolFailure:
    """공개 Web 근거를 검색하되 요청의 식별 정보와 민원 세부 내용은 보내지 않는다."""
    if _contains_private_details(args.query, request, private_context):
        return ToolFailure(code="invalid_arguments", message=_PRIVATE_QUERY)

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


def _contains_private_details(
    query: str, request: ConverseRequest, private_context: tuple[str, ...]
) -> bool:
    normalized_query = " ".join(query.casefold().split())
    identifiers = (request.resident_id, request.room_no)
    if any(value.casefold() in normalized_query for value in identifiers if value):
        return True

    sensitive_values: list[str] = list(private_context)
    exact_messages: list[str] = []
    draft = request.complaint_draft
    if draft:
        sensitive_values.extend((draft.location or "", draft.symptom or ""))
        if draft.occurred_at:
            sensitive_values.append(draft.occurred_at.isoformat())

    active_complaint = (
        request.current_route is Route.COMPLAINT
        or request.current_complaint_state is not None
        or draft is not None
    )
    if active_complaint:
        exact_messages.append(request.message.text or "")
        for turn in request.conversation_history:
            if turn.role != "user":
                continue
            exact_messages.append(turn.text or "")
            for image in turn.images or ():
                sensitive_values.extend((image.summary or "", image.ocr_text or ""))

    for item in private_context:
        if item and (item == request.message.text or any(item == turn.text for turn in request.conversation_history)):
            exact_messages.append(item)

    # shortcut: literal matching catches copied details, not paraphrases; use vetted DLP if policy requires semantic detection.
    if any(
        (value := " ".join(item.casefold().split()))
        and len(value) >= 2
        and value in normalized_query
        for item in exact_messages
    ):
        return True

    return any(
        (value := " ".join(item.casefold().split()))
        and len(value) >= 2
        and value in normalized_query
        for item in sensitive_values
    )
