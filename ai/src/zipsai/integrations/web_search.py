import httpx

from zipsai.errors import (
    WebSearchRateLimitedError,
    WebSearchTimeoutError,
    WebSearchUnavailableError,
    WebSearchUpstreamError,
)
from zipsai.settings import TAVILY_API_KEY, TAVILY_TIMEOUT_SECONDS

_BASE_URL = "https://api.tavily.com"
_QUOTA_LIMIT_STATUSES = {432, 433, 434}


def search(
    query: str,
    *,
    timeout: float = TAVILY_TIMEOUT_SECONDS,
    transport: httpx.BaseTransport | None = None,
) -> list[dict[str, object]]:
    """Call Tavily Search and return its excerpts without fetching result URLs."""
    if not TAVILY_API_KEY:
        raise WebSearchUnavailableError("TAVILY_API_KEY is not configured")

    with httpx.Client(base_url=_BASE_URL, timeout=timeout, transport=transport) as client:
        try:
            response = client.post(
                "/search",
                headers={"Authorization": f"Bearer {TAVILY_API_KEY}"},
                json={
                    "query": query,
                    "search_depth": "basic",
                    "max_results": 5,
                    "chunks_per_source": 2,
                },
            )
        except httpx.TimeoutException as exc:
            raise WebSearchTimeoutError("Tavily request timed out") from exc
        except httpx.HTTPError as exc:
            raise WebSearchUnavailableError("Tavily request failed") from exc

    if response.status_code == 429:
        retry_after = response.headers.get("Retry-After")
        try:
            retry_after_seconds = int(retry_after) if retry_after is not None else None
        except ValueError:
            retry_after_seconds = None
        raise WebSearchRateLimitedError("Tavily rate limit exceeded", retry_after_seconds)
    if response.status_code in _QUOTA_LIMIT_STATUSES:
        raise WebSearchRateLimitedError("Tavily quota limit exceeded")
    if response.status_code in {401, 403} or response.status_code >= 500:
        raise WebSearchUnavailableError(f"Tavily returned status {response.status_code}")
    if response.status_code >= 400:
        raise WebSearchUpstreamError(f"Tavily rejected the request: {response.status_code}")

    try:
        results = response.json()["results"]
    except (ValueError, KeyError, TypeError) as exc:
        raise WebSearchUpstreamError("Malformed Tavily response") from exc
    if not isinstance(results, list) or any(not isinstance(item, dict) for item in results):
        raise WebSearchUpstreamError("Tavily response results must be a list of objects")
    return results
