import httpx

from zipsai.errors import EmbeddingError
from zipsai.settings import EMBEDDING_API_URL

# CPU 임베딩은 배치가 크면 수십 초가 걸린다. 실측 후 조정한다.
DEFAULT_TIMEOUT_SECONDS = 120.0


class HttpEncoder:
    def __init__(
        self,
        base_url: str = EMBEDDING_API_URL,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            transport=transport,
        )

    def encode(
        self, texts: list[str]
    ) -> tuple[list[list[float]], list[dict[str, float]]]:
        try:
            response = self._client.post("/embed", json={"texts": texts})
            response.raise_for_status()
            body = response.json()
        except httpx.HTTPError as exc:
            raise EmbeddingError(f"Embedding request failed: {exc}") from exc

        try:
            return body["dense"], body["sparse"]
        except (KeyError, TypeError) as exc:
            raise EmbeddingError(f"Malformed embedding response: {body!r}") from exc
