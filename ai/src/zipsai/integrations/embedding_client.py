import time

import httpx

from zipsai.errors import EmbeddingError
from zipsai.settings import EMBEDDING_API_URL

# CPU 임베딩은 배치가 크면 수십 초가 걸린다. 실측 후 조정한다.
DEFAULT_TIMEOUT_SECONDS = 120.0
# embedding 서비스의 MAX_BATCH_SIZE와 같아야 한다. 별도 패키지라 값을 공유할
# 수 없으므로, 올릴 때는 embedding/src/embedding/api/embed.py도 같이 올린다.
BATCH_SIZE = 32
# 일시 장애만 한 번 더 시도한다. embedding 컨테이너 재시작이 주 대상이라
# 5초면 대개 되살아난다. 모자라면 다음 배치 실행이 다시 집어간다.
ATTEMPTS = 2
RETRY_DELAY_SECONDS = 5.0

EncodeOutput = tuple[list[list[float]], list[dict[str, float]]]


def _is_transient(exc: httpx.HTTPError) -> bool:
    # 4xx는 같은 요청을 다시 보내도 같은 답이 온다. 연결 실패와 5xx만 다시 시도한다.
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= 500
    return True


class HttpEncoder:
    def __init__(
        self,
        base_url: str = EMBEDDING_API_URL,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        transport: httpx.BaseTransport | None = None,
        retry_delay: float = RETRY_DELAY_SECONDS,
        attempts: int = ATTEMPTS,
    ) -> None:
        self._retry_delay = retry_delay
        self._attempts = attempts
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            transport=transport,
        )

    def encode(self, texts: list[str]) -> EncodeOutput:
        dense: list[list[float]] = []
        sparse: list[dict[str, float]] = []
        for start in range(0, len(texts), BATCH_SIZE):
            batch_dense, batch_sparse = self._encode_batch(
                texts[start : start + BATCH_SIZE]
            )
            dense.extend(batch_dense)
            sparse.extend(batch_sparse)
        return dense, sparse

    def _post_with_retry(self, texts: list[str]) -> dict:
        last: httpx.HTTPError | None = None
        for attempt in range(self._attempts):
            try:
                response = self._client.post("/embed", json={"texts": texts})
                response.raise_for_status()
                return response.json()
            except httpx.HTTPError as exc:
                if not _is_transient(exc):
                    detail = ""
                    if isinstance(exc, httpx.HTTPStatusError):
                        detail = f" {exc.response.text[:500]}"
                    raise EmbeddingError(
                        f"Embedding request failed: {exc}{detail}"
                    ) from exc
                last = exc
                if attempt + 1 < self._attempts:
                    time.sleep(self._retry_delay)
        raise EmbeddingError(
            f"Embedding request failed after {self._attempts} attempts: {last}"
        ) from last

    def _encode_batch(self, texts: list[str]) -> EncodeOutput:
        body = self._post_with_retry(texts)

        try:
            dense, sparse = body["dense"], body["sparse"]
        except (KeyError, TypeError) as exc:
            raise EmbeddingError(f"Malformed embedding response: {body!r}") from exc

        # 총합만 맞고 배치별로 어긋나면 청크와 벡터가 조용히 뒤섞인다.
        # ai-api와 embedding은 이미지 태그가 따로라 계약이 어긋날 수 있다.
        if len(dense) != len(texts) or len(sparse) != len(texts):
            raise EmbeddingError(
                f"Embedding response returned {len(dense)} dense and "
                f"{len(sparse)} sparse vectors for {len(texts)} texts"
            )
        return dense, sparse
