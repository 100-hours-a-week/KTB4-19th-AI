import json

import httpx
import pytest

from zipsai.errors import EmbeddingError
from zipsai.integrations.embedding_client import BATCH_SIZE, HttpEncoder


def encoder_returning(handler) -> HttpEncoder:
    return HttpEncoder(
        base_url="http://embedding:8000",
        transport=httpx.MockTransport(handler),
    )


def test_encode_posts_texts_and_unpacks_dense_and_sparse() -> None:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append({"url": str(request.url), "body": request.read().decode()})
        return httpx.Response(
            200,
            json={"dense": [[0.1, 0.2]], "sparse": [{"7": 0.5}]},
        )

    dense, sparse = encoder_returning(handler).encode(["첫 청크"])

    assert dense == [[0.1, 0.2]]
    assert sparse == [{"7": 0.5}]
    assert seen[0]["url"] == "http://embedding:8000/embed"
    assert "첫 청크" in seen[0]["body"]


def test_encode_raises_typed_error_on_server_error() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"detail": "Model is still loading"})

    with pytest.raises(EmbeddingError):
        encoder_returning(handler).encode(["text"])


def test_encode_raises_typed_error_on_transport_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with pytest.raises(EmbeddingError):
        encoder_returning(handler).encode(["text"])


@pytest.mark.parametrize("body", [{}, {"dense": [[0.1]]}, []])
def test_encode_raises_typed_error_on_malformed_response(body: object) -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body)

    with pytest.raises(EmbeddingError):
        encoder_returning(handler).encode(["text"])


def test_encode_splits_texts_into_server_sized_batches() -> None:
    batch_sizes: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        texts = json.loads(request.read())["texts"]
        batch_sizes.append(len(texts))
        return httpx.Response(
            200,
            json={
                "dense": [[float(len(text))] for text in texts],
                "sparse": [{} for _ in texts],
            },
        )

    texts = [f"청크 {index}" for index in range(BATCH_SIZE + 5)]

    dense, sparse = encoder_returning(handler).encode(texts)

    assert batch_sizes == [BATCH_SIZE, 5]
    assert dense == [[float(len(text))] for text in texts]
    assert len(sparse) == len(texts)


def test_encode_rejects_a_batch_whose_counts_do_not_match_the_request() -> None:
    # 첫 배치는 하나 모자라고 둘째 배치는 하나 남는다. 총합은 맞으므로
    # 배치마다 세지 않으면 청크와 벡터가 한 칸씩 밀린 채 통과한다.
    seen: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        texts = json.loads(request.read())["texts"]
        seen.append(len(texts))
        shift = -1 if len(seen) == 1 else 1
        return httpx.Response(
            200,
            json={
                "dense": [[0.1]] * (len(texts) + shift),
                "sparse": [{}] * len(texts),
            },
        )

    texts = [f"청크 {index}" for index in range(BATCH_SIZE + 5)]

    with pytest.raises(EmbeddingError):
        encoder_returning(handler).encode(texts)

    # 첫 배치에서 걸려야 한다. 둘째 배치까지 갔다면 배치별 검증이 아니다.
    assert seen == [BATCH_SIZE]
