import json
import re
from pathlib import Path

import httpx
import pytest

from zipsai.errors import (
    ChunkTooLongError,
    EmbeddingError,
    EmbeddingRequestInvalidError,
)
from zipsai.indexing.chunk import Chunk
from zipsai.indexing.embed import embed_chunks
from zipsai.integrations.embedding_client import (
    ATTEMPTS,
    BATCH_SIZE,
    MAX_TEXT_CHARS,
    HttpEncoder,
)
from zipsai.knowledge import retrieve


def encoder_returning(handler) -> HttpEncoder:
    # 테스트가 재시도 대기에 묶이지 않게 지연을 없앤다.
    return HttpEncoder(
        base_url="http://embedding:8000",
        transport=httpx.MockTransport(handler),
        retry_delay=0,
    )


def test_encode_posts_texts_and_unpacks_dense_and_sparse() -> None:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append({"url": str(request.url), "body": request.read().decode()})
        return httpx.Response(
            200,
            json={"dense": [[0.1, 0.2]], "sparse": [{"7": 0.5}]},
        )

    dense, sparse = encoder_returning(handler).encode(["첫 청크"], "trace-001")

    assert dense == [[0.1, 0.2]]
    assert sparse == [{"7": 0.5}]
    assert seen[0]["url"] == "http://embedding:8000/embed"
    assert "첫 청크" in seen[0]["body"]


def test_encode_sends_lock_wait_seconds_when_configured() -> None:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.read()))
        return httpx.Response(200, json={"dense": [[0.1]], "sparse": [{}]})

    encoder = HttpEncoder(
        base_url="http://embedding:8000",
        transport=httpx.MockTransport(handler),
        retry_delay=0,
        lock_wait_seconds=3.5,
    )
    encoder.encode(["텍스트"], "trace-001")

    assert seen[0]["max_wait_seconds"] == 3.5


def test_encode_sends_null_lock_wait_seconds_by_default() -> None:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.read()))
        return httpx.Response(200, json={"dense": [[0.1]], "sparse": [{}]})

    encoder_returning(handler).encode(["텍스트"], "trace-001")

    assert seen[0]["max_wait_seconds"] is None


def test_encode_sends_index_priority_by_default() -> None:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.read()))
        return httpx.Response(200, json={"dense": [[0.1]], "sparse": [{}]})

    encoder_returning(handler).encode(["텍스트"], "trace-001")

    assert seen[0]["priority"] == "index"


def test_query_encoder_sends_query_priority_and_wait_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.read()))
        return httpx.Response(200, json={"dense": [[0.1]], "sparse": [{}]})

    real = HttpEncoder
    monkeypatch.setattr(
        retrieve,
        "HttpEncoder",
        lambda **kwargs: real(
            base_url="http://embedding:8000",
            transport=httpx.MockTransport(handler),
            **kwargs,
        ),
    )
    retrieve.query_encoder.cache_clear()
    try:
        retrieve.query_encoder().encode(["질문"], "trace-001")
    finally:
        retrieve.query_encoder.cache_clear()

    assert seen[0]["priority"] == "query"
    assert seen[0]["max_wait_seconds"] == 3.5


def test_client_error_does_not_keep_the_response_body() -> None:
    sentence = "화장실 천장에서 물이 새요"

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(422, text=sentence)

    with pytest.raises(EmbeddingRequestInvalidError) as caught:
        encoder_returning(handler).encode(["텍스트"], "trace-001")

    assert sentence not in str(caught.value)
    assert str(caught.value) == "status=422"
    assert caught.value.__cause__ is None


def test_encode_raises_typed_error_on_server_error() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"detail": "Model is still loading"})

    with pytest.raises(EmbeddingError):
        encoder_returning(handler).encode(["text"], "trace-001")


def test_encode_raises_typed_error_on_transport_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with pytest.raises(EmbeddingError):
        encoder_returning(handler).encode(["text"], "trace-001")


@pytest.mark.parametrize("body", [{}, {"dense": [[0.1]]}, []])
def test_encode_raises_typed_error_on_malformed_response(body: object) -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body)

    with pytest.raises(EmbeddingError):
        encoder_returning(handler).encode(["text"], "trace-001")


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

    total = BATCH_SIZE + 5
    texts = [f"청크 {index}" for index in range(total)]

    dense, sparse = encoder_returning(handler).encode(texts, "trace-001")

    remainder = total % BATCH_SIZE
    expected_batches = [BATCH_SIZE] * (total // BATCH_SIZE) + (
        [remainder] if remainder else []
    )
    assert batch_sizes == expected_batches
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
        encoder_returning(handler).encode(texts, "trace-001")

    # 첫 배치에서 걸려야 한다. 둘째 배치까지 갔다면 배치별 검증이 아니다.
    assert seen == [BATCH_SIZE]


def test_encode_retries_once_and_succeeds_when_the_server_was_restarting() -> None:
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        if len(attempts) == 1:
            return httpx.Response(503, json={"detail": "Model is still loading"})
        return httpx.Response(200, json={"dense": [[0.1]], "sparse": [{}]})

    dense, _ = encoder_returning(handler).encode(["청크"], "trace-001")

    assert dense == [[0.1]]
    assert len(attempts) == ATTEMPTS


def test_encode_gives_up_after_the_retry() -> None:
    attempts: list[int] = []

    def handler(_: httpx.Request) -> httpx.Response:
        attempts.append(1)
        return httpx.Response(503, json={"detail": "Model is still loading"})

    with pytest.raises(EmbeddingError) as caught:
        encoder_returning(handler).encode(["청크"], "trace-001")

    assert type(caught.value) is EmbeddingError
    assert len(attempts) == ATTEMPTS


@pytest.mark.parametrize("status_code", [400, 422])
def test_encode_does_not_retry_a_rejected_request(status_code: int) -> None:
    # 4xx는 같은 요청을 다시 보내도 같은 거절이 온다. 408·429는 여기 없다.
    attempts: list[int] = []
    sentence = "texts must hold at most 32 items"

    def handler(_: httpx.Request) -> httpx.Response:
        attempts.append(1)
        return httpx.Response(status_code, json={"detail": sentence})

    with pytest.raises(EmbeddingRequestInvalidError) as caught:
        encoder_returning(handler).encode(["청크"], "trace-001")

    assert str(caught.value) == f"status={status_code}"
    assert sentence not in str(caught.value)
    assert caught.value.__cause__ is None
    assert len(attempts) == 1


@pytest.mark.parametrize("status_code", [408, 429])
def test_encode_retries_a_transient_client_status(status_code: int) -> None:
    attempts: list[int] = []

    def handler(_: httpx.Request) -> httpx.Response:
        attempts.append(1)
        return httpx.Response(status_code, text="slow down")

    with pytest.raises(EmbeddingError) as caught:
        encoder_returning(handler).encode(["청크"], "trace-001")

    assert type(caught.value) is EmbeddingError
    assert len(attempts) == ATTEMPTS


def test_encode_retries_a_connection_failure() -> None:
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        raise httpx.ConnectError("connection refused", request=request)

    with pytest.raises(EmbeddingError) as caught:
        encoder_returning(handler).encode(["청크"], "trace-001")

    assert type(caught.value) is EmbeddingError
    assert len(attempts) == ATTEMPTS


def test_overlong_chunk_is_rejected_before_http() -> None:
    calls = 0
    long_text = "가" * (MAX_TEXT_CHARS + 1)

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"dense": [[0.1]], "sparse": [{}]})

    chunks = [Chunk("짧은 청크", 1, None), Chunk(long_text, 2, None)]
    with pytest.raises(ChunkTooLongError) as caught:
        embed_chunks(chunks, encoder_returning(handler), "trace-001")

    assert calls == 0
    assert caught.value.index == 1
    assert caught.value.length == len(long_text)
    assert caught.value.limit == MAX_TEXT_CHARS
    assert long_text not in str(caught.value)


def test_text_limit_matches_embedding_server() -> None:
    source = (
        Path(__file__).resolve().parents[3] / "embedding/src/embedding/api/embed.py"
    )
    match = re.search(
        r"^MAX_TEXT_CHARS\s*=\s*(\d+)\s*$",
        source.read_text(),
        re.MULTILINE,
    )

    assert match is not None
    assert MAX_TEXT_CHARS == int(match.group(1))
