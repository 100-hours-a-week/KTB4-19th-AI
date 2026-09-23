from uuid import uuid4

from qdrant_client import QdrantClient, models

from zipsai.integrations.embedding_client import DEFAULT_TIMEOUT_SECONDS
from zipsai.integrations.qdrant import (
    DENSE_VECTOR,
    SPARSE_VECTOR,
    create_client,
    ensure_collection,
    to_sparse_vector,
)
from zipsai.knowledge.retrieve import (
    QUERY_TIMEOUT_SECONDS,
    TOP_K,
    encode_question,
    query_encoder,
    search_chunks,
)
from zipsai.settings import EMBEDDING_DIM

TEST_COLLECTION = "test_documents"


class FakeEncoder:
    def __init__(
        self,
        dense: list[list[float]] | None = None,
        sparse: list[dict[str, float]] | None = None,
    ) -> None:
        self.calls: list[list[str]] = []
        self._dense = dense
        self._sparse = sparse

    def encode(
        self, texts: list[str]
    ) -> tuple[list[list[float]], list[dict[str, float]]]:
        self.calls.append(texts)
        dense = self._dense or [[0.0] * EMBEDDING_DIM for _ in texts]
        sparse = self._sparse or [{} for _ in texts]
        return dense, sparse


def test_encode_question_sends_one_text_and_unwraps_both_vectors() -> None:
    encoder = FakeEncoder(dense=[[0.5] * EMBEDDING_DIM], sparse=[{"7": 0.25}])

    dense, sparse = encode_question("세탁실은 몇 시까지 쓸 수 있나요?", encoder=encoder)

    assert encoder.calls == [["세탁실은 몇 시까지 쓸 수 있나요?"]]
    assert dense == [0.5] * EMBEDDING_DIM
    assert sparse == {"7": 0.25}


def test_query_encoder_waits_far_less_than_the_indexing_batch() -> None:
    # 배치용 기본값을 그대로 물려받으면 백엔드가 먼저 요청을 끊는다.
    assert QUERY_TIMEOUT_SECONDS < DEFAULT_TIMEOUT_SECONDS
    assert query_encoder()._client.timeout.read == QUERY_TIMEOUT_SECONDS


class CountingClient:
    """query_points 호출 횟수를 센다. 게이트가 하이브리드를 건너뛰는지 보려면 필요하다."""

    def __init__(self, inner: QdrantClient) -> None:
        self._inner = inner
        self.query_calls = 0

    def query_points(self, **kwargs: object) -> models.QueryResponse:
        self.query_calls += 1
        return self._inner.query_points(**kwargs)


def _dense(*head: float) -> list[float]:
    vector = [0.0] * EMBEDDING_DIM
    for index, value in enumerate(head):
        vector[index] = value
    return vector


def _point(building_id: int, doc_id: str, dense: list[float]) -> models.PointStruct:
    return models.PointStruct(
        id=str(uuid4()),
        vector={
            DENSE_VECTOR: dense,
            SPARSE_VECTOR: to_sparse_vector({"7": 0.5}),
        },
        # 색인이 building_id를 문자열로 넣는다. 여기서 int로 넣으면 검색이 0건이 된다.
        payload={
            "building_id": str(building_id),
            "doc_id": doc_id,
            "title": f"{doc_id} 안내",
            "text": f"{doc_id} 본문",
        },
    )


def _client_with(*points: models.PointStruct) -> QdrantClient:
    client = create_client(":memory:")
    ensure_collection(client, TEST_COLLECTION)
    client.upsert(collection_name=TEST_COLLECTION, points=list(points), wait=True)
    return client


def test_search_returns_chunks_of_the_asked_building_only() -> None:
    # 같은 내용이라도 다른 건물 문서면 새어 나가면 안 된다.
    client = _client_with(
        _point(1, "b001-parking", _dense(1.0)),
        _point(2, "b002-parking", _dense(1.0)),
    )

    hits = search_chunks(
        (_dense(1.0), {"7": 0.5}), 1, client=client, collection=TEST_COLLECTION
    )

    assert [hit.payload["doc_id"] for hit in hits] == ["b001-parking"]


def test_search_skips_hybrid_query_when_gate_finds_nothing() -> None:
    # 근거가 없으면 하이브리드도 LLM도 돌리지 않는다. 실패가 성공보다 싸야 한다.
    client = CountingClient(_client_with(_point(1, "b001-parking", _dense(1.0))))

    hits = search_chunks(
        (_dense(0.0, 1.0), {"7": 0.5}), 1, client=client, collection=TEST_COLLECTION
    )

    assert hits == []
    assert client.query_calls == 1


def test_search_runs_hybrid_query_when_gate_passes() -> None:
    client = CountingClient(_client_with(_point(1, "b001-parking", _dense(1.0))))

    hits = search_chunks(
        (_dense(1.0), {"7": 0.5}), 1, client=client, collection=TEST_COLLECTION
    )

    assert [hit.payload["doc_id"] for hit in hits] == ["b001-parking"]
    assert client.query_calls == 2


def test_search_returns_at_most_top_k_chunks() -> None:
    client = _client_with(
        *(_point(1, f"b001-doc{index}", _dense(1.0)) for index in range(TOP_K + 2))
    )

    hits = search_chunks(
        (_dense(1.0), {"7": 0.5}), 1, client=client, collection=TEST_COLLECTION
    )

    assert len(hits) == TOP_K
