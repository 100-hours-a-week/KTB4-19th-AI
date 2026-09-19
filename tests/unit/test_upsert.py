from datetime import UTC, datetime

import pytest
from qdrant_client import QdrantClient

from zipsai.contracts.indexing import IndexingJobRequest, SourceType
from zipsai.errors import EmptyDocumentError
from zipsai.indexing.embed import EmbeddedChunk
from zipsai.indexing.upsert import upsert_document
from zipsai.integrations.qdrant import (
    DENSE_VECTOR,
    SPARSE_VECTOR,
    ensure_collection,
    to_sparse_vector,
)
from zipsai.settings import EMBEDDING_DIM, QDRANT_COLLECTION


@pytest.fixture
def client() -> QdrantClient:
    qdrant = QdrantClient(":memory:")
    ensure_collection(qdrant)
    return qdrant


def request(building_id: int = 101, doc_id: str = "doc-1") -> IndexingJobRequest:
    return IndexingJobRequest(
        building_id=building_id,
        doc_id=doc_id,
        source_type=SourceType.RULE,
        title="관리규약",
        published_at=datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC),
        file_key="documents/doc.pdf",
        trace_id="trace-1",
    )


def chunk(text: str, value: float = 0.5) -> EmbeddedChunk:
    return EmbeddedChunk(
        text=text,
        page=2,
        section="제1조",
        dense=[value] * EMBEDDING_DIM,
        sparse={"7": value},
    )


def points(client: QdrantClient):
    return client.scroll(
        collection_name=QDRANT_COLLECTION,
        limit=100,
        with_payload=True,
        with_vectors=True,
    )[0]


def test_round_trip_stores_payload_and_named_dense_sparse_vectors(
    client: QdrantClient,
) -> None:
    result = upsert_document(client, request(), [chunk("새 내용")], masked=True)

    assert result == 1
    point = points(client)[0]
    assert point.payload == {
        "building_id": "101",
        "doc_id": "doc-1",
        "source_type": "rule",
        "title": "관리규약",
        "published_at": "2026-01-02T03:04:05+00:00",
        "masked": True,
        "page": 2,
        "section": "제1조",
        "text": "새 내용",
    }
    assert len(point.vector[DENSE_VECTOR]) == EMBEDDING_DIM
    assert point.vector[SPARSE_VECTOR].indices == [7]
    assert point.vector[SPARSE_VECTOR].values == [0.5]


def test_same_doc_id_isolated_by_building_id(client: QdrantClient) -> None:
    upsert_document(client, request(1), [chunk("건물 1 이전")])
    upsert_document(client, request(2), [chunk("건물 2 유지")])

    upsert_document(client, request(1), [chunk("건물 1 교체")])

    assert sorted(
        (point.payload["building_id"], point.payload["text"])
        for point in points(client)
    ) == [("1", "건물 1 교체"), ("2", "건물 2 유지")]


def test_replacing_document_removes_old_chunks(client: QdrantClient) -> None:
    job = request()
    upsert_document(client, job, [chunk("old-1"), chunk("old-2"), chunk("old-3")])

    assert upsert_document(client, job, [chunk("new-1"), chunk("new-2")]) == 2
    assert len(points(client)) == 2
    assert {point.payload["text"] for point in points(client)} == {"new-1", "new-2"}


def test_empty_upsert_keeps_the_existing_document(client: QdrantClient) -> None:
    job = request()

    with pytest.raises(EmptyDocumentError):
        upsert_document(client, job, [])
    assert len(points(client)) == 0

    assert upsert_document(client, job, [chunk("one"), chunk("two")]) == 2

    # 넣을 게 없다고 기존 문서를 지우면 검색에서 문서가 조용히 사라진다.
    with pytest.raises(EmptyDocumentError):
        upsert_document(client, job, [])
    assert len(points(client)) == 2


def test_unconvertible_sparse_key_keeps_the_existing_document(
    client: QdrantClient,
) -> None:
    job = request()
    assert upsert_document(client, job, [chunk("one"), chunk("two")]) == 2

    broken = chunk("three")
    broken = type(broken)(
        text=broken.text,
        page=broken.page,
        section=broken.section,
        dense=broken.dense,
        sparse={"not-a-token-id": 0.5},
    )

    with pytest.raises(ValueError):
        upsert_document(client, job, [broken])
    assert len(points(client)) == 2


def test_to_sparse_vector_preserves_index_value_alignment() -> None:
    sparse = to_sparse_vector({"12": 0.2, "4": 0.8})

    assert sparse.indices == [12, 4]
    assert sparse.values == [0.2, 0.8]
    assert to_sparse_vector({}).indices == []
    assert to_sparse_vector({}).values == []


def test_ensure_collection_is_idempotent() -> None:
    client = QdrantClient(":memory:")

    ensure_collection(client)
    ensure_collection(client)

    assert client.collection_exists(collection_name=QDRANT_COLLECTION)


def test_ensure_collection_adds_indexes_to_a_collection_made_without_them() -> None:
    from qdrant_client import models

    from zipsai.integrations.qdrant import DENSE_VECTOR, SPARSE_VECTOR

    client = QdrantClient(":memory:")
    # doc_id 인덱스가 없던 시절의 컬렉션을 흉내 낸다.
    client.create_collection(
        collection_name="legacy",
        vectors_config={
            DENSE_VECTOR: models.VectorParams(
                size=1024, distance=models.Distance.COSINE
            )
        },
        sparse_vectors_config={SPARSE_VECTOR: models.SparseVectorParams()},
    )

    ensure_collection(client, "legacy")
    ensure_collection(client, "legacy")

    # 교체 필터가 쓰는 두 필드로 실제 삭제가 거부되지 않아야 한다.
    client.delete(
        collection_name="legacy",
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="building_id", match=models.MatchValue(value="101")
                    ),
                    models.FieldCondition(
                        key="doc_id", match=models.MatchValue(value="notice-001")
                    ),
                ]
            )
        ),
        wait=True,
    )
