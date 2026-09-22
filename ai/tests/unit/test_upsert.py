import pytest
from qdrant_client import QdrantClient

from zipsai.contracts.indexing import IndexingJobRequest
from zipsai.errors import EmptyDocumentError
from zipsai.indexing.embed import EmbeddedChunk
from zipsai.indexing.upsert import delete_missing_documents, upsert_document
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
        title="관리규약",
        file_key="documents/doc.pdf",
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
        "title": "관리규약",
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


def test_same_building_isolated_by_doc_id(client: QdrantClient) -> None:
    # delete 필터에서 doc_id가 빠지면 문서 하나를 고칠 때 건물 전체가 지워진다.
    upsert_document(client, request(101, "doc-a"), [chunk("문서 A 이전")])
    upsert_document(client, request(101, "doc-b"), [chunk("문서 B 유지")])

    upsert_document(client, request(101, "doc-a"), [chunk("문서 A 교체")])

    assert sorted(
        (point.payload["doc_id"], point.payload["text"]) for point in points(client)
    ) == [("doc-a", "문서 A 교체"), ("doc-b", "문서 B 유지")]


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


def test_reconcile_deletes_documents_missing_from_the_valid_list(
    client: QdrantClient,
) -> None:
    upsert_document(client, request(101, "doc-a"), [chunk("A")])
    upsert_document(client, request(101, "doc-b"), [chunk("B")])
    upsert_document(client, request(101, "doc-c"), [chunk("C")])

    delete_missing_documents(client, 101, ["doc-a", "doc-c"])

    assert sorted(point.payload["doc_id"] for point in points(client)) == [
        "doc-a",
        "doc-c",
    ]


def test_reconcile_leaves_other_buildings_alone(client: QdrantClient) -> None:
    # building_id 조건이 빠지면 한 건물 정리가 전체 색인을 지운다.
    upsert_document(client, request(101, "doc-a"), [chunk("건물 101")])
    upsert_document(client, request(202, "doc-a"), [chunk("건물 202")])

    delete_missing_documents(client, 101, ["doc-z"])

    assert [point.payload["building_id"] for point in points(client)] == ["202"]


def test_reconcile_refuses_an_empty_valid_list(client: QdrantClient) -> None:
    upsert_document(client, request(101, "doc-a"), [chunk("A")])

    with pytest.raises(ValueError):
        delete_missing_documents(client, 101, [])
    assert len(points(client)) == 1
