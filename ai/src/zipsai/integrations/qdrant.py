from qdrant_client import QdrantClient, models

from zipsai.settings import (
    EMBEDDING_DIM,
    QDRANT_API_KEY,
    QDRANT_COLLECTION,
    QDRANT_URL,
)

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "sparse"


def create_client(url: str | None = None) -> QdrantClient:
    target = url or QDRANT_URL
    if not target:
        raise RuntimeError("QDRANT_URL is not configured")
    # ":memory:"는 서버가 아니라 프로세스 안의 임시 저장소다. 테스트만 이 값을 넘긴다.
    if target == ":memory:":
        return QdrantClient(":memory:")
    # 인증을 켜지 않은 Qdrant면 키가 None이고, 그때는 헤더 없이 붙는다.
    return QdrantClient(url=target, api_key=QDRANT_API_KEY)


def to_sparse_vector(weights: dict[str, float]) -> models.SparseVector:
    return models.SparseVector(
        indices=[int(index) for index in weights],
        values=list(weights.values()),
    )


def ensure_collection(
    client: QdrantClient, collection: str = QDRANT_COLLECTION
) -> None:
    if not client.collection_exists(collection_name=collection):
        _create_collection(client, collection)
    # 인덱스는 컬렉션 생성 여부와 따로 보장한다. doc_id 인덱스가 없던 시절에
    # 만들어진 컬렉션을 그대로 쓰면 교체 필터가 거부된다. 재생성은 멱등이다.
    _ensure_payload_indexes(client, collection)


def _create_collection(client: QdrantClient, collection: str) -> None:
    client.create_collection(
        collection_name=collection,
        vectors_config={
            DENSE_VECTOR: models.VectorParams(
                size=EMBEDDING_DIM,
                distance=models.Distance.COSINE,
            )
        },
        sparse_vectors_config={SPARSE_VECTOR: models.SparseVectorParams()},
    )


def _ensure_payload_indexes(client: QdrantClient, collection: str) -> None:
    client.create_payload_index(
        collection_name=collection,
        field_name="building_id",
        field_schema=models.KeywordIndexParams(
            type=models.KeywordIndexType.KEYWORD,
            is_tenant=True,
        ),
        wait=True,
    )
    # doc_id는 문서 교체 시 delete 필터에 쓰인다.
    # 서버 Qdrant strict mode는 미인덱스 필드 필터를 거부하므로 인덱스를 만든다.
    client.create_payload_index(
        collection_name=collection,
        field_name="doc_id",
        field_schema=models.KeywordIndexParams(
            type=models.KeywordIndexType.KEYWORD,
        ),
        wait=True,
    )
