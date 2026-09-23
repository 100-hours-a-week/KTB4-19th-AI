from functools import lru_cache

from qdrant_client import QdrantClient, models

from zipsai.settings import (
    EMBEDDING_DIM,
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
    return QdrantClient(url=target)


@lru_cache(maxsize=1)
def get_client() -> QdrantClient:
    # 첫 요청이 들어올 때 만든다. 임포트 시점에 Qdrant로 붙지 않는다.
    return create_client()


def to_sparse_vector(weights: dict[str, float]) -> models.SparseVector:
    return models.SparseVector(
        indices=[int(index) for index in weights],
        values=list(weights.values()),
    )


def building_condition(building_id: int) -> models.FieldCondition:
    # is_tenant 키워드 인덱스에 맞추기 위해 문자열로 저장하고 문자열로 찾는다.
    # int로 넘기면 예외 없이 0건이 나온다.
    return models.FieldCondition(
        key="building_id",
        match=models.MatchValue(value=str(building_id)),
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
