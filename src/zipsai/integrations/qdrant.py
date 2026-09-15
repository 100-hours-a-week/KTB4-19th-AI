from qdrant_client import QdrantClient, models

from zipsai.settings import (
    EMBEDDING_DIM,
    QDRANT_COLLECTION,
    QDRANT_URL,
)

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "sparse"


def create_client(url: str = QDRANT_URL) -> QdrantClient:
    if url == ":memory:":
        return QdrantClient(":memory:")
    return QdrantClient(url=url)


def to_sparse_vector(weights: dict[str, float]) -> models.SparseVector:
    return models.SparseVector(
        indices=[int(index) for index in weights],
        values=list(weights.values()),
    )


def ensure_collection(
    client: QdrantClient, collection: str = QDRANT_COLLECTION
) -> None:
    if client.collection_exists(collection_name=collection):
        return

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
    client.create_payload_index(
        collection_name=collection,
        field_name="building_id",
        field_schema=models.KeywordIndexParams(
            type=models.KeywordIndexType.KEYWORD,
            is_tenant=True,
        ),
        wait=True,
    )
