from uuid import uuid4

from qdrant_client import QdrantClient, models

from zipsai.contracts.indexing import IndexingJobRequest
from zipsai.indexing.embed import EmbeddedChunk
from zipsai.integrations.qdrant import (
    DENSE_VECTOR,
    SPARSE_VECTOR,
    to_sparse_vector,
)
from zipsai.settings import QDRANT_COLLECTION


def upsert_document(
    client: QdrantClient,
    request: IndexingJobRequest,
    chunks: list[EmbeddedChunk],
    *,
    masked: bool = False,
    collection: str = QDRANT_COLLECTION,
) -> int:
    document_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="building_id",
                match=models.MatchValue(value=str(request.building_id)),
            ),
            models.FieldCondition(
                key="doc_id",
                match=models.MatchValue(value=request.doc_id),
            ),
        ]
    )
    client.delete(
        collection_name=collection,
        points_selector=models.FilterSelector(filter=document_filter),
        wait=True,
    )

    if not chunks:
        return 0

    client.upsert(
        collection_name=collection,
        points=[
            models.PointStruct(
                id=str(uuid4()),
                vector={
                    DENSE_VECTOR: chunk.dense,
                    SPARSE_VECTOR: to_sparse_vector(chunk.sparse),
                },
                payload={
                    # is_tenant 키워드 인덱스에 맞추기 위해 문자열로 저장한다.
                    "building_id": str(request.building_id),
                    "doc_id": request.doc_id,
                    "source_type": request.source_type.value,
                    "title": request.title,
                    "published_at": request.published_at.isoformat(),
                    "masked": masked,
                    "page": chunk.page,
                    "section": chunk.section,
                    "text": chunk.text,
                },
            )
            for chunk in chunks
        ],
        wait=True,
    )
    return len(chunks)
