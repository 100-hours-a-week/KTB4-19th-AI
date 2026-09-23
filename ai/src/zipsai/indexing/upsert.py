from uuid import uuid4

from qdrant_client import QdrantClient, models

from zipsai.contracts.indexing import IndexingJobRequest
from zipsai.errors import EmptyDocumentError
from zipsai.indexing.embed import EmbeddedChunk
from zipsai.integrations.qdrant import (
    DENSE_VECTOR,
    SPARSE_VECTOR,
    building_condition,
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
            building_condition(request.building_id),
            models.FieldCondition(
                key="doc_id",
                match=models.MatchValue(value=request.doc_id),
            ),
        ]
    )
    if not chunks:
        # 지우기만 하고 넣지 않으면 기존 문서가 조용히 사라진다.
        raise EmptyDocumentError(f"No chunks to store for document '{request.doc_id}'")

    # 포인트를 먼저 만든다. sparse 변환이 실패해도 기존 벡터는 남는다.
    points = [
        models.PointStruct(
            id=str(uuid4()),
            vector={
                DENSE_VECTOR: chunk.dense,
                SPARSE_VECTOR: to_sparse_vector(chunk.sparse),
            },
            payload={
                "building_id": str(request.building_id),
                "doc_id": request.doc_id,
                "title": request.title,
                "masked": masked,
                "page": chunk.page,
                "section": chunk.section,
                "text": chunk.text,
            },
        )
        for chunk in chunks
    ]

    client.delete(
        collection_name=collection,
        points_selector=models.FilterSelector(filter=document_filter),
        wait=True,
    )
    client.upsert(collection_name=collection, points=points, wait=True)
    return len(points)


def delete_missing_documents(
    client: QdrantClient,
    building_id: int,
    valid_doc_ids: list[str],
    *,
    collection: str = QDRANT_COLLECTION,
) -> None:
    """백엔드가 보낸 유효 목록에 없는 문서를 건물 단위로 지운다.

    백엔드에서 삭제된 문서는 색인 요청이 오지 않으므로 이 경로가 유일한 회수 수단이다.
    """
    if not valid_doc_ids:
        # 빈 목록이면 건물 문서가 전부 지워진다. 계약에서 막지만 여기서도 막는다.
        raise ValueError("valid_doc_ids must not be empty")

    client.delete(
        collection_name=collection,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[building_condition(building_id)],
                must_not=[
                    models.FieldCondition(
                        key="doc_id",
                        match=models.MatchAny(any=valid_doc_ids),
                    )
                ],
            )
        ),
        wait=True,
    )
