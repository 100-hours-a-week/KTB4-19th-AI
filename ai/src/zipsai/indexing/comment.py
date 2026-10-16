import logging
from datetime import datetime
from uuid import NAMESPACE_OID, uuid5

from qdrant_client import QdrantClient, models

from zipsai.contracts.comment import (
    CommentDeleteRequest,
    CommentItem,
    CommentRequest,
)
from zipsai.contracts.notification import ItemResult, ResultStatus
from zipsai.indexing.chunk import Chunk
from zipsai.indexing.embed import Encoder, embed_chunks
from zipsai.indexing.mask import apply_masking
from zipsai.integrations.qdrant import (
    COMMENT_SOURCE_TYPE,
    DENSE_VECTOR,
    SPARSE_VECTOR,
    to_sparse_vector,
)
from zipsai.observability import stage
from zipsai.settings import COMMENT_MIN_CHARS, LOCATION_PLACEHOLDER, QDRANT_COLLECTION

logger = logging.getLogger(__name__)

# converse 근거 표기에 쓴다. 입주민에게 comment id를 보여줄 수는 없다.
COMMENT_TITLE = "관리자 답변"

# 단계 이름은 로그의 failed_stage와 같고, BE 통지의 reason_code로 옮겨 간다.
_STAGE_REASON = {
    "mask": "MASK_FAILED",
    "embed": "EMBEDDING_FAILED",
    "lookup": "VECTOR_STORE_FAILED",
    "upsert": "VECTOR_STORE_FAILED",
    "delete": "VECTOR_STORE_FAILED",
}


def comment_point_id(building_id: int, complaint_detail_id: int) -> str:
    # 같은 댓글은 항상 같은 ID가 되어, upsert 한 번이 교체를 원자적으로 해낸다.
    return str(uuid5(NAMESPACE_OID, f"{building_id}:{complaint_detail_id}"))


def upsert_comments(
    request: CommentRequest,
    *,
    encoder: Encoder,
    client: QdrantClient,
    collection: str = QDRANT_COLLECTION,
) -> list[ItemResult]:
    """항목마다 독립으로 처리한다. 한 건이 실패해도 나머지는 계속 간다."""
    return [
        _upsert_one(item, request.trace_id, encoder, client, collection)
        for item in request.items
    ]


def delete_comments(
    request: CommentDeleteRequest,
    *,
    client: QdrantClient,
    collection: str = QDRANT_COLLECTION,
) -> list[ItemResult]:
    results = []
    for item in request.items:
        base = {"complaint_detail_id": item.complaint_detail_id}
        try:
            with stage("delete", logger):
                _delete_point(
                    client,
                    collection,
                    comment_point_id(item.building_id, item.complaint_detail_id),
                )
        except Exception as error:  # noqa: BLE001 — 항목 하나의 실패가 나머지를 막지 않는다
            results.append(_failed(error, base))
        else:
            # 없는 ID를 지워도 성공이다. 재전송이 같은 결과가 되게 한다.
            results.append(ItemResult(status=ResultStatus.SUCCEEDED, **base))
    return results


def _upsert_one(
    item: CommentItem,
    trace_id: str,
    encoder: Encoder,
    client: QdrantClient,
    collection: str,
) -> ItemResult:
    base = {
        "complaint_detail_id": item.complaint_detail_id,
        "updated_at": item.updated_at_raw,
    }
    point_id = comment_point_id(item.building_id, item.complaint_detail_id)
    try:
        with stage("lookup", logger):
            stored = _stored_updated_at(client, collection, point_id)
        if stored is not None and stored >= item.updated_at:
            # 같으면 이미 반영된 요청, 오래됐으면 지연된 재전송이다.
            if stored == item.updated_at:
                return ItemResult(status=ResultStatus.SUCCEEDED, **base)
            return ItemResult(
                status=ResultStatus.SKIPPED, reason_code="STALE_UPDATE", **base
            )

        with stage("mask", logger):
            comment, text, location, masked = _masked_text(item)

        if len(comment) < COMMENT_MIN_CHARS:
            # 새 내용이 쓸모없으면 옛 답변도 같이 내려야 낡은 답이 계속 나가지 않는다.
            with stage("delete", logger):
                _delete_point(client, collection, point_id)
            return ItemResult(
                status=ResultStatus.SKIPPED, reason_code="COMMENT_TOO_SHORT", **base
            )

        with stage("embed", logger):
            embedded = embed_chunks([Chunk(text, 1, None)], encoder, trace_id)[0]

        with stage("upsert", logger):
            client.upsert(
                collection_name=collection,
                points=[
                    models.PointStruct(
                        id=point_id,
                        vector={
                            DENSE_VECTOR: embedded.dense,
                            SPARSE_VECTOR: to_sparse_vector(embedded.sparse),
                        },
                        payload=_payload(item, text, location, masked),
                    )
                ],
                wait=True,
            )
    except Exception as error:  # noqa: BLE001 — 항목 하나의 실패가 나머지를 막지 않는다
        return _failed(error, base)
    return ItemResult(status=ResultStatus.SUCCEEDED, **base)


def _masked_text(item: CommentItem) -> tuple[str, str, str | None, bool]:
    """마스킹한 답변 본문, 임베딩 입력 전문, 위치를 만든다."""
    location = _location(item)
    parts = [item.symptom, item.comment] + ([location] if location else [])
    result = apply_masking(
        [{"page": index, "text": part} for index, part in enumerate(parts)]
    )
    texts = [str(page["text"]).strip() for page in result.pages]
    masked_location = texts[2] if location else None
    return texts[1], "\n".join(texts), masked_location, bool(result.detections)


def _location(item: CommentItem) -> str | None:
    # "미상"은 위치를 모른다는 BE의 채움 문구다. 그대로 임베딩하면 검색어에 섞인다.
    if item.location is None or item.location.strip() in ("", LOCATION_PLACEHOLDER):
        return None
    return item.location.strip()


def _payload(item: CommentItem, text: str, location: str | None, masked: bool) -> dict:
    return {
        "building_id": str(item.building_id),
        "source_type": COMMENT_SOURCE_TYPE,
        "complaint_type": item.complaint_type,
        "complaint_detail_id": item.complaint_detail_id,
        "title": COMMENT_TITLE,
        "masked": masked,
        "page": 1,
        "section": None,
        "text": text,
        "updated_at": item.updated_at.isoformat(),
        "location": location,
        # 임베딩에는 쓰지 않고 이후 필터용으로만 보관한다.
        "issue_type": item.issue_type,
    }


def _stored_updated_at(
    client: QdrantClient, collection: str, point_id: str
) -> datetime | None:
    points = client.retrieve(
        collection_name=collection,
        ids=[point_id],
        with_payload=["updated_at"],
        with_vectors=False,
    )
    if not points:
        return None
    return datetime.fromisoformat(points[0].payload["updated_at"])


def _delete_point(client: QdrantClient, collection: str, point_id: str) -> None:
    client.delete(
        collection_name=collection,
        points_selector=models.PointIdsList(points=[point_id]),
        wait=True,
    )


def _failed(error: Exception, base: dict) -> ItemResult:
    reason = _STAGE_REASON.get(getattr(error, "stage", None), "VECTOR_STORE_FAILED")
    return ItemResult(status=ResultStatus.FAILED, reason_code=reason, **base)
