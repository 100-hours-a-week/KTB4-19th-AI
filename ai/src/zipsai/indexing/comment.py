import logging
from datetime import datetime
from uuid import NAMESPACE_OID, uuid5

from qdrant_client import QdrantClient, models

from zipsai.contracts.comment import (
    CommentDeleteItem,
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
    "tombstone": "VECTOR_STORE_FAILED",
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
        base = {
            "complaint_detail_id": item.complaint_detail_id,
            "updated_at": item.updated_at_raw,
        }
        point_id = comment_point_id(item.building_id, item.complaint_detail_id)
        try:
            with stage("lookup", logger):
                stored = _stored_updated_at(client, collection, point_id)
            stale = _stale_result(stored, item.updated_at, base)
            if stale is not None:
                results.append(stale)
                continue
            # 지우지 않고 삭제 시각만 남긴다. 기준이 사라지면 삭제 전에 보낸 요청이
            # 재시도로 늦게 도착했을 때 걸러낼 수 없어 지운 답변이 되살아난다.
            with stage("tombstone", logger):
                _tombstone(client, collection, point_id, item.building_id, item)
        except Exception as error:  # noqa: BLE001 — 항목 하나의 실패가 나머지를 막지 않는다
            results.append(_failed(error, base))
        else:
            # 없던 댓글을 지워도 성공이다. 툼스톤이 남아 이후 옛 요청을 막는다.
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
        # 툼스톤의 삭제 시각도 기준이 된다. 삭제 뒤 늦게 온 옛 요청은 여기서 걸린다.
        stale = _stale_result(stored, item.updated_at, base)
        if stale is not None:
            return stale

        with stage("mask", logger):
            comment, text, location, masked = _masked_text(item)

        if len(comment) < COMMENT_MIN_CHARS:
            # 새 내용이 쓸모없으면 옛 답변도 같이 내려야 낡은 답이 계속 나가지 않는다.
            # 지우지 않고 툼스톤으로 바꿔야 이 시각보다 옛 요청이 되살아나지 않는다.
            with stage("tombstone", logger):
                _tombstone(client, collection, point_id, item.building_id, item)
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
                        payload=_payload(item, comment, text, location, masked),
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


def _payload(
    item: CommentItem, answer: str, text: str, location: str | None, masked: bool
) -> dict:
    return {
        "building_id": str(item.building_id),
        "source_type": COMMENT_SOURCE_TYPE,
        # 입주민 검색은 COMPLAINT를 거른다(resident_visible_filter). 저장은 이후 에이전트용이다.
        "complaint_type": item.complaint_type,
        "complaint_detail_id": item.complaint_detail_id,
        "title": COMMENT_TITLE,
        "masked": masked,
        "page": 1,
        "section": None,
        "text": text,
        # 근거 표기에는 마스킹한 관리자 답변만 보인다. text 앞부분은 입주민이 쓴 질문이다.
        "answer": answer,
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


def _stale_result(
    stored: datetime | None, requested: datetime, base: dict
) -> ItemResult | None:
    """저장된 시각이 같거나 더 새로우면 이 요청은 반영하지 않는다."""
    if stored is None or stored < requested:
        return None
    # 같으면 이미 반영된 요청, 오래됐으면 지연된 재전송이다.
    if stored == requested:
        return ItemResult(status=ResultStatus.SUCCEEDED, **base)
    return ItemResult(status=ResultStatus.SKIPPED, reason_code="STALE_UPDATE", **base)


def _tombstone(
    client: QdrantClient,
    collection: str,
    point_id: str,
    building_id: int,
    item: CommentItem | CommentDeleteItem,
) -> None:
    # 벡터가 없어 어떤 검색에도 걸리지 않고, ID로 조회하면 updated_at이 보인다.
    # source_type이 있어 문서 정리(delete_missing_documents)에도 지워지지 않는다.
    client.upsert(
        collection_name=collection,
        points=[
            models.PointStruct(
                id=point_id,
                vector={},
                payload={
                    "building_id": str(building_id),
                    "source_type": COMMENT_SOURCE_TYPE,
                    "complaint_detail_id": item.complaint_detail_id,
                    "deleted": True,
                    "updated_at": item.updated_at.isoformat(),
                },
            )
        ],
        wait=True,
    )


def _failed(error: Exception, base: dict) -> ItemResult:
    reason = _STAGE_REASON.get(getattr(error, "stage", None), "VECTOR_STORE_FAILED")
    return ItemResult(status=ResultStatus.FAILED, reason_code=reason, **base)
