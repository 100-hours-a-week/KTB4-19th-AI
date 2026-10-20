import logging
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Response, status

from zipsai.api.indexing import _dependencies
from zipsai.contracts.comment import CommentDeleteRequest, CommentRequest
from zipsai.contracts.notification import (
    ItemResult,
    Notification,
    NotificationKind,
    ResultStatus,
)
from zipsai.indexing.comment import delete_comments, upsert_comments
from zipsai.integrations.sqs import send_notification
from zipsai.observability import bind, stage

logger = logging.getLogger(__name__)
router = APIRouter()

ACCEPTED = status.HTTP_202_ACCEPTED


def _run(
    kind: NotificationKind, payload: CommentRequest | CommentDeleteRequest, job_id: str
) -> None:
    with bind(job_id=job_id, trace_id=payload.trace_id):
        try:
            with stage("setup", logger):
                client, encoder = _dependencies()
            if kind is NotificationKind.COMMENT_UPSERT:
                results = upsert_comments(payload, encoder=encoder, client=client)
            else:
                results = delete_comments(payload, client=client)
        except Exception:  # noqa: BLE001 — 결과를 통지로 돌려줘야 한다
            # 준비 단계가 죽으면 항목별 결과가 없다. 전부 실패로 알려야 BE가 재전송한다.
            results = [
                ItemResult(
                    status=ResultStatus.FAILED,
                    reason_code="SETUP_FAILED",
                    complaint_detail_id=item.complaint_detail_id,
                    updated_at=getattr(item, "updated_at_raw", None),
                )
                for item in payload.items
            ]
        logger.info(
            "comment_done",
            extra={
                "kind": kind.value,
                "items": len(results),
                "failed": sum(r.status is ResultStatus.FAILED for r in results),
            },
        )
        send_notification(
            Notification(kind=kind, trace_id=payload.trace_id, results=results)
        )


@router.post("/comment", status_code=ACCEPTED)
def upsert(payload: CommentRequest, background_tasks: BackgroundTasks) -> Response:
    job_id = str(uuid4())
    logger.info(
        "comment_accepted",
        extra={
            "job_id": job_id,
            "trace_id": payload.trace_id,
            "items": len(payload.items),
        },
    )
    background_tasks.add_task(_run, NotificationKind.COMMENT_UPSERT, payload, job_id)
    return Response(status_code=ACCEPTED)


@router.post("/comment/delete", status_code=ACCEPTED)
def delete(
    payload: CommentDeleteRequest, background_tasks: BackgroundTasks
) -> Response:
    job_id = str(uuid4())
    logger.info(
        "comment_accepted",
        extra={
            "job_id": job_id,
            "trace_id": payload.trace_id,
            "items": len(payload.items),
        },
    )
    background_tasks.add_task(_run, NotificationKind.COMMENT_DELETE, payload, job_id)
    return Response(status_code=ACCEPTED)
