import logging
from functools import lru_cache

from fastapi import APIRouter, BackgroundTasks, Response, status
from qdrant_client import QdrantClient

from zipsai.contracts.indexing import IndexingJobRequest
from zipsai.indexing.pipeline import run_indexing_job
from zipsai.indexing.upsert import delete_missing_documents
from zipsai.integrations.embedding_client import HttpEncoder
from zipsai.integrations.qdrant import create_client, ensure_collection
from zipsai.integrations.s3 import download

logger = logging.getLogger(__name__)
router = APIRouter()

ACCEPTED = status.HTTP_202_ACCEPTED


@lru_cache(maxsize=1)
def _dependencies() -> tuple[QdrantClient, HttpEncoder]:
    # 첫 요청이 들어올 때 만든다. 임포트 시점에 Qdrant·Embedding에 붙지 않는다.
    client = create_client()
    ensure_collection(client)
    return client, HttpEncoder()


def _run_job(payload: IndexingJobRequest) -> None:
    try:
        client, encoder = _dependencies()
    except Exception:
        # 응답은 이미 202로 나갔다. 여기서 터지면 로그가 유일한 흔적이다.
        logger.exception("indexing_job_setup_failed doc_id=%s", payload.doc_id)
        return

    # 색인이 먼저다. 정리가 앞서면 이번에 넣을 문서가 잠깐 검색에서 빠진다.
    if payload.has_document:
        run_indexing_job(payload, download=download, encoder=encoder, client=client)

    if payload.valid_doc_ids is None:
        return

    try:
        delete_missing_documents(client, payload.building_id, payload.valid_doc_ids)
    except Exception:
        logger.exception("reconcile_failed building_id=%s", payload.building_id)
        return

    logger.info(
        "reconcile_succeeded building_id=%s kept=%d",
        payload.building_id,
        len(payload.valid_doc_ids),
    )


@router.post("/jobs", status_code=ACCEPTED, response_class=Response)
def create_job(
    payload: IndexingJobRequest, background_tasks: BackgroundTasks
) -> Response:
    logger.info(
        "indexing_job_accepted building_id=%s doc_id=%s reconcile=%s",
        payload.building_id,
        payload.doc_id,
        len(payload.valid_doc_ids) if payload.valid_doc_ids else 0,
    )
    background_tasks.add_task(_run_job, payload)
    return Response(status_code=ACCEPTED)
