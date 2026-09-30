import logging
from functools import lru_cache
from time import perf_counter
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, status
from fastapi.responses import JSONResponse
from qdrant_client import QdrantClient

from zipsai.contracts.indexing import IndexingJobRequest, JobStatus
from zipsai.indexing.pipeline import run_indexing_job
from zipsai.indexing.upsert import delete_missing_documents
from zipsai.integrations.embedding_client import HttpEncoder
from zipsai.integrations.qdrant import create_client, ensure_collection
from zipsai.integrations.s3 import download
from zipsai.observability import bind, elapsed_ms, stage
from zipsai.tracing import trace_job

logger = logging.getLogger(__name__)
router = APIRouter()

ACCEPTED = status.HTTP_202_ACCEPTED


@lru_cache(maxsize=1)
def _dependencies() -> tuple[QdrantClient, HttpEncoder]:
    # 첫 요청이 들어올 때 만든다. 임포트 시점에 Qdrant·Embedding에 붙지 않는다.
    client = create_client()
    ensure_collection(client)
    return client, HttpEncoder()


def _run_job(payload: IndexingJobRequest, job_id: str) -> None:
    started = perf_counter()
    with (
        bind(
            job_id=job_id,
            trace_id=payload.trace_id,
            doc_id=payload.doc_id,
            building_id=payload.building_id,
        ),
        trace_job(payload, job_id) as trace,
    ):
        try:
            with stage("setup", logger):
                client, encoder = _dependencies()
        except Exception as error:
            # 응답은 이미 202로 나갔다. 여기서 터지면 로그가 유일한 흔적이다.
            if trace is not None:
                trace.update(output={"outcome": "fail", "failed_stage": "setup"})
            logger.exception(
                "job_done",
                extra=_job_fields(started, "fail", failed_stage="setup", error=error),
            )
            return

        # 색인이 먼저다. 정리가 앞서면 이번에 넣을 문서가 잠깐 검색에서 빠진다.
        if payload.has_document:
            result = run_indexing_job(
                payload, download=download, encoder=encoder, client=client
            )
            if result is JobStatus.FAILED:
                if trace is not None:
                    trace.update(output={"outcome": "fail", "failed_stage": "index"})
                logger.error(
                    "job_done", extra=_job_fields(started, "fail", failed_stage="index")
                )
                return

        if payload.valid_doc_ids is None:
            if trace is not None:
                trace.update(output={"outcome": "ok"})
            logger.info("job_done", extra=_job_fields(started, "ok"))
            return

        try:
            with stage("reconcile", logger) as step:
                delete_missing_documents(
                    client, payload.building_id, payload.valid_doc_ids
                )
                step["kept"] = len(payload.valid_doc_ids)
        except Exception as error:
            # 스택은 reconcile 단계 로그가 남겼다.
            if trace is not None:
                trace.update(output={"outcome": "fail", "failed_stage": "reconcile"})
            logger.exception(
                "job_done",
                exc_info=False,
                extra=_job_fields(
                    started, "fail", failed_stage="reconcile", error=error
                ),
            )
            return

        if trace is not None:
            trace.update(output={"outcome": "ok"})
        logger.info("job_done", extra=_job_fields(started, "ok"))


def _job_fields(
    started: float,
    outcome: str,
    *,
    failed_stage: str | None = None,
    error: Exception | None = None,
) -> dict[str, object]:
    """요청 하나에 정확히 한 줄. job_accepted와 짝이 맞는지로 작업 증발을 탐지한다."""
    return {
        "outcome": outcome,
        "total_ms": elapsed_ms(started),
        "failed_stage": failed_stage,
        "error_type": type(error).__name__ if error else None,
    }


@router.post("/jobs", status_code=ACCEPTED, response_class=JSONResponse)
def create_job(
    payload: IndexingJobRequest, background_tasks: BackgroundTasks
) -> JSONResponse:
    # 색인 계약에는 turn_id가 없다. 단계 로그를 한 작업으로 묶으려면 여기서 발급해야 한다.
    job_id = str(uuid4())
    logger.info(
        "job_accepted",
        extra={
            "job_id": job_id,
            "trace_id": payload.trace_id,
            "doc_id": payload.doc_id,
            "building_id": payload.building_id,
            "has_document": payload.has_document,
            "reconcile_count": len(payload.valid_doc_ids)
            if payload.valid_doc_ids
            else 0,
        },
    )
    background_tasks.add_task(_run_job, payload, job_id)
    return JSONResponse(status_code=ACCEPTED, content={"trace_id": payload.trace_id})
