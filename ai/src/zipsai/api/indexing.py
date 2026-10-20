import logging
from dataclasses import dataclass
from functools import lru_cache
from time import perf_counter
from typing import Literal
from uuid import uuid4

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from qdrant_client import QdrantClient

from zipsai.contracts.indexing import IndexingJobRequest, JobStatus
from zipsai.contracts.notification import ResultStatus
from zipsai.contracts.sync_response import DocumentResult, SyncBody
from zipsai.indexing.admission import (
    Rejected,
    acquire,
    document_key,
    in_flight,
    release,
)
from zipsai.indexing.pipeline import run_indexing_job
from zipsai.indexing.reasons import retryable_for
from zipsai.indexing.upsert import delete_missing_documents
from zipsai.integrations.embedding_client import HttpEncoder
from zipsai.integrations.qdrant import create_client, ensure_collection
from zipsai.integrations.s3 import download
from zipsai.metrics import record_indexing_result
from zipsai.observability import bind, elapsed_ms, stage
from zipsai.tracing import trace_job

logger = logging.getLogger(__name__)
router = APIRouter()


@dataclass
class JobOutcome:
    results: list[DocumentResult]
    pages: int | None = None
    chunks: int | None = None
    points: int | None = None


@lru_cache(maxsize=1)
def _dependencies() -> tuple[QdrantClient, HttpEncoder]:
    # 첫 요청이 들어올 때 만든다. 임포트 시점에 Qdrant·Embedding에 붙지 않는다.
    client = create_client()
    ensure_collection(client)
    return client, HttpEncoder()


def _document_result(
    kind: Literal["document_index", "document_cleanup"],
    status: ResultStatus,
    reason_code: str | None,
    doc_id: str | None = None,
) -> DocumentResult:
    return DocumentResult(
        kind=kind,
        status=status,
        reason_code=reason_code,
        retryable=retryable_for(reason_code),
        doc_id=doc_id,
    )


def _outcome(
    results: list[DocumentResult], stats: dict[str, int | None]
) -> JobOutcome:
    return JobOutcome(
        results,
        pages=stats["pages"],
        chunks=stats["chunks"],
        points=stats["points"],
    )


def _run_job(payload: IndexingJobRequest, job_id: str) -> JobOutcome:
    started = perf_counter()
    stats: dict[str, int | None] = {"pages": None, "chunks": None, "points": None}
    results: list[DocumentResult] = []

    def capture(pages: int | None, chunks: int | None, points: int | None) -> None:
        stats["pages"] = pages
        stats["chunks"] = chunks
        stats["points"] = points

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
            if trace is not None:
                trace.update(output={"outcome": "fail", "failed_stage": "setup"})
            logger.exception(
                "job_done",
                extra=_job_fields(started, "fail", failed_stage="setup", error=error),
            )
            if payload.has_document:
                results.append(
                    _document_result(
                        "document_index",
                        ResultStatus.FAILED,
                        "SETUP_FAILED",
                        payload.doc_id,
                    )
                )
            if payload.valid_doc_ids is not None:
                results.append(
                    _document_result(
                        "document_cleanup", ResultStatus.FAILED, "SETUP_FAILED"
                    )
                )
            return _outcome(results, stats)

        # 색인이 먼저다. 정리가 앞서면 이번에 넣을 문서가 잠깐 검색에서 빠진다.
        if payload.has_document:
            reasons: list[str] = []
            result = run_indexing_job(
                payload,
                download=download,
                encoder=encoder,
                client=client,
                on_fail=reasons.append,
                on_stats=capture,
            )
            if result is JobStatus.SUCCEEDED:
                results.append(
                    _document_result(
                        "document_index",
                        ResultStatus.SUCCEEDED,
                        None,
                        payload.doc_id,
                    )
                )
            else:
                results.append(
                    _document_result(
                        "document_index",
                        ResultStatus.FAILED,
                        reasons[0] if reasons else "INDEX_FAILED",
                        payload.doc_id,
                    )
                )
                if payload.valid_doc_ids is not None:
                    results.append(
                        _document_result(
                            "document_cleanup",
                            ResultStatus.SKIPPED,
                            "INDEX_FAILED",
                        )
                    )
            if result is JobStatus.FAILED:
                if trace is not None:
                    trace.update(output={"outcome": "fail", "failed_stage": "index"})
                logger.error(
                    "job_done", extra=_job_fields(started, "fail", failed_stage="index")
                )
                return _outcome(results, stats)

        if payload.valid_doc_ids is None:
            if trace is not None:
                trace.update(output={"outcome": "ok"})
            logger.info("job_done", extra=_job_fields(started, "ok"))
            return _outcome(results, stats)

        try:
            with stage("reconcile", logger) as step:
                delete_missing_documents(
                    client, payload.building_id, payload.valid_doc_ids
                )
                step["kept"] = len(payload.valid_doc_ids)
        except Exception as error:
            if trace is not None:
                trace.update(output={"outcome": "fail", "failed_stage": "reconcile"})
            logger.exception(
                "job_done",
                exc_info=False,
                extra=_job_fields(
                    started, "fail", failed_stage="reconcile", error=error
                ),
            )
            results.append(
                _document_result(
                    "document_cleanup", ResultStatus.FAILED, "CLEANUP_FAILED"
                )
            )
            return _outcome(results, stats)

        if trace is not None:
            trace.update(output={"outcome": "ok"})
        logger.info("job_done", extra=_job_fields(started, "ok"))
        results.append(
            _document_result("document_cleanup", ResultStatus.SUCCEEDED, None)
        )
        return _outcome(results, stats)


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


@router.post("/jobs", response_class=JSONResponse)
def create_job(payload: IndexingJobRequest) -> JSONResponse:
    # 색인 계약에는 turn_id가 없다. 단계 로그를 한 작업으로 묶으려면 여기서 발급해야 한다.
    started = perf_counter()
    job_id = str(uuid4())
    key = document_key(payload)
    try:
        acquire(key)
    except Rejected as rejected:
        return _rejected(payload, rejected, job_id)
    # 거절된 요청은 indexing_rejected만 남긴다. 접수 로그는 들어간 요청만 job_done과 짝이 된다.
    file_key_path, file_key_ext = _file_key_parts(payload.file_key)
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
            "doc_title": payload.title,
            "file_key_path": file_key_path,
            "file_key_ext": file_key_ext,
            "valid_doc_ids": None
            if payload.valid_doc_ids is None
            else ",".join(payload.valid_doc_ids),
            "valid_doc_count": None
            if payload.valid_doc_ids is None
            else len(payload.valid_doc_ids),
        },
    )
    try:
        outcome = _run_job(payload, job_id)
    finally:
        release(key)
    _log_index_result(payload, job_id, outcome, started)
    for item in outcome.results:
        record_indexing_result(item.kind, item.status.value, item.reason_code)
    body = SyncBody(trace_id=payload.trace_id, results=outcome.results).to_body()
    return JSONResponse(status_code=200, content=body)


def _file_key_parts(file_key: str | None) -> tuple[str | None, str | None]:
    if not file_key:
        return None, None
    path = file_key.split("?", 1)[0]
    name = path.rsplit("/", 1)[-1]
    if "." not in name:
        return path, None
    suffix = name.rsplit(".", 1)[-1]
    return path, suffix or None


def _log_index_result(
    payload: IndexingJobRequest, job_id: str, outcome: JobOutcome, started: float
) -> None:
    index = next((item for item in outcome.results if item.kind == "document_index"), None)
    cleanup = next(
        (item for item in outcome.results if item.kind == "document_cleanup"), None
    )
    logger.info(
        "index_result",
        extra={
            "trace_id": payload.trace_id,
            "job_id": job_id,
            "doc_id": payload.doc_id,
            "building_id": payload.building_id,
            "http_status": 200,
            "index_status": None if index is None else index.status.value,
            "index_reason": None if index is None else index.reason_code,
            "index_retryable": None if index is None else index.retryable,
            "cleanup_status": None if cleanup is None else cleanup.status.value,
            "cleanup_reason": None if cleanup is None else cleanup.reason_code,
            "cleanup_retryable": None if cleanup is None else cleanup.retryable,
            "total_ms": elapsed_ms(started),
            "pages": outcome.pages,
            "chunks": outcome.chunks,
            "points": outcome.points,
            "reconcile_count": None
            if payload.valid_doc_ids is None
            else len(payload.valid_doc_ids),
        },
    )


def _rejected(
    payload: IndexingJobRequest, rejected: Rejected, job_id: str
) -> JSONResponse:
    logger.info(
        "indexing_rejected",
        extra={
            "trace_id": payload.trace_id,
            "job_id": job_id,
            "route": "jobs",
            "error_code": rejected.error_code,
            "doc_id": payload.doc_id,
            "building_id": payload.building_id,
            "items": None,
            "indexing_in_flight": in_flight(),
            "retry_after_s": rejected.retry_after,
        },
    )
    return JSONResponse(
        status_code=rejected.status_code,
        content={"trace_id": payload.trace_id, "error_code": rejected.error_code},
        headers={"Retry-After": str(rejected.retry_after)},
    )
