import logging
from functools import lru_cache

from fastapi import APIRouter, BackgroundTasks, status
from qdrant_client import QdrantClient

from zipsai.contracts.indexing import (
    IndexingJobRequest,
    IndexingJobResponse,
)
from zipsai.indexing.pipeline import run_indexing_job
from zipsai.indexing.store import InMemoryJobStore
from zipsai.integrations.embedding_client import HttpEncoder
from zipsai.integrations.qdrant import create_client, ensure_collection
from zipsai.integrations.s3 import download

logger = logging.getLogger(__name__)
router = APIRouter()
job_store = InMemoryJobStore()


@lru_cache(maxsize=1)
def _dependencies() -> tuple[QdrantClient, HttpEncoder]:
    # 첫 작업이 들어올 때 만든다. 임포트 시점에 Qdrant·Embedding에 붙지 않는다.
    client = create_client()
    ensure_collection(client)
    return client, HttpEncoder()


def _run_job(job_id: str, payload: IndexingJobRequest) -> None:
    client, encoder = _dependencies()
    run_indexing_job(
        job_store,
        job_id,
        payload,
        download=download,
        encoder=encoder,
        client=client,
    )


@router.post(
    "/jobs",
    response_model=IndexingJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def create_job(
    payload: IndexingJobRequest, background_tasks: BackgroundTasks
) -> IndexingJobResponse:
    job_id = job_store.create()
    logger.info(
        "indexing_job_accepted job_id=%s trace_id=%s doc_id=%s",
        job_id,
        payload.trace_id,
        payload.doc_id,
    )
    background_tasks.add_task(_run_job, job_id, payload)
    return IndexingJobResponse(job_id=job_id, status="accepted")


@router.get("/jobs/{job_id}", response_model=IndexingJobResponse)
def get_job(job_id: str) -> IndexingJobResponse:
    return IndexingJobResponse(job_id=job_id, status=job_store.get_status(job_id))
