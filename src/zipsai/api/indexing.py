import logging

from fastapi import APIRouter, status

from zipsai.contracts.indexing import (
    IndexingJobRequest,
    IndexingJobResponse,
)
from zipsai.indexing.store import InMemoryJobStore

logger = logging.getLogger(__name__)
router = APIRouter()
job_store = InMemoryJobStore()


@router.post(
    "/jobs",
    response_model=IndexingJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def create_job(payload: IndexingJobRequest) -> IndexingJobResponse:
    job_id = job_store.create()
    logger.info(
        "indexing_job_accepted job_id=%s trace_id=%s doc_id=%s",
        job_id,
        payload.trace_id,
        payload.doc_id,
    )
    return IndexingJobResponse(job_id=job_id, status="accepted")


@router.get("/jobs/{job_id}", response_model=IndexingJobResponse)
def get_job(job_id: str) -> IndexingJobResponse:
    return IndexingJobResponse(job_id=job_id, status=job_store.get_status(job_id))
