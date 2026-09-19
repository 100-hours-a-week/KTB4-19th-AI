import logging
import tempfile
from collections.abc import Callable
from pathlib import Path

from qdrant_client import QdrantClient

from zipsai.contracts.indexing import IndexingJobRequest, JobStatus
from zipsai.errors import EmptyDocumentError
from zipsai.indexing.chunk import chunk_pages
from zipsai.indexing.clean import apply_cleaning
from zipsai.indexing.embed import Encoder, embed_chunks
from zipsai.indexing.mask import apply_masking
from zipsai.indexing.parse import parse_pdf
from zipsai.indexing.store import InMemoryJobStore
from zipsai.indexing.upsert import upsert_document

logger = logging.getLogger(__name__)

Download = Callable[[str, Path], Path]


def run_indexing_job(
    store: InMemoryJobStore,
    job_id: str,
    request: IndexingJobRequest,
    *,
    download: Download,
    encoder: Encoder,
    client: QdrantClient,
) -> JobStatus:
    store.set_status(job_id, JobStatus.RUNNING)
    try:
        with tempfile.TemporaryDirectory() as workdir:
            source = download(request.file_key, Path(workdir))
            pages = parse_pdf(source)

            masking = apply_masking(store, job_id, pages)
            if store.get_status(job_id) is JobStatus.NEEDS_REVIEW:
                # 보류 문서는 적재하지 않는다. 기존 버전의 벡터를 지우지 않아야
                # 관리자가 확인할 때까지 검색 결과에 공백이 생기지 않는다.
                logger.info("indexing_job_held job_id=%s reason=pii", job_id)
                return JobStatus.NEEDS_REVIEW

            cleaning = apply_cleaning(store, job_id, masking.pages)
            if store.get_status(job_id) is JobStatus.NEEDS_REVIEW:
                logger.info("indexing_job_held job_id=%s reason=cleaning", job_id)
                return JobStatus.NEEDS_REVIEW

            chunks = embed_chunks(chunk_pages(cleaning.pages), encoder)
            try:
                stored = upsert_document(client, request, chunks)
            except EmptyDocumentError:
                # 본문이 비면 기존 벡터를 지우지 않고 관리자 확인으로 넘긴다.
                logger.info("indexing_job_held job_id=%s reason=empty_document", job_id)
                store.set_status(job_id, JobStatus.NEEDS_REVIEW)
                return JobStatus.NEEDS_REVIEW
    except Exception:
        # 백그라운드 실행이라 예외를 삼키면 작업이 accepted로 굳는다.
        logger.exception(
            "indexing_job_failed job_id=%s doc_id=%s", job_id, request.doc_id
        )
        store.set_status(job_id, JobStatus.FAILED)
        return JobStatus.FAILED

    logger.info(
        "indexing_job_succeeded job_id=%s doc_id=%s chunks=%d",
        job_id,
        request.doc_id,
        stored,
    )
    store.set_status(job_id, JobStatus.SUCCEEDED)
    return JobStatus.SUCCEEDED
