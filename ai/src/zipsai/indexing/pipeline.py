import logging
import tempfile
from collections.abc import Callable
from pathlib import Path

from qdrant_client import QdrantClient

from zipsai.contracts.indexing import IndexingJobRequest, JobStatus
from zipsai.indexing.chunk import chunk_pages
from zipsai.indexing.clean import apply_cleaning
from zipsai.indexing.embed import Encoder, embed_chunks
from zipsai.indexing.mask import apply_masking
from zipsai.indexing.parse import parse_document
from zipsai.indexing.upsert import upsert_document

logger = logging.getLogger(__name__)

Download = Callable[[str, Path], Path]


def run_indexing_job(
    request: IndexingJobRequest,
    *,
    download: Download,
    encoder: Encoder,
    client: QdrantClient,
) -> JobStatus:
    doc_id = request.doc_id
    try:
        with tempfile.TemporaryDirectory() as workdir:
            source = download(request.file_key, Path(workdir))
            pages = parse_document(source)

            masking = apply_masking(doc_id, pages)
            cleaning = apply_cleaning(doc_id, masking.pages)

            chunks = embed_chunks(chunk_pages(cleaning.pages), encoder)
            stored = upsert_document(
                client, request, chunks, masked=bool(masking.detections)
            )
    except Exception:
        # 배경 실행이라 예외를 삼키면 아무 기록도 남지 않는다.
        logger.exception("indexing_job_failed doc_id=%s", doc_id)
        return JobStatus.FAILED

    logger.info("indexing_job_succeeded doc_id=%s chunks=%d", doc_id, stored)
    return JobStatus.SUCCEEDED
