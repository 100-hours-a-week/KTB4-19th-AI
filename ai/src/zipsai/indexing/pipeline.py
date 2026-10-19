import logging
import tempfile
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from time import perf_counter

from qdrant_client import QdrantClient

from zipsai.contracts.indexing import IndexingJobRequest, JobStatus
from zipsai.errors import EmptyDocumentError, error_code
from zipsai.indexing.chunk import chunk_pages
from zipsai.indexing.clean import apply_cleaning
from zipsai.indexing.embed import Encoder, embed_chunks
from zipsai.indexing.mask import apply_masking
from zipsai.indexing.parse import parse_document
from zipsai.indexing.upsert import upsert_document
from zipsai.integrations.embedding_client import BATCH_SIZE
from zipsai.metrics import observe_index_job, observe_tables
from zipsai.observability import collect_timings, elapsed_ms, is_cold, stage

logger = logging.getLogger(__name__)

Download = Callable[[str, Path], Path]

# embedding 서비스가 거부하는 길이. 이 값을 넘는 청크가 있으면 embed가 422로 확정 실패한다.
MAX_TEXT_CHARS = 8000

# 실패한 단계 → BE 통지의 reason_code. 단계 이름은 로그의 failed_stage와 같다.
_STAGE_REASON = {
    "download": "FILE_DOWNLOAD_FAILED",
    "parse": "PARSE_FAILED",
    "mask": "MASK_FAILED",
    "clean": "CLEAN_FAILED",
    "chunk": "CHUNK_FAILED",
    "embed": "EMBEDDING_FAILED",
    "upsert": "VECTOR_STORE_FAILED",
}


def run_indexing_job(
    request: IndexingJobRequest,
    *,
    download: Download,
    encoder: Encoder,
    client: QdrantClient,
    on_fail: Callable[[str], None] | None = None,
) -> JobStatus:
    doc_id = request.doc_id
    # 첫 작업은 docling이 OCR·레이아웃 모델을 올리느라 수십 초가 더 걸린다.
    cold = is_cold("indexing")
    started = perf_counter()
    doc_bytes: int | None = None
    page_count: int | None = None
    chunk_count: int | None = None
    embed_batches: int | None = None
    point_count: int | None = None

    with collect_timings() as timings:
        try:
            with tempfile.TemporaryDirectory() as workdir:
                with stage("download", logger, cold=cold) as step:
                    source = download(request.file_key, Path(workdir))
                    # 로그로 쓰는 값이 본 작업을 죽이면 안 된다.
                    doc_bytes = source.stat().st_size if source.is_file() else None
                    step["bytes"] = doc_bytes
                    step["format"] = source.suffix.lstrip(".").lower()

                with stage("parse", logger, cold=cold) as step:
                    pages = parse_document(source)
                    page_count = len(pages)
                    step["pages"] = page_count
                    step["chars"] = sum(len(str(page["text"])) for page in pages)
                    step["empty_pages"] = getattr(pages, "empty_pages", None)
                    step["min_chars_per_page"] = getattr(pages, "min_chars_per_page", None)
                    step["tables_found"] = getattr(pages, "tables_found", None)
                    step["tables_parsed"] = getattr(pages, "tables_parsed", None)
                    step["tables_failed"] = getattr(pages, "tables_failed", None)
                    step["table_fail_reasons"] = getattr(pages, "table_fail_reasons", None)
                    step["images_found"] = getattr(pages, "images_found", None)
                    step["parser_status"] = getattr(pages, "parser_status", None)
                    step["ocr_used"] = getattr(pages, "ocr_used", None)
                    step["failed_pages"] = getattr(pages, "failed_pages", None)
                    observe_tables(
                        parsed=getattr(pages, "tables_parsed", None) or 0,
                        failed=getattr(pages, "tables_failed", None),
                    )

                with stage("mask", logger) as step:
                    masking = apply_masking(pages)
                    step["detections"] = len(masking.detections)
                    step["counts"] = dict(
                        Counter(detection.kind.value for detection in masking.detections)
                    )

                with stage("clean", logger) as step:
                    cleaning = apply_cleaning(masking.pages)
                    step["removed_ratio"] = round(cleaning.removed_ratio, 3)
                    step["skipped"] = cleaning.skipped
                    step["skip_reason"] = getattr(cleaning, "skip_reason", None)

                with stage("chunk", logger) as step:
                    chunks = chunk_pages(cleaning.pages)
                    lengths = [len(chunk.text) for chunk in chunks]
                    chunk_count = len(chunks)
                    step["chunks"] = chunk_count
                    step["max_len"] = max(lengths, default=0)
                    step["min_len"] = min(lengths, default=0)
                    step["over_limit"] = sum(
                        1 for length in lengths if length > MAX_TEXT_CHARS
                    )

                with stage("embed", logger, cold=cold) as step:
                    step["texts"] = len(chunks)
                    embed_batches = -(-len(chunks) // BATCH_SIZE) if chunks else 0
                    step["batches"] = embed_batches
                    step["total_chars"] = sum(len(chunk.text) for chunk in chunks)
                    embedded = embed_chunks(chunks, encoder, request.trace_id)

                with stage("upsert", logger) as step:
                    stored = upsert_document(
                        client, request, embedded, masked=bool(masking.detections)
                    )
                    point_count = stored
                    step["points"] = stored
        except Exception as error:
            # 배경 실행이라 예외를 삼키면 아무 기록도 남지 않는다.
            # 스택은 단계 로그가 이미 남겼다. 여기서 또 실으면 같은 줄이 두 번 쌓인다.
            failed_stage = getattr(error, "stage", None)
            logger.exception(
                "index_done",
                exc_info=False,
                extra={
                    "outcome": "fail",
                    "total_ms": elapsed_ms(started),
                    "failed_stage": failed_stage,
                    "error_code": error_code(error),
                    "error_type": type(error).__name__,
                    "doc_bytes": doc_bytes,
                    "pages": page_count,
                    "chunks": chunk_count,
                    "embed_batches": embed_batches,
                    "stage_ms": dict(timings),
                },
            )
            observe_index_job("fail", failed_stage)
            if on_fail is not None:
                on_fail(
                    "EMPTY_DOCUMENT"
                    if isinstance(error, EmptyDocumentError)
                    else _STAGE_REASON.get(failed_stage, "INDEX_FAILED")
                )
            return JobStatus.FAILED

        logger.info(
            "index_done",
            extra={
                "outcome": "ok",
                "total_ms": elapsed_ms(started),
                "failed_stage": None,
                "chunks": chunk_count,
                "points": point_count,
                "doc_id": doc_id,
                "doc_bytes": doc_bytes,
                "pages": page_count,
                "embed_batches": embed_batches,
                "stage_ms": dict(timings),
            },
        )
    observe_index_job("ok", None)
    return JobStatus.SUCCEEDED
