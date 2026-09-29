import logging
import tempfile
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from time import perf_counter

from qdrant_client import QdrantClient

from zipsai.contracts.indexing import IndexingJobRequest, JobStatus
from zipsai.indexing.chunk import chunk_pages
from zipsai.indexing.clean import apply_cleaning
from zipsai.indexing.embed import Encoder, embed_chunks
from zipsai.indexing.mask import apply_masking
from zipsai.indexing.parse import parse_document
from zipsai.indexing.upsert import upsert_document
from zipsai.integrations.embedding_client import BATCH_SIZE
from zipsai.observability import elapsed_ms, is_cold, stage

logger = logging.getLogger(__name__)

Download = Callable[[str, Path], Path]

# embedding 서비스가 거부하는 길이. 이 값을 넘는 청크가 있으면 embed가 422로 확정 실패한다.
MAX_TEXT_CHARS = 8000


def run_indexing_job(
    request: IndexingJobRequest,
    *,
    download: Download,
    encoder: Encoder,
    client: QdrantClient,
) -> JobStatus:
    doc_id = request.doc_id
    # 첫 작업은 docling이 OCR·레이아웃 모델을 올리느라 수십 초가 더 걸린다.
    cold = is_cold("indexing")
    started = perf_counter()

    try:
        with tempfile.TemporaryDirectory() as workdir:
            with stage("download", logger, cold=cold) as step:
                source = download(request.file_key, Path(workdir))
                # 로그로 쓰는 값이 본 작업을 죽이면 안 된다.
                step["bytes"] = source.stat().st_size if source.is_file() else None
                step["format"] = source.suffix.lstrip(".").lower()

            with stage("parse", logger, cold=cold) as step:
                pages = parse_document(source)
                step["pages"] = len(pages)
                step["chars"] = sum(len(str(page["text"])) for page in pages)

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

            with stage("chunk", logger) as step:
                chunks = chunk_pages(cleaning.pages)
                lengths = [len(chunk.text) for chunk in chunks]
                step["chunks"] = len(chunks)
                step["max_len"] = max(lengths, default=0)
                step["over_limit"] = sum(
                    1 for length in lengths if length > MAX_TEXT_CHARS
                )

            with stage("embed", logger, cold=cold) as step:
                step["texts"] = len(chunks)
                step["batches"] = -(-len(chunks) // BATCH_SIZE)
                embedded = embed_chunks(chunks, encoder)

            with stage("upsert", logger) as step:
                stored = upsert_document(
                    client, request, embedded, masked=bool(masking.detections)
                )
                step["points"] = stored
    except Exception as error:
        # 배경 실행이라 예외를 삼키면 아무 기록도 남지 않는다.
        # 스택은 단계 로그가 이미 남겼다. 여기서 또 실으면 같은 줄이 두 번 쌓인다.
        logger.exception(
            "index_done",
            exc_info=False,
            extra={
                "outcome": "fail",
                "total_ms": elapsed_ms(started),
                "failed_stage": getattr(error, "stage", None),
                "error_type": type(error).__name__,
            },
        )
        return JobStatus.FAILED

    logger.info(
        "index_done",
        extra={
            "outcome": "ok",
            "total_ms": elapsed_ms(started),
            "failed_stage": None,
            "chunks": len(chunks),
            "points": stored,
            "doc_id": doc_id,
        },
    )
    return JobStatus.SUCCEEDED
