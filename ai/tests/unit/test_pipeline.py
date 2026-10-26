import logging
from pathlib import Path
from typing import Any

import pytest
from qdrant_client import QdrantClient

from zipsai.contracts.indexing import IndexingJobRequest, JobStatus
from zipsai.errors import (
    DocumentFetchError,
    EmbeddingError,
    EmbeddingRequestInvalidError,
)
from zipsai.indexing import pipeline
from zipsai.indexing.chunk import Chunk
from zipsai.indexing.embed import count_embed_texts
from zipsai.indexing.parse import ParsedPages
from zipsai.indexing.pipeline import run_indexing_job
from zipsai.indexing.reasons import retryable_for
from zipsai.integrations.embedding_client import MAX_TEXT_CHARS
from zipsai.integrations.qdrant import create_client, ensure_collection
from zipsai.observability import JsonFormatter
from zipsai.settings import EMBEDDING_DIM, QDRANT_COLLECTION

FIXTURE = Path(__file__).parents[1] / "fixtures" / "two-pages.pdf"


class FakeEncoder:
    def encode(
        self, texts: list[str], trace_id: str
    ) -> tuple[list[list[float]], list[dict[str, float]]]:
        return [[0.0] * EMBEDDING_DIM for _ in texts], [{} for _ in texts]


def make_request() -> IndexingJobRequest:
    return IndexingJobRequest(
        building_id=101,
        trace_id="trace-001",
        doc_id="notice-001",
        title="물탱크 청소 안내",
        file_key="buildings/101/notice-001.pdf",
    )


def copy_fixture(_: str, target_dir: Path) -> Path:
    target = target_dir / FIXTURE.name
    target.write_bytes(FIXTURE.read_bytes())
    return target


def stub_download(_: str, target_dir: Path) -> Path:
    return target_dir / "unused.pdf"


def stub_parse(pages: list[dict[str, Any]]):
    def parse(_source: Any) -> list[dict[str, Any]]:
        return [dict(page) for page in pages]

    return parse


def stored_points(client: QdrantClient) -> int:
    return client.count(collection_name=QDRANT_COLLECTION, exact=True).count


@pytest.fixture
def qdrant() -> QdrantClient:
    client = create_client(":memory:")
    ensure_collection(client)
    return client


def test_parse_log_records_parser_route(
    monkeypatch: pytest.MonkeyPatch,
    qdrant: QdrantClient,
    caplog: pytest.LogCaptureFixture,
) -> None:
    pages = ParsedPages([{"page": 1, "text": "본문"}])
    pages.parser = "pdfplumber"
    pages.fallback_reason = None
    pages.text_layer = "ok"
    monkeypatch.setattr(pipeline, "parse_document", lambda _source: pages)

    with caplog.at_level(logging.INFO, logger="zipsai.indexing.pipeline"):
        result = run_indexing_job(
            make_request(),
            download=stub_download,
            encoder=FakeEncoder(),
            client=qdrant,
        )

    assert result is JobStatus.SUCCEEDED
    parse_logs = [
        record
        for record in caplog.records
        if record.getMessage() == "stage_done" and getattr(record, "stage", None) == "parse"
    ]
    assert parse_logs[-1].parser == "pdfplumber"
    assert parse_logs[-1].text_layer == "ok"
    assert parse_logs[-1].fallback_reason is None


def test_embed_stage_counts_duplicate_texts(
    monkeypatch: pytest.MonkeyPatch,
    qdrant: QdrantClient,
    caplog: pytest.LogCaptureFixture,
) -> None:
    chunks = [
        Chunk("같은", 1, None),
        Chunk("같은", 1, None),
        Chunk("같은", 1, None),
        Chunk("다른1", 1, None),
        Chunk("다른2", 1, None),
    ]
    assert count_embed_texts(chunks) == (5, 3)
    monkeypatch.setattr(pipeline, "parse_document", lambda _source: [{"page": 1, "text": "본문"}])
    monkeypatch.setattr(pipeline, "chunk_pages", lambda _pages: chunks)

    with caplog.at_level(logging.INFO, logger="zipsai.indexing.pipeline"):
        result = run_indexing_job(
            make_request(),
            download=stub_download,
            encoder=FakeEncoder(),
            client=qdrant,
        )

    assert result is JobStatus.SUCCEEDED
    embed_logs = [
        record
        for record in caplog.records
        if record.getMessage() == "stage_done" and getattr(record, "stage", None) == "embed"
    ]
    assert embed_logs[-1].unique_texts == 3
    assert embed_logs[-1].duplicate_texts == 2


def test_success_path_upserts_chunks_and_marks_succeeded(qdrant: QdrantClient) -> None:
    result = run_indexing_job(
        make_request(),
        download=copy_fixture,
        encoder=FakeEncoder(),
        client=qdrant,
    )

    assert result is JobStatus.SUCCEEDED
    assert stored_points(qdrant) > 0


def payloads(client: QdrantClient) -> list[dict]:
    records, _ = client.scroll(
        collection_name=QDRANT_COLLECTION, limit=100, with_payload=True
    )
    return [record.payload for record in records]


def test_pii_is_masked_and_still_indexed(
    monkeypatch: pytest.MonkeyPatch,
    qdrant: QdrantClient,
) -> None:
    # v1은 보류하지 않는다. 치환본을 적재하고 masked 표시만 남긴다.
    monkeypatch.setattr(
        pipeline,
        "parse_document",
        stub_parse([{"page": 1, "text": "문의는 admin@example.com 으로 주세요."}]),
    )

    result = run_indexing_job(
        make_request(),
        download=stub_download,
        encoder=FakeEncoder(),
        client=qdrant,
    )

    assert result is JobStatus.SUCCEEDED
    stored = payloads(qdrant)
    assert stored and all(payload["masked"] is True for payload in stored)
    assert all("admin@example.com" not in payload["text"] for payload in stored)
    assert any("[이메일]" in payload["text"] for payload in stored)


def test_excessive_cleaning_indexes_the_original_text(
    monkeypatch: pytest.MonkeyPatch,
    qdrant: QdrantClient,
) -> None:
    # 정제가 과하면 정제를 포기한다. 지워졌던 머리글이 본문에 남아야 한다.
    repeated = "공동주택 관리사무소 알림 머리글"
    monkeypatch.setattr(
        pipeline,
        "parse_document",
        stub_parse(
            [
                {"page": number, "text": f"{repeated}\n본문{number}"}
                for number in (1, 2, 3)
            ]
        ),
    )

    result = run_indexing_job(
        make_request(),
        download=stub_download,
        encoder=FakeEncoder(),
        client=qdrant,
    )

    assert result is JobStatus.SUCCEEDED
    stored = payloads(qdrant)
    assert stored
    assert any(repeated in payload["text"] for payload in stored)
    assert all(payload["masked"] is False for payload in stored)


def test_download_failure_marks_job_failed(qdrant: QdrantClient) -> None:
    def failing_download(file_key: str, _: Path) -> Path:
        raise DocumentFetchError(f"Failed to fetch '{file_key}'")

    reasons: list[str] = []
    result = run_indexing_job(
        make_request(),
        download=failing_download,
        encoder=FakeEncoder(),
        client=qdrant,
        on_fail=reasons.append,
    )

    assert result is JobStatus.FAILED
    assert reasons == ["FILE_DOWNLOAD_FAILED"]
    assert stored_points(qdrant) == 0


def test_empty_document_fails_the_job_and_keeps_existing_vectors(
    monkeypatch: pytest.MonkeyPatch,
    qdrant: QdrantClient,
) -> None:
    # 먼저 같은 doc_id로 정상 적재해 기존 벡터를 만든다.
    run_indexing_job(
        make_request(),
        download=copy_fixture,
        encoder=FakeEncoder(),
        client=qdrant,
    )
    before = stored_points(qdrant)
    assert before > 0

    # 본문이 빈 문서로 재색인을 시도한다.
    monkeypatch.setattr(
        pipeline, "parse_document", stub_parse([{"page": 1, "text": "   "}])
    )

    result = run_indexing_job(
        make_request(),
        download=stub_download,
        encoder=FakeEncoder(),
        client=qdrant,
    )

    assert result is JobStatus.FAILED
    # 실패로 끝나도 기존 문서는 남아야 한다. upsert가 delete보다 먼저 거절한다.
    assert stored_points(qdrant) == before


class _RaisingEncoder:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls = 0

    def encode(
        self, texts: list[str], trace_id: str
    ) -> tuple[list[list[float]], list[dict[str, float]]]:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return [[0.0] * EMBEDDING_DIM for _ in texts], [{} for _ in texts]


def _stub_chunks(text: str):
    def chunk(_pages: list[dict[str, Any]]) -> list[Chunk]:
        return [Chunk(text, 1, None)]

    return chunk


def test_overlong_chunk_is_not_retryable(
    monkeypatch: pytest.MonkeyPatch,
    qdrant: QdrantClient,
    caplog: pytest.LogCaptureFixture,
) -> None:
    long_text = "가" * (MAX_TEXT_CHARS + 1)
    encoder = _RaisingEncoder()
    monkeypatch.setattr(
        pipeline, "parse_document", stub_parse([{"page": 1, "text": "본문"}])
    )
    monkeypatch.setattr(pipeline, "chunk_pages", _stub_chunks(long_text))
    reasons: list[str] = []

    with caplog.at_level(logging.ERROR, logger="zipsai.indexing.pipeline"):
        result = run_indexing_job(
            make_request(),
            download=stub_download,
            encoder=encoder,
            client=qdrant,
            on_fail=reasons.append,
        )

    assert result is JobStatus.FAILED
    assert reasons == ["CHUNK_TOO_LONG"]
    assert retryable_for(reasons[0]) is False
    assert encoder.calls == 0
    embed_logs = [
        record
        for record in caplog.records
        if record.getMessage() == "stage_done" and record.stage == "embed"
    ]
    assert embed_logs
    record = embed_logs[-1]
    assert record.error_code == "CHUNK_TOO_LONG"
    assert record.chunk_index == 0
    assert record.chunk_chars == len(long_text)
    assert record.max_text_chars == MAX_TEXT_CHARS
    assert long_text not in record.error
    assert long_text not in JsonFormatter().format(record)


def test_rejected_embed_is_not_retryable(
    monkeypatch: pytest.MonkeyPatch, qdrant: QdrantClient
) -> None:
    encoder = _RaisingEncoder(EmbeddingRequestInvalidError(422))
    monkeypatch.setattr(
        pipeline, "parse_document", stub_parse([{"page": 1, "text": "본문"}])
    )
    monkeypatch.setattr(pipeline, "chunk_pages", _stub_chunks("짧은 청크"))
    reasons: list[str] = []

    result = run_indexing_job(
        make_request(),
        download=stub_download,
        encoder=encoder,
        client=qdrant,
        on_fail=reasons.append,
    )

    assert result is JobStatus.FAILED
    assert reasons == ["EMBEDDING_REQUEST_INVALID"]
    assert retryable_for(reasons[0]) is False
    assert encoder.calls == 1


def test_embedding_outage_stays_retryable(
    monkeypatch: pytest.MonkeyPatch, qdrant: QdrantClient
) -> None:
    encoder = _RaisingEncoder(EmbeddingError("down"))
    monkeypatch.setattr(
        pipeline, "parse_document", stub_parse([{"page": 1, "text": "본문"}])
    )
    monkeypatch.setattr(pipeline, "chunk_pages", _stub_chunks("짧은 청크"))
    reasons: list[str] = []

    result = run_indexing_job(
        make_request(),
        download=stub_download,
        encoder=encoder,
        client=qdrant,
        on_fail=reasons.append,
    )

    assert result is JobStatus.FAILED
    assert reasons == ["EMBEDDING_FAILED"]
    assert retryable_for(reasons[0]) is True
    assert encoder.calls == 1
