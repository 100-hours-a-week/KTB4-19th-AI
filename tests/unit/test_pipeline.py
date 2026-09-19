from pathlib import Path
from typing import Any

import pytest
from qdrant_client import QdrantClient

from zipsai.contracts.indexing import IndexingJobRequest, JobStatus, SourceType
from zipsai.errors import DocumentFetchError
from zipsai.indexing import pipeline
from zipsai.indexing.pipeline import run_indexing_job
from zipsai.indexing.store import InMemoryJobStore
from zipsai.integrations.qdrant import create_client, ensure_collection
from zipsai.settings import EMBEDDING_DIM, QDRANT_COLLECTION

FIXTURE = Path(__file__).parents[1] / "fixtures" / "two-pages.pdf"


class FakeEncoder:
    def encode(
        self, texts: list[str]
    ) -> tuple[list[list[float]], list[dict[str, float]]]:
        return [[0.0] * EMBEDDING_DIM for _ in texts], [{} for _ in texts]


def make_request() -> IndexingJobRequest:
    return IndexingJobRequest(
        building_id=101,
        doc_id="notice-001",
        source_type=SourceType.NOTICE,
        title="물탱크 청소 안내",
        published_at="2026-09-15T00:00:00Z",
        file_key="buildings/101/notice-001.pdf",
        trace_id="trace-001",
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


@pytest.fixture
def job() -> tuple[InMemoryJobStore, str]:
    store = InMemoryJobStore()
    return store, store.create()


def test_success_path_upserts_chunks_and_marks_succeeded(
    qdrant: QdrantClient, job: tuple[InMemoryJobStore, str]
) -> None:
    store, job_id = job

    result = run_indexing_job(
        store,
        job_id,
        make_request(),
        download=copy_fixture,
        encoder=FakeEncoder(),
        client=qdrant,
    )

    assert result is JobStatus.SUCCEEDED
    assert store.get_status(job_id) is JobStatus.SUCCEEDED
    assert stored_points(qdrant) > 0


def test_pii_detection_holds_job_without_storing_vectors(
    monkeypatch: pytest.MonkeyPatch,
    qdrant: QdrantClient,
    job: tuple[InMemoryJobStore, str],
) -> None:
    store, job_id = job
    monkeypatch.setattr(
        pipeline,
        "parse_pdf",
        stub_parse([{"page": 1, "text": "문의는 admin@example.com 으로 주세요."}]),
    )

    result = run_indexing_job(
        store,
        job_id,
        make_request(),
        download=stub_download,
        encoder=FakeEncoder(),
        client=qdrant,
    )

    assert result is JobStatus.NEEDS_REVIEW
    assert store.get_status(job_id) is JobStatus.NEEDS_REVIEW
    assert stored_points(qdrant) == 0


def test_excessive_cleaning_holds_job_without_storing_vectors(
    monkeypatch: pytest.MonkeyPatch,
    qdrant: QdrantClient,
    job: tuple[InMemoryJobStore, str],
) -> None:
    store, job_id = job
    repeated = "공동주택 관리사무소 알림 머리글"
    monkeypatch.setattr(
        pipeline,
        "parse_pdf",
        stub_parse(
            [{"page": number, "text": f"{repeated}\n본문{number}"} for number in (1, 2, 3)]
        ),
    )

    result = run_indexing_job(
        store,
        job_id,
        make_request(),
        download=stub_download,
        encoder=FakeEncoder(),
        client=qdrant,
    )

    assert result is JobStatus.NEEDS_REVIEW
    assert store.get_status(job_id) is JobStatus.NEEDS_REVIEW
    assert stored_points(qdrant) == 0


def test_download_failure_marks_job_failed(
    qdrant: QdrantClient, job: tuple[InMemoryJobStore, str]
) -> None:
    store, job_id = job

    def failing_download(file_key: str, _: Path) -> Path:
        raise DocumentFetchError(f"Failed to fetch '{file_key}'")

    result = run_indexing_job(
        store,
        job_id,
        make_request(),
        download=failing_download,
        encoder=FakeEncoder(),
        client=qdrant,
    )

    assert result is JobStatus.FAILED
    assert store.get_status(job_id) is JobStatus.FAILED
    assert stored_points(qdrant) == 0
