from pathlib import Path
from typing import Any

import pytest
from qdrant_client import QdrantClient

from zipsai.contracts.indexing import IndexingJobRequest, JobStatus
from zipsai.errors import DocumentFetchError
from zipsai.indexing import pipeline
from zipsai.indexing.pipeline import run_indexing_job
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

    result = run_indexing_job(
        make_request(),
        download=failing_download,
        encoder=FakeEncoder(),
        client=qdrant,
    )

    assert result is JobStatus.FAILED
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
