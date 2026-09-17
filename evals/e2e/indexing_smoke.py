"""서버 Qdrant 대상 인덱싱 e2e 스모크.

파싱 → 정제 → 마스킹 → 청킹 → BGE-M3 실인코딩 → 적재 → 격리·교체·인덱스 확인.
전용 컬렉션(documents_e2e)을 만들고 끝나면 지운다. 사전 조건: localhost:6333 Qdrant.
"""

from datetime import UTC, datetime
from pathlib import Path

from qdrant_client import models

from zipsai.contracts.indexing import IndexingJobRequest, SourceType
from zipsai.indexing.chunk import chunk_pages
from zipsai.indexing.clean import clean_pages
from zipsai.indexing.embed import embed_chunks
from zipsai.indexing.mask import mask_pages
from zipsai.indexing.parse import parse_pdf
from zipsai.indexing.upsert import upsert_document
from zipsai.integrations.bge_m3 import BgeM3Encoder
from zipsai.integrations.qdrant import create_client, ensure_collection

QDRANT_URL = "http://localhost:6333"
COLLECTION = "documents_e2e"
FIXTURE = Path(__file__).parents[2] / "tests" / "fixtures" / "two-pages.pdf"


def request() -> IndexingJobRequest:
    return IndexingJobRequest(
        building_id=101,
        doc_id="doc-e2e",
        source_type=SourceType.RULE,
        title="e2e 스모크",
        published_at=datetime(2026, 9, 17, tzinfo=UTC),
        file_key="documents/e2e.pdf",
        trace_id="trace-e2e",
    )


def count_building(client, building_id: int) -> int:
    return client.count(
        collection_name=COLLECTION,
        count_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="building_id",
                    match=models.MatchValue(value=str(building_id)),
                )
            ]
        ),
        exact=True,
    ).count


def main() -> None:
    pages = parse_pdf(FIXTURE)
    cleaned = clean_pages(pages)
    masked = mask_pages(cleaned.pages)
    assert not masked.detections, f"예상 밖 PII 탐지: {masked.detections}"
    chunks = chunk_pages(masked.pages)
    embedded = embed_chunks(chunks, BgeM3Encoder())

    client = create_client(QDRANT_URL)
    if client.collection_exists(collection_name=COLLECTION):
        client.delete_collection(collection_name=COLLECTION)
    ensure_collection(client, COLLECTION)

    stored = upsert_document(client, request(), embedded, collection=COLLECTION)
    assert stored > 0, stored
    print(f"PASS 적재: {stored} chunks (pages={len(pages)}, chunks={len(chunks)})")

    assert count_building(client, 999) == 0
    print("PASS 격리: building_id=999 필터 조회 0건")

    replaced = upsert_document(client, request(), embedded[:1], collection=COLLECTION)
    total = client.count(collection_name=COLLECTION, exact=True).count
    assert replaced == 1 and total == 1, (replaced, total)
    print("PASS 교체: 같은 doc_id 재업로드 후 총 1건 — 옛 청크 없음")

    schema = client.get_collection(COLLECTION).payload_schema
    assert "doc_id" in schema and "building_id" in schema, schema
    print(f"PASS 인덱스: payload_schema={sorted(schema)}")

    client.delete_collection(collection_name=COLLECTION)
    print("PASS 정리: documents_e2e 삭제 완료")


if __name__ == "__main__":
    main()
