from datetime import datetime, timedelta, timezone

import pytest
from qdrant_client import QdrantClient

from zipsai.contracts.comment import CommentDeleteRequest, CommentRequest
from zipsai.contracts.notification import ResultStatus
from zipsai.indexing.comment import (
    comment_point_id,
    delete_comments,
    upsert_comments,
)
from zipsai.indexing.upsert import delete_missing_documents
from zipsai.integrations.qdrant import ensure_collection
from zipsai.settings import EMBEDDING_DIM, QDRANT_COLLECTION

KST = timezone(timedelta(hours=9))
T0 = datetime(2026, 10, 15, 0, 0, 12, tzinfo=KST)


class FakeEncoder:
    def __init__(self, fail: bool = False):
        self.fail = fail
        self.calls = 0

    def encode(self, texts, trace_id):
        self.calls += 1
        if self.fail:
            raise RuntimeError("down")
        return [[0.5] * EMBEDDING_DIM for _ in texts], [{"7": 0.5} for _ in texts]


@pytest.fixture
def client() -> QdrantClient:
    qdrant = QdrantClient(":memory:")
    ensure_collection(qdrant)
    return qdrant


def item(
    detail_id=9001,
    updated_at=T0,
    comment="매월 둘째 주 화요일 10시에 점검합니다.",
    **kw,
):
    base = {
        "complaint_detail_id": detail_id,
        "building_id": 101,
        "complaint_type": "QA",
        "symptom": "엘리베이터 정기 점검은 언제 하나요",
        "comment": comment,
        "updated_at": updated_at.isoformat(),
        "location": None,
        "issue_type": None,
    }
    return {**base, **kw}


def request(*items):
    return CommentRequest.model_validate({"trace_id": "t", "items": list(items)})


def stored(client, detail_id=9001):
    found = client.retrieve(
        QDRANT_COLLECTION, [comment_point_id(101, detail_id)], with_payload=True
    )
    return found[0].payload if found else None


def delete(client, detail_id=9001, updated_at=T0, building_id=101):
    body = {
        "trace_id": "t",
        "items": [
            {
                "building_id": building_id,
                "complaint_detail_id": detail_id,
                "updated_at": updated_at.isoformat(),
            }
        ],
    }
    return delete_comments(CommentDeleteRequest.model_validate(body), client=client)


def is_tombstone(payload):
    return (
        payload is not None and payload.get("deleted") is True and "text" not in payload
    )


def search(client, question="엘리베이터 정기 점검은 언제 하나요", building_id=101):
    from zipsai.knowledge.retrieve import encode_question, search_chunks

    vector = encode_question(question, encoder=FakeEncoder(), trace_id="t")
    return search_chunks(vector, building_id, client=client)


def run(client, *items, encoder=None):
    return upsert_comments(
        request(*items), encoder=encoder or FakeEncoder(), client=client
    )


def test_new_comment_is_stored_as_qa_history(client):
    [result] = run(client, item(location="미상"))

    payload = stored(client)
    assert result.status is ResultStatus.SUCCEEDED
    assert payload["source_type"] == "qa_history"
    assert payload["title"] == "관리자 답변"
    assert payload["location"] is None
    assert "엘리베이터 정기 점검은 언제 하나요" in payload["text"]
    assert "미상" not in payload["text"]


def test_same_updated_at_is_already_applied(client):
    run(client, item())
    encoder = FakeEncoder()

    [result] = run(client, item(), encoder=encoder)

    assert result.status is ResultStatus.SUCCEEDED
    assert encoder.calls == 0


def test_older_updated_at_is_stale(client):
    run(client, item(comment="새로운 안내 문구입니다 열 글자 이상"))

    [result] = run(
        client,
        item(
            updated_at=T0 - timedelta(days=1),
            comment="옛날 안내 문구입니다 열 글자 이상",
        ),
    )

    assert (result.status, result.reason_code) == (ResultStatus.SKIPPED, "STALE_UPDATE")
    assert "새로운" in stored(client)["text"]


def test_newer_updated_at_replaces(client):
    run(client, item())

    run(
        client,
        item(
            updated_at=T0 + timedelta(days=1),
            comment="이제는 매주 월요일에 점검합니다.",
        ),
    )

    assert "매주 월요일" in stored(client)["text"]
    assert client.count(QDRANT_COLLECTION).count == 1


def test_newer_but_too_short_removes_old_comment(client):
    run(client, item())

    [result] = run(client, item(updated_at=T0 + timedelta(days=1), comment="완료"))

    assert (result.status, result.reason_code) == (
        ResultStatus.SKIPPED,
        "COMMENT_TOO_SHORT",
    )
    assert is_tombstone(stored(client))


def test_failure_is_isolated_per_item(client):
    results = run(client, item(9001), item(9002), encoder=FakeEncoder(fail=True))

    assert [(r.status, r.reason_code) for r in results] == [
        (ResultStatus.FAILED, "EMBEDDING_FAILED")
    ] * 2


def test_comment_is_masked_before_embedding(client):
    run(client, item(comment="전화는 010-1234-5678 로 주세요. 점검은 화요일입니다."))

    payload = stored(client)
    assert "010-1234-5678" not in payload["text"]
    assert payload["masked"] is True


def test_delete_is_idempotent(client):
    run(client, item())
    later = T0 + timedelta(minutes=5)

    first = delete(client, updated_at=later)
    second = delete(client, updated_at=later)

    assert is_tombstone(stored(client))
    assert [r.status for r in first + second] == [ResultStatus.SUCCEEDED] * 2
    assert first[0].updated_at == later.isoformat()


def test_late_retry_after_short_comment_does_not_resurrect(client):
    # 10:00 v1이 실패해 재시도 목록에 들어가고, 10:05 v2("확인중")가 먼저 반영된 뒤
    # 10:06에 v1 재시도가 도착하는 경우. 툼스톤의 10:05가 v1을 걸러야 한다.
    v1 = item(comment="보일러 점검 완료했습니다. 다시 확인해주세요")
    run(client, item(updated_at=T0 + timedelta(minutes=5), comment="확인중"))

    [result] = run(client, v1)

    assert (result.status, result.reason_code) == (ResultStatus.SKIPPED, "STALE_UPDATE")
    assert is_tombstone(stored(client))
    assert search(client) == []


def test_late_retry_after_delete_does_not_resurrect(client):
    v1 = item(comment="보일러 점검 완료했습니다. 다시 확인해주세요")
    run(client, v1)
    delete(client, updated_at=T0 + timedelta(minutes=5))

    [result] = run(client, v1)

    assert (result.status, result.reason_code) == (ResultStatus.SKIPPED, "STALE_UPDATE")
    assert is_tombstone(stored(client))


def test_delete_arriving_after_a_newer_answer_is_ignored(client):
    run(client, item(updated_at=T0 + timedelta(minutes=10)))

    [result] = delete(client, updated_at=T0 + timedelta(minutes=5))

    assert (result.status, result.reason_code) == (ResultStatus.SKIPPED, "STALE_UPDATE")
    assert not is_tombstone(stored(client))


def test_newer_answer_replaces_a_tombstone(client):
    delete(client, updated_at=T0)

    [result] = run(client, item(updated_at=T0 + timedelta(minutes=5)))

    assert result.status is ResultStatus.SUCCEEDED
    assert "deleted" not in stored(client)
    assert "둘째 주 화요일" in stored(client)["text"]
    assert [p.payload["complaint_detail_id"] for p in search(client)] == [9001]


def test_tombstone_is_never_searched(client):
    run(client, item())
    run(client, item(9002, updated_at=T0 + timedelta(minutes=1)))
    delete(client, 9002, updated_at=T0 + timedelta(minutes=5))

    assert [p.payload["complaint_detail_id"] for p in search(client)] == [9001]


def test_complaint_answer_is_stored_but_hidden_from_residents(client):
    run(client, item(9100, complaint_type="COMPLAINT", location="302호 화장실"))
    run(client, item(9101))

    assert stored(client, 9100)["complaint_type"] == "COMPLAINT"
    assert [p.payload["complaint_detail_id"] for p in search(client)] == [9101]


def test_only_complaint_answers_means_no_evidence(client):
    run(client, item(9100, complaint_type="COMPLAINT"))

    assert search(client) == []


def test_document_cleanup_removes_documents_but_keeps_comments(client):
    from zipsai.contracts.indexing import IndexingJobRequest
    from zipsai.indexing.embed import EmbeddedChunk
    from zipsai.indexing.upsert import upsert_document

    run(client, item())
    document = IndexingJobRequest(
        building_id=101, trace_id="t", doc_id="doc-1", title="공지", file_key="k"
    )
    chunk = EmbeddedChunk("본문", 1, None, [0.5] * EMBEDDING_DIM, {"7": 0.5})
    upsert_document(client, document, [chunk])
    assert client.count(QDRANT_COLLECTION).count == 2

    delete_missing_documents(client, 101, [])

    assert client.count(QDRANT_COLLECTION).count == 1
    assert stored(client) is not None


def test_result_echoes_the_request_updated_at_string(client):
    raw = "2026-10-15T00:00:12.123+09:00"

    [result] = run(
        client, item(updated_at=T0.replace(microsecond=123000)) | {"updated_at": raw}
    )

    assert result.updated_at == raw


def test_location_is_masked_in_the_payload(client):
    run(client, item(location="1203호 김민수 앞 복도"))

    assert "김민수" not in stored(client)["location"]


def test_comment_becomes_qa_history_citation():
    from qdrant_client import models

    from zipsai.knowledge.node import _citations

    def point(payload):
        return models.ScoredPoint(id="x", version=1, score=0.9, payload=payload)

    comment = {
        "source_type": "qa_history",
        "complaint_detail_id": 9002,
        "title": "관리자 답변",
        "text": "302호인데 엘리베이터 점검 언제예요\n매월 둘째 주 화요일에 점검합니다.",
        "answer": "매월 둘째 주 화요일에 점검합니다.",
    }

    citations = _citations(
        [point(comment), point({**comment, "complaint_detail_id": 9003})]
    )

    assert [(c.source_type, c.source_id) for c in citations] == [
        ("qa_history", "9002"),
        ("qa_history", "9003"),
    ]
    # 입주민이 쓴 질문은 다른 입주민 화면에 나가지 않는다.
    assert citations[0].snippet == "매월 둘째 주 화요일에 점검합니다."


def test_stored_payload_keeps_the_masked_answer_separately(client):
    run(client, item(comment="010-1234-5678로 연락주세요. 점검은 화요일입니다."))

    payload = stored(client)
    assert payload["answer"].startswith("[")
    assert "엘리베이터" not in payload["answer"]


def test_stored_comment_is_found_by_the_same_question(client):
    from zipsai.knowledge.retrieve import encode_question, search_chunks

    run(client, item())
    run(client, item(9002, building_id=202))

    vector = encode_question(
        "엘리베이터 정기 점검은 언제 하나요", encoder=FakeEncoder(), trace_id="t"
    )
    found = search_chunks(vector, 101, client=client)

    assert [p.payload["complaint_detail_id"] for p in found] == [9001]


def test_embeddings_are_sent_in_batches(client):
    from math import ceil

    from zipsai.integrations.embedding_client import BATCH_SIZE

    encoder = FakeEncoder()
    stats: dict = {}
    items = [item(detail_id=9000 + index) for index in range(100)]

    results = upsert_comments(
        request(*items), encoder=encoder, client=client, stats=stats
    )

    assert encoder.calls == ceil(100 / BATCH_SIZE)
    assert stats["embed_batches"] == ceil(100 / BATCH_SIZE)
    assert stats["embed_ms"] >= 0
    assert len(stats["item_ms"]) == 100
    assert [result.status for result in results] == [ResultStatus.SUCCEEDED] * 100


def test_one_embed_batch_failure_leaves_the_others(client):
    class FlakyEncoder(FakeEncoder):
        def encode(self, texts, trace_id):
            if self.calls == 1:
                self.calls += 1
                raise RuntimeError("down")
            return super().encode(texts, trace_id)

    results = run(
        client,
        *[item(detail_id=9001 + index) for index in range(6)],
        encoder=FlakyEncoder(),
    )

    assert [(result.status, result.reason_code) for result in results] == [
        (ResultStatus.SUCCEEDED, None),
        (ResultStatus.SUCCEEDED, None),
        (ResultStatus.FAILED, "EMBEDDING_FAILED"),
        (ResultStatus.FAILED, "EMBEDDING_FAILED"),
        (ResultStatus.SUCCEEDED, None),
        (ResultStatus.SUCCEEDED, None),
    ]
    assert stored(client, 9001) is not None
    assert stored(client, 9003) is None
    assert stored(client, 9005) is not None


def test_one_upsert_batch_failure_leaves_the_others(client):
    original = client.upsert
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("qdrant")
        return original(*args, **kwargs)

    client.upsert = flaky

    results = run(client, *[item(detail_id=9001 + index) for index in range(6)])

    assert [(result.status, result.reason_code) for result in results] == [
        (ResultStatus.SUCCEEDED, None),
        (ResultStatus.SUCCEEDED, None),
        (ResultStatus.FAILED, "VECTOR_STORE_FAILED"),
        (ResultStatus.FAILED, "VECTOR_STORE_FAILED"),
        (ResultStatus.SUCCEEDED, None),
        (ResultStatus.SUCCEEDED, None),
    ]


def test_short_comment_is_left_out_of_the_embed_batch(client):
    encoder = FakeEncoder()

    results = run(
        client,
        item(9001),
        item(9002, comment="짧음"),
        item(9003),
        encoder=encoder,
    )

    assert encoder.calls == 1
    assert [(result.status, result.reason_code) for result in results] == [
        (ResultStatus.SUCCEEDED, None),
        (ResultStatus.SKIPPED, "COMMENT_TOO_SHORT"),
        (ResultStatus.SUCCEEDED, None),
    ]


def test_mask_failure_is_skipped(client, monkeypatch):
    def boom(_pages):
        raise RuntimeError("mask")

    monkeypatch.setattr("zipsai.indexing.comment.apply_masking", boom)

    [result] = run(client, item())

    assert (result.status, result.reason_code) == (ResultStatus.SKIPPED, "MASK_FAILED")


def test_delete_stats_have_no_embed(client):
    stats: dict = {}
    body = {
        "trace_id": "t",
        "items": [
            {
                "building_id": 101,
                "complaint_detail_id": 9001,
                "updated_at": T0.isoformat(),
            }
        ],
    }

    delete_comments(
        CommentDeleteRequest.model_validate(body), client=client, stats=stats
    )

    assert stats["embed_batches"] == 0
    assert stats["embed_ms"] == 0
    assert len(stats["item_ms"]) == 1
