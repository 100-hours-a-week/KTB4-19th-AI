import logging

import pytest

from zipsai.contracts.indexing import JobStatus
from zipsai.errors import JobNotFoundError
from zipsai.indexing.mask import PiiKind, apply_masking, mask_pages
from zipsai.indexing.store import InMemoryJobStore


@pytest.mark.parametrize(
    ("text", "kind", "expected"),
    [
        ("010-6183-4275", PiiKind.PHONE, "[전화번호]"),
        ("02-538-1234", PiiKind.PHONE, "[전화번호]"),
        ("900101-1234567", PiiKind.RESIDENT_ID, "[주민등록번호]"),
        ("123-4567-8901", PiiKind.ACCOUNT, "[계좌번호]"),
        ("12가3456", PiiKind.CAR_PLATE, "[차량번호]"),
        ("56라9012", PiiKind.CAR_PLATE, "[차량번호]"),
        ("person@example.com", PiiKind.EMAIL, "[이메일]"),
        ("302호 김미정님", PiiKind.UNIT_NAME, "302호 [이름]"),
        ("302호\n김미정", PiiKind.UNIT_NAME, "302호\n[이름]"),
    ],
)
def test_masks_supported_pii(text: str, kind: PiiKind, expected: str) -> None:
    result = mask_pages([{"page": 1, "text": text}])

    assert result.pages == [{"page": 1, "text": expected}]
    assert len(result.detections) == 1
    assert result.detections[0].kind is kind
    assert result.detections[0].matched in text


@pytest.mark.parametrize(
    "text",
    [
        "제2026-4호",
        "제3조",
        "101-506호",
        "500원",
        "15,000원",
        "2026-09-20",
        "2026년 9월 20일",
        "30호실에 12면",
        "123456-789012-345",
        "101호 관리실",
        "203호 입주민",
        "301호 내지 305호",
    ],
)
def test_leaves_non_pii_text_unchanged(text: str) -> None:
    pages = [{"page": 1, "text": text}]

    result = mask_pages(pages)

    assert result.pages == pages
    assert result.detections == []


def test_replaces_multiple_matches_right_to_left_and_sorts_detections() -> None:
    pages = [{"page": 2, "text": "010-4821-7733 alice@example.com"}]

    result = mask_pages(pages)

    assert result.pages == [{"page": 2, "text": "[전화번호] [이메일]"}]
    assert [d.kind for d in result.detections] == [
        PiiKind.PHONE,
        PiiKind.EMAIL,
    ]
    assert [d.start for d in result.detections] == [0, 14]


def test_phone_is_not_also_detected_as_account() -> None:
    result = mask_pages([{"page": 1, "text": "010-4821-7733"}])

    assert [d.kind for d in result.detections] == [PiiKind.PHONE]


def test_does_not_mutate_input_pages() -> None:
    pages = [{"page": 1, "text": "010-6183-4275", "source": "index"}]

    result = mask_pages(pages)

    assert pages == [{"page": 1, "text": "010-6183-4275", "source": "index"}]
    assert result.pages == [{"page": 1, "text": "[전화번호]", "source": "index"}]


def test_clean_pages_keep_accepted_status() -> None:
    store = InMemoryJobStore()
    job_id = store.create()

    result = apply_masking(store, job_id, [{"page": 1, "text": "공지사항"}])

    assert result.detections == []
    assert store.get_status(job_id) is JobStatus.ACCEPTED


def test_pii_on_page_two_holds_job_for_review() -> None:
    store = InMemoryJobStore()
    job_id = store.create()

    result = apply_masking(
        store,
        job_id,
        [{"page": 1, "text": "공지사항"}, {"page": 2, "text": "010-6183-4275"}],
    )

    assert result.detections[0].page == 2
    assert store.get_status(job_id) is JobStatus.NEEDS_REVIEW


def test_unknown_job_with_pii_raises_job_not_found() -> None:
    with pytest.raises(JobNotFoundError):
        apply_masking(
            InMemoryJobStore(), "missing", [{"page": 2, "text": "010-6183-4275"}]
        )


def test_logs_job_id_and_counts_without_matched_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = InMemoryJobStore()
    job_id = store.create()

    with caplog.at_level(logging.INFO, logger="zipsai.indexing.mask"):
        apply_masking(store, job_id, [{"page": 1, "text": "person@example.com"}])

    assert job_id in caplog.text
    assert "이메일" in caplog.text
    assert "1" in caplog.text
    assert "person@example.com" not in caplog.text
