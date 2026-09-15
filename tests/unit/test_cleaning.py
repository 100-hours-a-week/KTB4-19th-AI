import copy
import logging

import pytest

from zipsai.contracts.indexing import JobStatus
from zipsai.errors import JobNotFoundError
from zipsai.indexing.clean import HOLD_RATIO, apply_cleaning, clean_pages
from zipsai.indexing.store import InMemoryJobStore


def test_drops_repeated_line_seen_on_three_pages() -> None:
    pages = [
        {"page": 1, "text": "건물 관리규정\n본문 1"},
        {"page": 2, "text": "건물 관리규정\n본문 2"},
        {"page": 3, "text": "건물 관리규정\n본문 3"},
    ]

    result = clean_pages(pages)

    assert [page["text"] for page in result.pages] == ["본문 1", "본문 2", "본문 3"]


def test_keeps_repeated_line_seen_on_two_pages() -> None:
    pages = [
        {"page": 1, "text": "건물 관리규정\n본문 1"},
        {"page": 2, "text": "건물 관리규정\n본문 2"},
    ]

    result = clean_pages(pages)

    assert all("건물 관리규정" in page["text"] for page in result.pages)


def test_drops_standalone_page_numbers() -> None:
    pages = [
        {
            "page": 1,
            "text": "제목\n- 3 -\n3 / 5\n페이지 3\n42\n본문",
        }
    ]

    result = clean_pages(pages)

    assert result.pages[0]["text"] == "제목\n본문"


def test_drops_table_of_contents_block_until_pattern_breaks() -> None:
    pages = [
        {
            "page": 1,
            "text": "목차\n제1장 총칙 ........ 1\n제2장 운영 2\n본문 시작",
        }
    ]

    result = clean_pages(pages)

    assert result.pages[0]["text"] == "본문 시작"


def test_never_drops_structural_lines_even_when_repeated() -> None:
    pages = [
        {
            "page": page,
            "text": "공통 헤더\n# 제목\n제1조 목적\n① 항목\n1. 목록\n제1항 내용",
        }
        for page in (1, 2, 3)
    ]

    result = clean_pages(pages)

    for page in result.pages:
        assert "공통 헤더" not in page["text"]
        assert "# 제목" in page["text"]
        assert "제1조 목적" in page["text"]
        assert "① 항목" in page["text"]
        assert "1. 목록" in page["text"]
        assert "제1항 내용" in page["text"]


def test_keeps_article_boundary_when_repeated() -> None:
    pages = [{"page": page, "text": "제1조 목적\n공통 헤더"} for page in (1, 2, 3)]

    result = clean_pages(pages)

    assert all(page["text"] == "제1조 목적" for page in result.pages)


def test_does_not_mutate_pages_and_passes_extra_keys_through() -> None:
    pages = [{"page": 1, "text": "- 3 -\n본문", "source": "scan"}]
    original = copy.deepcopy(pages)

    result = clean_pages(pages)

    assert pages == original
    assert result.pages == [{"page": 1, "text": "본문", "source": "scan"}]


def test_whitespace_normalization_is_not_counted_as_removal() -> None:
    result = clean_pages([{"page": 1, "text": "본문\n\n\n끝   "}])

    assert result.pages[0]["text"] == "본문\n\n끝"
    assert result.removed_ratio == 0.0


def test_apply_cleaning_holds_job_when_removal_exceeds_threshold() -> None:
    store = InMemoryJobStore()
    job_id = store.create()

    result = apply_cleaning(store, job_id, [{"page": 1, "text": "불필요\n- 3 -\n본문"}])

    assert result.removed_ratio > HOLD_RATIO
    assert store.get_status(job_id) is JobStatus.NEEDS_REVIEW


def test_apply_cleaning_unknown_job_raises_job_not_found() -> None:
    with pytest.raises(JobNotFoundError):
        apply_cleaning(InMemoryJobStore(), "missing", [{"page": 1, "text": "- 3 -"}])


def test_logs_job_id_and_ratio_without_body_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = InMemoryJobStore()
    job_id = store.create()
    body = "비밀본문"

    with caplog.at_level(logging.INFO, logger="zipsai.indexing.clean"):
        apply_cleaning(store, job_id, [{"page": 1, "text": body}])

    assert f"cleaning job_id={job_id} removed_ratio=0.000" in caplog.text
    assert body not in caplog.text


def test_preserves_dates_and_mask_placeholders() -> None:
    text = "2026년 9월 20일\n302호 [이름]\n[전화번호]"

    result = clean_pages([{"page": 1, "text": text}])

    assert result.pages[0]["text"] == text
    assert result.removed_ratio == 0.0


def test_preserves_dates_and_mask_placeholders_when_repeated() -> None:
    pages = [
        {
            "page": page,
            "text": "2026년 9월 20일\n302호 [이름]\n[전화번호]",
        }
        for page in (1, 2, 3)
    ]

    result = clean_pages(pages)

    assert [page["text"] for page in result.pages] == [page["text"] for page in pages]
