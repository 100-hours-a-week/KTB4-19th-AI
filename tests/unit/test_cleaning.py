import copy
import logging

import pytest

from zipsai.indexing.clean import HOLD_RATIO, apply_cleaning, clean_pages


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


def test_apply_cleaning_falls_back_to_the_original_when_removal_is_excessive() -> None:
    # 너무 많이 지우면 정제를 포기한다. 잘린 본문이 색인되면 안 된다.
    original = "불필요\n- 3 -\n본문"
    assert clean_pages([{"page": 1, "text": original}]).removed_ratio > HOLD_RATIO

    result = apply_cleaning("doc-1", [{"page": 1, "text": original}])

    assert result.pages[0]["text"] == original
    assert result.removed_ratio == 0.0


def test_apply_cleaning_keeps_the_cleaned_pages_below_the_threshold() -> None:
    pages = [{"page": 1, "text": "본문입니다\n본문이 이어집니다\n- 3 -"}]
    cleaned = clean_pages(pages)
    assert 0 < cleaned.removed_ratio <= HOLD_RATIO

    result = apply_cleaning("doc-1", pages)

    assert result.pages[0]["text"] == cleaned.pages[0]["text"]


def test_logs_doc_id_and_ratio_without_body_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    body = "비밀본문"

    with caplog.at_level(logging.INFO, logger="zipsai.indexing.clean"):
        apply_cleaning("doc-abc", [{"page": 1, "text": body}])

    assert "cleaning doc_id=doc-abc removed_ratio=0.000" in caplog.text
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
