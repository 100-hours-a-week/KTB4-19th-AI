from itertools import pairwise

from zipsai.indexing.chunk import chunk_pages


def test_splits_two_articles_into_separate_small_chunks() -> None:
    pages = [
        {
            "page": 1,
            "text": "제1조 목적\n이 규정의 목적을 정한다.\n제2조 범위\n적용 범위를 정한다.",
        }
    ]

    chunks = chunk_pages(pages)

    assert [chunk.section for chunk in chunks] == ["제1조 목적", "제2조 범위"]
    assert all(len(chunk.text) <= 400 for chunk in chunks)
    assert all(chunk.text.startswith(chunk.section) for chunk in chunks)


def test_overlong_article_uses_fifty_character_overlap() -> None:
    text = "제1조 내용\n" + " ".join(f"문장{i}." for i in range(100))

    chunks = chunk_pages([{"page": 1, "text": text}])

    assert len(chunks) > 1
    assert all(len(chunk.text) <= 400 for chunk in chunks)
    assert all(
        current.text[:50] == previous.text[-50:]
        for previous, current in pairwise(chunks)
    )
    assert {chunk.section for chunk in chunks} == {"제1조 내용"}


def test_prefers_hang_boundaries_when_splitting_a_long_article() -> None:
    pages = [
        {
            "page": 1,
            "text": ("제1조 항목\n" + "① " + "가" * 340 + "\n제2항 " + "나" * 340),
        }
    ]

    chunks = chunk_pages(pages)

    assert len(chunks) >= 2
    assert chunks[1].text[50:].startswith("제2항")


def test_does_not_merge_adjacent_markdown_sections() -> None:
    pages = [{"page": 1, "text": "## 첫째\n짧은 내용\n## 둘째\n다른 내용"}]

    chunks = chunk_pages(pages)

    assert [chunk.section for chunk in chunks] == ["첫째", "둘째"]
    assert len(chunks) == 2


def test_keeps_short_unstructured_document_without_section() -> None:
    chunks = chunk_pages([{"page": 1, "text": "구조 없는 짧은 문서"}])

    assert [(chunk.text, chunk.section) for chunk in chunks] == [
        ("구조 없는 짧은 문서", None)
    ]


def test_splits_long_unstructured_document_on_sentences_with_overlap() -> None:
    text = " ".join(f"문장{i}." for i in range(100))

    chunks = chunk_pages([{"page": 1, "text": text}])

    assert len(chunks) > 1
    assert {chunk.section for chunk in chunks} == {None}
    assert all(len(chunk.text) <= 400 for chunk in chunks)
    assert all(
        current.text[:50] == previous.text[-50:]
        for previous, current in pairwise(chunks)
    )


def test_keeps_consecutive_table_lines_atomic_even_when_oversized() -> None:
    table = "\n".join(f"| {i} | {'값' * 30} |" for i in range(20))

    chunks = chunk_pages([{"page": 1, "text": table}])

    assert len(chunks) == 1
    assert chunks[0].text == table
    assert len(chunks[0].text) > 400


def test_maps_content_on_second_page_and_keeps_placeholders() -> None:
    text = "302호 [이름]\n[전화번호]"
    chunks = chunk_pages([{"page": 1, "text": "   "}, {"page": 2, "text": text}])

    assert len(chunks) == 1
    assert chunks[0].page == 2
    assert text in chunks[0].text


def test_returns_no_chunks_for_empty_pages() -> None:
    assert chunk_pages([{"page": 1, "text": ""}, {"page": 2, "text": "  \n"}]) == []


def test_does_not_treat_article_mention_as_boundary() -> None:
    text = "제3조에 따라 이 절차를 따른다.\n계속한다."

    chunks = chunk_pages([{"page": 1, "text": text}])

    assert len(chunks) == 1
    assert chunks[0].section is None
    assert chunks[0].text == text
