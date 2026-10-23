from itertools import pairwise
from pathlib import Path

import pytest

from zipsai.indexing.chunk import TABLE_CHUNK_CHARS, TABLE_CHUNK_FORMAT, chunk_pages

HEADER = "| 항목 | 구분 | 요금 | 소요 시간 | 비고 |"
SEPARATOR = "| --- | --- | --- | --- | --- |"


def _row(page: int) -> str:
    return f"| p{page} | 구분 | 1원 | 1분 | {'가' * 200} |"


def _pages(count: int) -> list[dict[str, int | str]]:
    pages = []
    for page in range(1, count + 1):
        body = _row(page)
        if page == 1:
            text = f"합성 표\n{HEADER}\n{SEPARATOR}\n{body}"
        else:
            text = f"{HEADER}\n{SEPARATOR}\n{body}"
        pages.append({"page": page, "text": text})
    return pages


def _table_pieces(chunks: list) -> list:
    return [chunk for chunk in chunks if chunk.text.startswith("합성 표\n항목:")]


def test_joins_nine_and_twenty_one_page_tables_and_repeats_the_header() -> None:
    for count in (9, 21):
        chunks = chunk_pages(_pages(count))
        pieces = _table_pieces(chunks)
        assert pieces
        assert all(chunk.text.startswith("합성 표\n") for chunk in pieces)
        assert all(chunk.text.count("항목:") == 1 for chunk in pieces)
        for page in range(1, count + 1):
            owners = [chunk for chunk in pieces if f"항목: p{page} /" in chunk.text]
            assert len(owners) == 1
            assert owners[0].page == page
        assert all(
            not current.text.startswith(previous.text[-50:])
            for previous, current in pairwise(pieces)
        )


def test_default_format_names_cells_without_pipes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TABLE_CHUNK_FORMAT", raising=False)
    table = "| 항목 | 요금 |\n| --- | --- |\n| 세탁 | 1000 |\n| 건조 | 2000 |"

    chunks = chunk_pages([{"page": 1, "text": table}])

    assert TABLE_CHUNK_FORMAT == "fields"
    assert chunks
    assert all("|" not in chunk.text for chunk in chunks)
    assert any("항목: 세탁" in chunk.text and "요금: 1000" in chunk.text for chunk in chunks)


def test_repeats_the_section_on_each_table_piece() -> None:
    body = "\n".join(f"| 세탁{index} | {index}000 |" for index in range(30))
    text = f"## 세탁실\n| 항목 | 요금 |\n| --- | --- |\n{body}"

    chunks = chunk_pages([{"page": 1, "text": text}])

    pieces = [chunk for chunk in chunks if "세탁0" in chunk.text or "요금" in chunk.text]
    assert pieces
    assert all(chunk.section == "세탁실" for chunk in chunks)
    assert all(chunk.text.startswith("세탁실\n") for chunk in pieces)


def test_keeps_one_oversized_row_and_does_not_invent_a_header() -> None:
    long_row = "| " + ("가" * 500) + " |"
    table = long_row + "\n| 짧은 |"

    chunks = chunk_pages([{"page": 1, "text": table}])

    assert len(chunks) == 2
    assert chunks[0].text == long_row
    assert chunks[1].text == "| 짧은 |"
    assert len(chunks[1].text) <= TABLE_CHUNK_CHARS


def test_keeps_a_short_markdown_table_intact(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TABLE_CHUNK_FORMAT", "markdown")
    table = "| 항목 | 요금 |\n| --- | --- |\n| 세탁 | 1000 |"

    chunks = chunk_pages([{"page": 1, "text": table}])

    assert [(chunk.text, chunk.section) for chunk in chunks] == [(table, None)]


def test_joins_headers_that_differ_only_by_dash_and_spaces() -> None:
    pages = [
        {"page": 1, "text": "| 비고 — |\n| --- |\n| 가 |"},
        {"page": 2, "text": "| 비고 - |\n| --- |\n| 비고 - |\n| 나 |"},
    ]

    chunks = chunk_pages(pages)
    text = "\n".join(chunk.text for chunk in chunks)

    assert text.count("비고 —: 가") == 1
    assert text.count("비고 —: 나") == 1
    assert "비고 -" not in text


def test_packed_table_stays_within_512_bge_m3_tokens() -> None:
    source = Path(chunk_pages.__code__.co_filename).read_text(encoding="utf-8")
    assert "transformers" not in source
    assert "tokenizer" not in source

    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        pytest.fail(f"BGE-M3 tokenizer dependency missing: {error}")
    try:
        tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
    except (OSError, ValueError) as error:
        pytest.fail(f"BGE-M3 tokenizer is not available: {error}")

    rows = "\n".join(
        f"| 세탁기 {index}호기 | 표준 | 2,500원 | 45분 | 최대 8kg |" for index in range(40)
    )
    chunks = chunk_pages([{"page": 1, "text": f"{HEADER}\n{SEPARATOR}\n{rows}"}])

    assert len(chunks) > 1
    for chunk in chunks:
        tokens = tokenizer.encode(chunk.text, add_special_tokens=True)
        assert len(tokens) <= 512
