from pathlib import Path
from types import SimpleNamespace

import pytest
from docling.datamodel.pipeline_options import TableFormerMode

from zipsai.indexing import parse
from zipsai.indexing.parse import (
    fast_pipeline_options,
    parse_document,
    pipeline_options,
)
from zipsai.indexing.pdf_text import (
    CID_RATIO_MAX,
    GARBAGE_RATIO_MAX,
    MIN_CHARS_PER_PAGE,
    PdfInspection,
    classify_text_layer,
    garbage_ratio,
    table_is_ragged,
)


def test_thresholds_match_the_locked_cuts() -> None:
    assert GARBAGE_RATIO_MAX == 0.2
    assert MIN_CHARS_PER_PAGE == 40
    assert CID_RATIO_MAX == 0.05


def test_a_short_cover_page_stays_ok() -> None:
    assert classify_text_layer(["표지", "가" * 80, "나" * 80]) == "ok"


def test_one_empty_page_is_none_even_when_the_rest_is_long() -> None:
    assert classify_text_layer(["가" * 80, ""]) == "none"
    assert classify_text_layer(["가" * 80, "   "]) == "none"


def test_document_median_below_forty_characters_is_broken() -> None:
    assert classify_text_layer(["가" * 39]) == "broken"
    assert classify_text_layer(["가" * 40]) == "ok"


def test_cid_pattern_is_broken_while_garbage_stays_under_the_cut() -> None:
    text = "가" * 1000 + "(cid:12345)" * 10

    assert garbage_ratio(text) <= GARBAGE_RATIO_MAX
    assert classify_text_layer([text]) == "broken"


def test_ragged_rows_need_more_than_one_width() -> None:
    assert table_is_ragged([["a", "b"], ["a", "b", "c"]])
    assert not table_is_ragged([["a", "b"], ["c", ""]])


def test_fast_options_leave_the_scan_path_on_ocr() -> None:
    fast = fast_pipeline_options()
    scan = pipeline_options()

    assert fast.do_ocr is False
    assert fast.table_structure_options.mode == TableFormerMode.FAST
    assert scan.do_ocr is True
    assert scan.table_structure_options.mode == TableFormerMode.ACCURATE


class _Document:
    def __init__(self) -> None:
        self.pages = {1: object()}
        self.tables: list[object] = []
        self.pictures: list[object] = []

    def export_to_text(self, page_no: int) -> str:
        return "본문"


class _Converter:
    def __init__(self) -> None:
        self.calls = 0

    def convert(self, path: Path) -> SimpleNamespace:
        self.calls += 1
        return SimpleNamespace(
            document=_Document(),
            status=SimpleNamespace(name="SUCCESS"),
            errors=[],
        )


def _patch_converters(monkeypatch: pytest.MonkeyPatch) -> tuple[_Converter, _Converter]:
    scan = _Converter()
    fast = _Converter()
    monkeypatch.setattr(parse, "_converter", lambda: scan)
    monkeypatch.setattr(parse, "_fast_converter", lambda: fast)
    return scan, fast


def _inspection(**overrides: object) -> PdfInspection:
    values = {
        "page_texts": ["가" * 80],
        "markdown_pages": ["본문"],
        "tables_found": 0,
        "ragged": False,
        "borderless": False,
    }
    values.update(overrides)
    return PdfInspection(**values)


def test_routes_follow_the_text_layer_and_the_table_shape(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "note.pdf"
    path.write_bytes(b"%PDF-1.4")
    cases = [
        (_inspection(page_texts=["가" * 80, ""]), "text_layer_none", "scan"),
        (
            _inspection(page_texts=["가" * 1000 + "(cid:12345)" * 10]),
            "text_layer_broken",
            "scan",
        ),
        (_inspection(ragged=True, tables_found=1), "ragged_table", "fast"),
        (_inspection(borderless=True), "borderless_table", "fast"),
    ]
    for report, reason, kind in cases:
        scan, fast = _patch_converters(monkeypatch)
        monkeypatch.setattr(parse, "inspect_pdf", lambda _path, report=report: report)

        pages = parse_document(path)

        assert pages.fallback_reason == reason
        assert pages.parser == "docling"
        assert (fast.calls, scan.calls) == ((1, 0) if kind == "fast" else (0, 1))


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def write_pdf(path: Path, pages: list[list[list[str]]]) -> None:
    streams = []
    for lines in pages:
        commands = ["BT", "/F1 12 Tf"]
        for line_index, columns in enumerate(lines):
            for column_index, column in enumerate(columns):
                x = 72 + column_index * 180
                y = 720 - line_index * 24
                commands.append(f"1 0 0 1 {x} {y} Tm ({_escape(column)}) Tj")
        commands.append("ET")
        streams.append("\n".join(commands).encode("ascii"))

    blobs: dict[int, bytes] = {
        3: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"
    }
    page_ids = []
    for index, stream in enumerate(streams):
        content_id = 4 + index * 2
        page_id = content_id + 1
        page_ids.append(page_id)
        blobs[content_id] = (
            f"<< /Length {len(stream)} >>\nstream\n".encode("ascii")
            + stream
            + b"\nendstream"
        )
        blobs[page_id] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Contents {content_id} 0 R /Resources << /Font << /F1 3 0 R >> >> >>"
        ).encode("ascii")
    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
    blobs[2] = f"<< /Type /Pages /Count {len(streams)} /Kids [{kids}] >>".encode(
        "ascii"
    )
    blobs[1] = b"<< /Type /Catalog /Pages 2 0 R >>"
    max_id = 3 + len(streams) * 2
    out = bytearray(b"%PDF-1.4\n")
    offsets = [0] * (max_id + 1)
    for number in range(1, max_id + 1):
        offsets[number] = len(out)
        out += f"{number} 0 obj\n".encode("ascii")
        out += blobs[number]
        out += b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {max_id + 1}\n".encode("ascii")
    out += b"0000000000 65535 f \n"
    for number in range(1, max_id + 1):
        out += f"{offsets[number]:010d} 00000 n \n".encode("ascii")
    out += (
        f"trailer << /Size {max_id + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n"
    ).encode("ascii")
    path.write_bytes(bytes(out))


def test_short_cover_pdf_stays_ok(tmp_path: Path) -> None:
    path = tmp_path / "cover.pdf"
    write_pdf(
        path,
        [[["COVER"]], [[("A" * 80)]], [[("B" * 80)]]],
    )

    pages = parse_document(path)

    assert pages.text_layer == "ok"
    assert pages.parser == "pdfplumber"
    assert pages.fallback_reason is None
    assert "COVER" in pages[0]["text"]
    assert "A" * 80 in pages[1]["text"]


def test_cid_pdf_is_broken_and_uses_the_scan_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "cid.pdf"
    write_pdf(path, [[["(cid:12345)" * 12]]])
    scan, fast = _patch_converters(monkeypatch)

    pages = parse_document(path)

    assert pages.text_layer == "broken"
    assert pages.fallback_reason == "text_layer_broken"
    assert pages.parser == "docling"
    assert scan.calls == 1
    assert fast.calls == 0


def test_borderless_pdf_uses_fast_without_touching_scan(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "borderless.pdf"
    line = ["ColumnAlpha", "ColumnBravo", "ColumnDelta"]
    write_pdf(path, [[line, line, line]])
    scan, fast = _patch_converters(monkeypatch)

    pages = parse_document(path)

    assert pages.text_layer == "ok"
    assert pages.fallback_reason == "borderless_table"
    assert fast.calls == 1
    assert scan.calls == 0
