from pathlib import Path

import pytest

from zipsai.errors import PdfParseError
from zipsai.indexing.parse import (
    OCR_LANGUAGE,
    parse_document,
    pipeline_options,
)

FIXTURE = Path(__file__).parents[1] / "fixtures" / "two-pages.pdf"


def test_ocr_reads_korean_not_the_engine_default() -> None:
    # 지정하지 않으면 RapidOCR이 중국어로 읽어 한글 자리에 한자가 들어온다.
    options = pipeline_options()

    assert options.do_ocr is True
    assert options.ocr_options.lang == [OCR_LANGUAGE]


def test_images_use_the_index_reader(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from zipsai.indexing.parse import ParsedPages

    path = tmp_path / "notice.jpg"
    path.write_bytes(b"unused")
    seen: dict[str, Path] = {}

    def fake_reader(source: Path) -> ParsedPages:
        seen["path"] = Path(source)
        return ParsedPages([{"page": 1, "text": "안내문 본문"}])

    monkeypatch.setattr("zipsai.indexing.image_read.read_image_pages", fake_reader)
    monkeypatch.setattr(
        "zipsai.indexing.parse._converter",
        lambda: (_ for _ in ()).throw(AssertionError("image must not use docling")),
    )

    pages = parse_document(path)

    assert seen["path"] == path
    assert pages[0]["text"] == "안내문 본문"


def test_parse_document_returns_ordered_pages_with_expected_text() -> None:
    pages = parse_document(FIXTURE)

    assert [page["page"] for page in pages] == [1, 2]
    assert "page-one-known-text" in pages[0]["text"]
    assert "page-two-known-text" in pages[1]["text"]


def test_parse_document_missing_file_raises_typed_error(tmp_path: Path) -> None:
    with pytest.raises(PdfParseError):
        parse_document(tmp_path / "missing.pdf")


@pytest.mark.parametrize("contents", [b"", b"not a PDF"])
def test_parse_document_empty_or_invalid_file_raises_typed_error(
    tmp_path: Path, contents: bytes
) -> None:
    path = tmp_path / "invalid.pdf"
    path.write_bytes(contents)

    with pytest.raises(PdfParseError):
        parse_document(path)
