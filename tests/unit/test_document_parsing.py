from pathlib import Path

import pytest
from docling.datamodel.base_models import InputFormat

from zipsai.errors import PdfParseError
from zipsai.indexing.parse import (
    OCR_LANGUAGE,
    _converter,
    parse_document,
    pipeline_options,
)

FIXTURE = Path(__file__).parents[1] / "fixtures" / "two-pages.pdf"


def test_ocr_reads_korean_not_the_engine_default() -> None:
    # 지정하지 않으면 RapidOCR이 중국어로 읽어 한글 자리에 한자가 들어온다.
    options = pipeline_options()

    assert options.do_ocr is True
    assert options.ocr_options.lang == [OCR_LANGUAGE]


def test_images_are_parsed_with_the_same_ocr_settings() -> None:
    # 사진으로 찍어 올린 공지가 이미지로 들어온다. 등록하지 않으면 엔진 기본값으로 떨어진다.
    registered = _converter().format_to_options

    for input_format in (InputFormat.PDF, InputFormat.IMAGE):
        options = registered[input_format].pipeline_options
        assert options.ocr_options.lang == [OCR_LANGUAGE]


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
