from pathlib import Path

import pytest

from zipsai.errors import PdfParseError
from zipsai.indexing.parse import parse_pdf

FIXTURE = Path(__file__).parents[1] / "fixtures" / "two-pages.pdf"


def test_parse_pdf_returns_ordered_pages_with_expected_text() -> None:
    pages = parse_pdf(FIXTURE)

    assert [page["page"] for page in pages] == [1, 2]
    assert "page-one-known-text" in pages[0]["text"]
    assert "page-two-known-text" in pages[1]["text"]


def test_parse_pdf_missing_file_raises_typed_error(tmp_path: Path) -> None:
    with pytest.raises(PdfParseError):
        parse_pdf(tmp_path / "missing.pdf")


@pytest.mark.parametrize("contents", [b"", b"not a PDF"])
def test_parse_pdf_empty_or_invalid_file_raises_typed_error(
    tmp_path: Path, contents: bytes
) -> None:
    path = tmp_path / "invalid.pdf"
    path.write_bytes(contents)

    with pytest.raises(PdfParseError):
        parse_pdf(path)
