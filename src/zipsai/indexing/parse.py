from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from zipsai.errors import PdfParseError


def parse_pdf(source: str | Path) -> list[dict[str, int | str]]:
    path = Path(source)
    try:
        if not path.is_file():
            raise PdfParseError(f"PDF file not found: {path}")
        if path.stat().st_size == 0:
            raise PdfParseError(f"PDF file is empty: {path}")
    except OSError as exc:
        raise PdfParseError(f"PDF file is unreadable: {path}") from exc

    try:
        pipeline_options = PdfPipelineOptions(do_ocr=True)
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        document = converter.convert(path).document
        page_numbers = sorted(document.pages)
        if not page_numbers:
            raise PdfParseError(f"PDF has no pages: {path}")

        pages = [
            {"page": page_number, "text": document.export_to_text(page_no=page_number)}
            for page_number in page_numbers
        ]
        if not any(page["text"].strip() for page in pages):
            raise PdfParseError(f"PDF has no usable text: {path}")
        return pages
    except PdfParseError:
        raise
    except Exception as exc:
        raise PdfParseError(f"Could not parse PDF: {path}") from exc
