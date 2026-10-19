from functools import lru_cache
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
from docling.document_converter import (
    DocumentConverter,
    ImageFormatOption,
    PdfFormatOption,
)

from zipsai.errors import PdfParseError


class ParsedPages(list[dict[str, int | str]]):
    """페이지 목록에 표·빈 페이지 개수를 붙인다. 실패 이유는 docling이 안 준다."""

    tables_found: int | None = None
    tables_parsed: int | None = None
    tables_failed: int | None = None
    table_fail_reasons: dict[str, int] | None = None
    empty_pages: int | None = None
    min_chars_per_page: int | None = None
    images_found: int | None = None
    parser_status: str | None = None
    ocr_used: bool | None = None
    ocr_pages: int | None = None
    failed_pages: list[int] | None = None


def conversion_stats(result: object, pages: list[dict[str, int | str]]) -> ParsedPages:
    parsed = ParsedPages(pages)
    document = getattr(result, "document", None)
    tables = list(getattr(document, "tables", []) or [])
    parsed.tables_found = len(tables)
    parsed.tables_parsed = len(tables)
    # 버린 표의 이유는 ErrorItem이 구조 인식 실패를 구분하지 않는다.
    parsed.tables_failed = None
    parsed.table_fail_reasons = None
    texts = [str(page.get("text", "")) for page in pages]
    parsed.empty_pages = sum(1 for text in texts if not text.strip())
    parsed.min_chars_per_page = min((len(text) for text in texts), default=0)
    pictures = getattr(document, "pictures", None)
    parsed.images_found = len(pictures) if pictures is not None else None
    status = getattr(result, "status", None)
    parsed.parser_status = getattr(status, "name", None) if status is not None else None
    if parsed.parser_status is None and status is not None:
        parsed.parser_status = str(status)
    parsed.ocr_used = None
    parsed.ocr_pages = None
    failed: list[int] = []
    for error in getattr(result, "errors", []) or []:
        page_no = getattr(error, "page_no", None)
        if isinstance(page_no, int):
            failed.append(page_no)
    parsed.failed_pages = sorted(set(failed)) or None
    return parsed

# RapidOCR 기본 언어가 중국어라 지정하지 않으면 한글이 한자로 나온다.
OCR_LANGUAGE = "korean"
# RapidOcrOptions 기본 backend는 onnxruntime인데 설치돼 있지 않다. 이미 받아둔 torch를 쓴다.
OCR_BACKEND = "torch"


def pipeline_options() -> PdfPipelineOptions:
    return PdfPipelineOptions(
        do_ocr=True,
        ocr_options=RapidOcrOptions(lang=[OCR_LANGUAGE], backend=OCR_BACKEND),
    )


@lru_cache(maxsize=1)
def _converter() -> DocumentConverter:
    # 문서마다 새로 만들면 OCR·레이아웃 모델을 매번 다시 올린다.
    options = pipeline_options()
    return DocumentConverter(
        format_options={
            # 스캔본은 PDF, 사진으로 찍어 올린 공지는 이미지로 들어온다. 둘 다 같은 설정으로 읽는다.
            InputFormat.PDF: PdfFormatOption(pipeline_options=options),
            InputFormat.IMAGE: ImageFormatOption(pipeline_options=options),
        }
    )


def parse_document(source: str | Path) -> ParsedPages:
    path = Path(source)
    try:
        if not path.is_file():
            raise PdfParseError(f"File not found: {path}")
        if path.stat().st_size == 0:
            raise PdfParseError(f"File is empty: {path}")
    except OSError as exc:
        raise PdfParseError(f"File is unreadable: {path}") from exc

    try:
        converted = _converter().convert(path)
        document = converted.document
        page_numbers = sorted(document.pages)
        if not page_numbers:
            raise PdfParseError(f"Document has no pages: {path}")

        pages = [
            {"page": page_number, "text": document.export_to_text(page_no=page_number)}
            for page_number in page_numbers
        ]
        if not any(page["text"].strip() for page in pages):
            raise PdfParseError(f"Document has no usable text: {path}")
        return conversion_stats(converted, pages)
    except PdfParseError:
        raise
    except Exception as exc:
        raise PdfParseError(f"Could not parse document: {path}") from exc
