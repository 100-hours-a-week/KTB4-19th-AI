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


def parse_document(source: str | Path) -> list[dict[str, int | str]]:
    path = Path(source)
    try:
        if not path.is_file():
            raise PdfParseError(f"File not found: {path}")
        if path.stat().st_size == 0:
            raise PdfParseError(f"File is empty: {path}")
    except OSError as exc:
        raise PdfParseError(f"File is unreadable: {path}") from exc

    try:
        document = _converter().convert(path).document
        page_numbers = sorted(document.pages)
        if not page_numbers:
            raise PdfParseError(f"Document has no pages: {path}")

        pages = [
            {"page": page_number, "text": document.export_to_text(page_no=page_number)}
            for page_number in page_numbers
        ]
        if not any(page["text"].strip() for page in pages):
            raise PdfParseError(f"Document has no usable text: {path}")
        return pages
    except PdfParseError:
        raise
    except Exception as exc:
        raise PdfParseError(f"Could not parse document: {path}") from exc
