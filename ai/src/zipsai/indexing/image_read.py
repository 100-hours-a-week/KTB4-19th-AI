"""색인 사진·스캔 PDF 전용. 채팅 VLM 대체 호출(_call_with_fallback)은 쓰지 않는다."""

from __future__ import annotations

import base64
import logging
import os
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from openai import APIConnectionError, APIError, APIStatusError, APITimeoutError

from zipsai.errors import LlmUnavailableError
from zipsai.indexing.vlm_table import pipe_runs_to_markdown
from zipsai.integrations.llm import _get_client, _reason
from zipsai.settings import (
    INDEX_VLM_CONCURRENCY,
    INDEX_VLM_TIMEOUT_SECONDS,
    get_settings,
)

logger = logging.getLogger(__name__)

LONG_EDGE = 2048
JPEG_QUALITY = 90
CONTEXT_LEN = 8192
TOKEN_RESERVE = 300
MAX_OUTPUT_TOKENS = 4096
MIN_OUTPUT_TOKENS = 512
# 2048 출력 22장 중 R2-06만 5,983자이고 다음은 2,423자다.
# 12,000자 한도는 그 폭주를 놓쳐서, 둘 사이에 둔다.
RUNAWAY_CHARS = 5000
RUNAWAY_REPEAT = 5
RUNAWAY_LINE_CHARS = 4
# 공백을 뺀 글자가 이 값 이하면 빈 출력이다. 0이라 "공사중" 같은 짧은 안내문도 남긴다.
MIN_TEXT_CHARS = 0
PROMPT = """이 사진 속 안내문(문서)의 글을 그대로 옮겨 적어라.
- 사진에 보이는 글자를 그대로 적는다. 맞춤법이나 오타를 고치지 않는다.
- 요약하거나 설명하지 않는다. 없는 내용을 지어내지 않는다.
- 위에서 아래, 왼쪽에서 오른쪽 읽는 순서대로 적는다.
- 손글씨도 읽을 수 있으면 적는다.
- 표는 한 행을 한 줄에 쓰고, 칸 사이를 " | "로 구분한다.
- 안내문 본문이 아닌 배경(옆에 붙은 다른 종이, 광고, 워터마크, 로고)은 적지 않는다.
- 결과 글만 출력한다.
"""

_vlm_gate: threading.BoundedSemaphore | None = None
_vlm_gate_size: int | None = None
_paddle_engine_obj = None
_paddle_lock = threading.Lock()


@dataclass(frozen=True)
class PreparedImage:
    orig_size: tuple[int, int]
    upright: object
    sent_size: tuple[int, int]
    jpeg: bytes


@dataclass(frozen=True)
class VlmResult:
    text: str | None
    reason: str | None
    seconds: float
    est_tokens: int


def estimated_input_tokens(width: int, height: int) -> int:
    return (width // 32) * (height // 32) + TOKEN_RESERVE


def output_budget(width: int, height: int) -> int:
    room = CONTEXT_LEN - estimated_input_tokens(width, height) - TOKEN_RESERVE
    return min(MAX_OUTPUT_TOKENS, room)


def is_runaway(text: str) -> bool:
    if len(text) > RUNAWAY_CHARS:
        return True
    counts: dict[str, int] = {}
    for line in text.splitlines():
        key = "".join(line.split())
        if len(key) < RUNAWAY_LINE_CHARS:
            continue
        counts[key] = counts.get(key, 0) + 1
        if counts[key] >= RUNAWAY_REPEAT:
            return True
    return False


def _vlm_limit() -> threading.BoundedSemaphore:
    global _vlm_gate, _vlm_gate_size
    size = INDEX_VLM_CONCURRENCY
    if _vlm_gate is None or _vlm_gate_size != size:
        _vlm_gate = threading.BoundedSemaphore(size)
        _vlm_gate_size = size
    return _vlm_gate


def prepare_image(path: Path) -> PreparedImage:
    from PIL import Image, ImageOps

    with Image.open(path) as image:
        orig_size = image.size
        upright = ImageOps.exif_transpose(image).convert("RGB")
    return _shrink(upright, orig_size)


def _shrink(upright: object, orig_size: tuple[int, int]) -> PreparedImage:
    from PIL import Image

    sent = upright
    long_edge = max(sent.size)
    if long_edge > LONG_EDGE:
        scale = LONG_EDGE / long_edge
        sent = sent.resize(
            (
                max(1, round(sent.width * scale)),
                max(1, round(sent.height * scale)),
            ),
            Image.Resampling.LANCZOS,
        )
    from io import BytesIO

    buffer = BytesIO()
    sent.save(buffer, format="JPEG", quality=JPEG_QUALITY)
    return PreparedImage(
        orig_size=orig_size,
        upright=upright,
        sent_size=sent.size,
        jpeg=buffer.getvalue(),
    )


def _message_text(message: object) -> str | None:
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return None
    parts: list[str] = []
    for part in content:
        if isinstance(part, str):
            parts.append(part)
            continue
        text = getattr(part, "text", None)
        if isinstance(text, str):
            parts.append(text)
    return "".join(parts)


def _vlm_complete(settings: object, jpeg: bytes, budget: int) -> object:
    data_url = "data:image/jpeg;base64," + base64.standard_b64encode(jpeg).decode("ascii")
    client = _get_client(
        settings.llm_api_key,
        settings.llm_base_url,
        INDEX_VLM_TIMEOUT_SECONDS,
    )
    return client.chat.completions.create(
        model=settings.vlm_model,
        temperature=0,
        max_tokens=budget,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
    )


def call_index_vlm(jpeg: bytes, width: int, height: int) -> VlmResult:
    est = estimated_input_tokens(width, height)
    budget = output_budget(width, height)
    provider = os.getenv("ACTIVE_MODEL_PROVIDER", "serverless")
    if provider == "openrouter" or budget < MIN_OUTPUT_TOKENS:
        return VlmResult(None, "vlm_skipped", 0.0, est)
    try:
        settings = get_settings()
    except LlmUnavailableError:
        return VlmResult(None, "vlm_error", 0.0, est)
    if settings.provider == "openrouter":
        return VlmResult(None, "vlm_skipped", 0.0, est)

    started = perf_counter()
    try:
        with _vlm_limit():
            response = _vlm_complete(settings, jpeg, budget)
    except (APITimeoutError, APIConnectionError, APIStatusError, APIError):
        return VlmResult(None, "vlm_error", perf_counter() - started, est)

    seconds = perf_counter() - started
    choices = getattr(response, "choices", None)
    if not choices:
        return VlmResult(None, "vlm_error", seconds, est)
    message = choices[0].message
    finish = _reason(getattr(choices[0], "finish_reason", None))
    if finish == "length":
        return VlmResult(None, "vlm_truncated", seconds, est)
    text = _message_text(message)
    if text is None:
        return VlmResult(None, "vlm_error", seconds, est)
    if is_runaway(text):
        return VlmResult(text, "vlm_runaway", seconds, est)
    if len("".join(text.split())) <= MIN_TEXT_CHARS:
        return VlmResult(text, "vlm_empty", seconds, est)
    return VlmResult(text, None, seconds, est)


def _paddle_engine():
    global _paddle_engine_obj
    if _paddle_engine_obj is None:
        from paddleocr import PaddleOCR

        _paddle_engine_obj = PaddleOCR(
            text_detection_model_name="PP-OCRv6_small_det",
            text_recognition_model_name="korean_PP-OCRv5_mobile_rec",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            text_rec_score_thresh=0,
            device="cpu",
        )
    return _paddle_engine_obj


def _points_of(poly: object) -> list[tuple[float, float]]:
    if hasattr(poly, "tolist"):
        poly = poly.tolist()
    return [(float(point[0]), float(point[1])) for point in poly]


def join_paddle_lines(result: list) -> str:
    if not result:
        return ""
    payload = result[0].json
    body = payload.get("res", payload) if isinstance(payload, dict) else payload
    texts = list(body.get("rec_texts") or [])
    polys = body.get("rec_polys")
    if polys is None:
        polys = body.get("dt_polys") or []
    ranked = []
    for index, text in enumerate(texts):
        poly = polys[index] if index < len(polys) else [[0, index], [0, index]]
        points = _points_of(poly)
        top = min(point[1] for point in points)
        left = min(point[0] for point in points)
        ranked.append((top, left, index, str(text)))
    ranked.sort()
    return "\n".join(item[3] for item in ranked)


def call_index_paddle(image: object) -> tuple[str, float]:
    started = perf_counter()
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
        temp_path = handle.name
    try:
        image.save(temp_path, format="PNG")
        with _paddle_lock:
            result = _paddle_engine().predict(temp_path)
    finally:
        Path(temp_path).unlink(missing_ok=True)
    return join_paddle_lines(result), perf_counter() - started


def _log_image(
    *,
    route: str,
    reason: str | None,
    prepared: PreparedImage,
    est_tokens: int,
    out_chars: int,
    vlm_s: float,
    paddle_s: float,
) -> None:
    logger.info(
        "index_image",
        extra={
            "image_route": route,
            "image_reason": reason,
            "orig_size": f"{prepared.orig_size[0]}x{prepared.orig_size[1]}",
            "sent_size": f"{prepared.sent_size[0]}x{prepared.sent_size[1]}",
            "est_tokens": est_tokens,
            "out_chars": out_chars,
            "vlm_s": round(vlm_s, 3),
            "paddle_s": round(paddle_s, 3),
        },
    )


def read_image_pages(source: str | Path):
    from zipsai.errors import PdfParseError
    from zipsai.indexing.parse import ParsedPages, _annotate_route

    path = Path(source)
    prepared = prepare_image(path)
    vlm = call_index_vlm(prepared.jpeg, prepared.sent_size[0], prepared.sent_size[1])
    paddle_s = 0.0
    text: str | None = None
    route = "failed"
    reason = vlm.reason
    out_chars = len(vlm.text or "")
    if vlm.reason is None and vlm.text is not None:
        text = pipe_runs_to_markdown(vlm.text)
        route = "vlm"
        reason = None
        out_chars = len(text)
    else:
        try:
            paddle_text, paddle_s = call_index_paddle(prepared.upright)
        except (OSError, RuntimeError, ValueError, ImportError, TypeError):
            paddle_text = ""
        if len("".join(paddle_text.split())) > MIN_TEXT_CHARS:
            text = paddle_text
            route = "paddle"
            out_chars = len(text)
        else:
            out_chars = len(paddle_text)
    _log_image(
        route=route,
        reason=reason,
        prepared=prepared,
        est_tokens=vlm.est_tokens,
        out_chars=out_chars,
        vlm_s=vlm.seconds,
        paddle_s=paddle_s,
    )
    if text is None:
        raise PdfParseError(f"Could not read image: {path}")
    pages = ParsedPages([{"page": 1, "text": text}])
    pages.parser = "qwen-vl" if route == "vlm" else "paddleocr"
    pages.parser_status = "SUCCESS"
    pages.ocr_used = True
    pages.ocr_pages = 1
    pages.empty_pages = 0 if text.strip() else 1
    pages.min_chars_per_page = len(text)
    pages.text_layer = None
    return _annotate_route(
        pages,
        route="index_vlm" if route == "vlm" else "index_paddle",
        reason=reason or "image",
        convert_s=vlm.seconds + paddle_s,
    )


def _log_scan_page(
    *,
    page: int,
    route: str,
    reason: str | None,
    prepared: PreparedImage,
    est_tokens: int,
    out_chars: int,
    vlm_s: float,
    ocr_s: float,
) -> None:
    logger.info(
        "index_scan_page",
        extra={
            "page": page,
            "scan_route": route,
            "scan_reason": reason,
            "sent_size": f"{prepared.sent_size[0]}x{prepared.sent_size[1]}",
            "est_tokens": est_tokens,
            "out_chars": out_chars,
            "vlm_s": round(vlm_s, 3),
            "ocr_s": round(ocr_s, 3),
        },
    )


def read_scan_pages(source: str | Path, *, reason: str | None, layer: str | None):
    """쪽마다 VLM을 먼저 부르고, 실패한 쪽만 docling EasyOCR로 읽는다.

    pdfium이 파일을 못 열면 None을 돌려 문서 전체를 기존 docling 경로로 넘긴다.
    """

    import pypdfium2 as pdfium

    from zipsai.errors import PdfParseError
    from zipsai.indexing.parse import ParsedPages, _annotate_route, ocr_scan_page

    path = Path(source)
    try:
        document = pdfium.PdfDocument(path)
    except pdfium.PdfiumError:
        return None
    pages: list[dict[str, int | str]] = []
    routes: set[str] = set()
    total_s = 0.0
    try:
        for index in range(len(document)):
            page_no = index + 1
            page = document[index]
            try:
                width, height = page.get_size()
                # 사진 경로와 같은 긴 변 2048px로 바로 그려 다시 줄이지 않는다.
                bitmap = page.render(scale=LONG_EDGE / max(width, height))
                image = bitmap.to_pil().convert("RGB")
            finally:
                page.close()
            prepared = _shrink(image, image.size)
            vlm = call_index_vlm(
                prepared.jpeg, prepared.sent_size[0], prepared.sent_size[1]
            )
            ocr_s = 0.0
            if vlm.reason is None and vlm.text is not None:
                text = pipe_runs_to_markdown(vlm.text)
                route = "vlm"
            else:
                text, ocr_s = ocr_scan_page(path, page_no)
                route = "ocr"
            routes.add(route)
            total_s += vlm.seconds + ocr_s
            _log_scan_page(
                page=page_no,
                route=route,
                reason=vlm.reason,
                prepared=prepared,
                est_tokens=vlm.est_tokens,
                out_chars=len(text),
                vlm_s=vlm.seconds,
                ocr_s=ocr_s,
            )
            pages.append({"page": page_no, "text": text})
    finally:
        document.close()
    if not pages:
        raise PdfParseError(f"Document has no pages: {path}")
    if not any(str(item["text"]).strip() for item in pages):
        raise PdfParseError(f"Document has no usable text: {path}")
    parsed = ParsedPages(pages)
    texts = [str(item["text"]) for item in pages]
    parsed.empty_pages = sum(1 for text in texts if not text.strip())
    parsed.min_chars_per_page = min(len(text) for text in texts)
    parsed.parser_status = "SUCCESS"
    parsed.ocr_used = True
    parsed.ocr_pages = len(pages)
    parsed.fallback_reason = reason
    parsed.text_layer = layer
    if routes == {"vlm"}:
        parsed.parser, route_name = "qwen-vl", "index_vlm"
    elif routes == {"ocr"}:
        parsed.parser, route_name = "docling", "docling_ocr"
    else:
        parsed.parser, route_name = "qwen-vl+docling", "index_vlm_mixed"
    return _annotate_route(
        parsed,
        route=route_name,
        reason=reason or "text_layer_none",
        convert_s=total_s,
    )
