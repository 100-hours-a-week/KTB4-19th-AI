"""구조화 로그. 단계마다 걸린 시간과 결과를 한 줄씩 남긴다.

평문 로그는 CloudWatch Logs Insights가 필드로 쪼개지 못해 집계가 안 된다.
JSON 한 줄로 남기고, 한 요청에 속한 줄들은 turn_id·job_id로 묶는다.
"""

import json
import logging
import os
import re
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any, Final
from urllib.parse import urlparse

from zipsai.errors import error_code
from zipsai.tracing import trace_step

# 백엔드·embedding 로그와 같은 스키마로 조회하기 위한 공통 필드. 설정으로 뺄 값이
# 아니라 이 프로세스의 정체라서 여기가 단일 출처다.
SERVICE_NAME: Final = "ai-api"

# 두 서비스 포매터가 같이 맞춰야 하는 줄 공통 키.
COMMON_FIELDS: Final = frozenset(
    {"timestamp", "level", "service", "env", "version", "logger", "event"}
)

# 값 자체가 입주민 문장·프롬프트·본문인 키. 여기 들어오면 통째로 가린다.
_SENSITIVE_KEYS: Final = frozenset(
    {
        "answer",
        "body",
        "chunk",
        "content",
        "document",
        "document_text",
        "input",
        "location",
        "message_text",
        "ocr",
        "ocr_text",
        "output",
        "prompt",
        "query",
        "question",
        "reply",
        "summary",
        "symptom",
        "system_prompt",
        "text",
        "texts",
        "url",
        "user_prompt",
    }
)

_SECRET_PATTERNS: Final = (
    re.compile(r"sk-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"Bearer\s+\S+", re.IGNORECASE),
    re.compile(
        r"(?i)(api[-_]?key|authorization|secret)([\"'\s:=]+)[^\s\"']{6,}"
    ),
    re.compile(r"\b[0-9a-fA-F]{32,}\b"),
    # 슬래시는 빼 둔다. 넣으면 트레이스백의 파일 경로가 통째로 지워진다.
    re.compile(r"\b[A-Za-z0-9+]{40,}={0,2}\b"),
    re.compile(r"01[016789]-?\d{3,4}-?\d{4}"),
    re.compile(r"\d{2,3}[가-힣]\d{4}"),
    re.compile(r"X-Amz-[A-Za-z0-9\-_]+=[^&\s]+"),
    # 띄어쓴 한글은 입주민 문장이다. 분류 이름(한 단어)은 남겨 둔다.
    re.compile(r"[가-힣]{2,}(?:\s+[가-힣0-9A-Za-z]{1,})+"),
)
_URL_PATTERN: Final = re.compile(r"https?://[^\s\"']+")
_REDACTED: Final = "[REDACTED]"

# 로그 그룹마다 시각 기준이 다르면(ai-api는 KST, qdrant·mysql은 UTC) 같은 사건을
# 대조할 때마다 9시간을 암산해야 한다. 여기서 UTC로 고정한다.
_RESERVED = frozenset(logging.LogRecord("", 0, "", 0, "", None, None).__dict__) | {
    "message",
    "asctime",
    "taskName",
}

_context: ContextVar[dict[str, Any] | None] = ContextVar("log_context", default=None)
_timings: ContextVar[dict[str, int] | None] = ContextVar("stage_timings", default=None)
_usage_totals: ContextVar[dict[str, int] | None] = ContextVar(
    "llm_usage_totals", default=None
)
_stage_name: ContextVar[str | None] = ContextVar("stage_name", default=None)

_in_flight = 0
_in_flight_lock = threading.Lock()

# 첫 요청은 docling·인코더·Qdrant 클라이언트를 만들며 수십 초가 더 걸린다.
# 이 값이 없으면 "파싱이 52초"라는 잘못된 결론으로 간다. 질의와 색인은 서로 다른
# 모델을 올리므로 플래그를 하나로 묶으면 먼저 온 쪽이 나머지의 첫 판을 가린다.
_warm: set[str] = set()


def runtime_env() -> str | None:
    return os.getenv("LANGFUSE_TRACING_ENVIRONMENT") or None


def runtime_version() -> str | None:
    return os.getenv("ZIPSAI_VERSION") or None


def current_stage() -> str | None:
    return _stage_name.get()


def _observe_stage(
    name: str,
    outcome: str,
    duration_seconds: float,
    error_code: str | None = None,
) -> None:
    from zipsai.metrics import observe_stage

    observe_stage(name, outcome, duration_seconds, error_code)


def _scrub_url(url: str) -> str:
    parsed = urlparse(url)
    host = parsed.hostname
    if parsed.username or parsed.password or parsed.query or not host:
        return _REDACTED
    if re.search(r"/[A-Za-z0-9_\-]{16,}", parsed.path or ""):
        return f"{parsed.scheme}://{host}/[REDACTED]"
    return url


def scrub_text(value: str) -> tuple[str, bool]:
    """키·토큰·전화·문장처럼 보이면 지운다. 지웠으면 True."""
    redacted = False
    text = value

    def mark(match: re.Match[str]) -> str:
        nonlocal redacted
        redacted = True
        return _REDACTED

    text = _URL_PATTERN.sub(lambda match: _scrub_url(match.group(0)), text)
    if text != value:
        redacted = True
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(mark, text)
    return text, redacted


def scrub_value(key: str, value: Any) -> tuple[Any, bool]:
    # 키 이름이 민감해도 건수는 남긴다. 본문 문자열·목록·객체만 통째로 가린다.
    if key in _SENSITIVE_KEYS and isinstance(value, (str, list, dict)):
        return _REDACTED, True
    if isinstance(value, str):
        return scrub_text(value)
    if isinstance(value, dict):
        redacted = False
        cleaned: dict[Any, Any] = {}
        for child_key, child in value.items():
            name = str(child_key)
            cleaned[child_key], child_redacted = scrub_value(name, child)
            redacted = redacted or child_redacted
        return cleaned, redacted
    if isinstance(value, list):
        redacted = False
        cleaned_list = []
        for child in value:
            item, child_redacted = scrub_value("", child)
            cleaned_list.append(item)
            redacted = redacted or child_redacted
        return cleaned_list, redacted
    return value, False


class JsonFormatter(logging.Formatter):
    def __init__(self, service: str = SERVICE_NAME) -> None:
        super().__init__()
        self.service = service

    def format(self, record: logging.LogRecord) -> str:
        redacted = False
        event, event_redacted = scrub_text(record.getMessage())
        redacted = redacted or event_redacted
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(
                timespec="milliseconds"
            ),
            "level": record.levelname,
            "service": self.service,
            "env": runtime_env(),
            "version": runtime_version(),
            "logger": record.name,
            "event": event,
        }
        for key, value in record.__dict__.items():
            if key in _RESERVED:
                continue
            cleaned, value_redacted = scrub_value(key, value)
            payload[key] = cleaned
            redacted = redacted or value_redacted
        if record.exc_info:
            traceback, trace_redacted = scrub_text(self.formatException(record.exc_info))
            payload["traceback"] = traceback
            redacted = redacted or trace_redacted
        if redacted:
            payload["redacted"] = True
        return json.dumps(payload, ensure_ascii=False, default=str)


class _ContextFilter(logging.Filter):
    """공통 필드를 로그를 만든 시점에 붙인다.

    포매터에서 읽으면 출력 시점의 값을 보게 되어, 그 사이 컨텍스트를 벗어난
    로그는 turn_id를 잃는다.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        for key, value in (_context.get() or {}).items():
            if not hasattr(record, key):
                setattr(record, key, value)
        return True


class _HealthFilter(logging.Filter):
    """헬스체크 접근 로그를 버린다.

    10초마다 도는 헬스체크가 운영 로그의 96%를 차지해 실제 업무 로그가 묻힌다.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return "/health" not in message and "/metrics" not in message


def _json_handler() -> logging.Handler:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    handler.addFilter(_ContextFilter())
    return handler


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, handlers=[_json_handler()], force=True)

    # uvicorn CLI가 먼저 붙인 평문 핸들러를 같은 JSON 핸들러로 갈아끼운다.
    # 접속 줄은 미들웨어가 http_access로 남기므로 uvicorn 기본 접속 로그는 끈다.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        target = logging.getLogger(name)
        target.handlers = [_json_handler()]
        target.propagate = False
    logging.getLogger("uvicorn.access").addFilter(_HealthFilter())
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    # ai-api의 헬스체크는 embedding·qdrant를 실제로 찔러 httpx 로그를 두 줄 더 만든다.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    # 파싱은 문서 한 건마다 모델 로딩·변환 로그를 수십 줄 쏟는다.
    # 걸린 시간은 parse 단계가 재므로 경고부터만 받는다.
    for noisy in ("docling", "RapidOCR", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


@contextmanager
def bind(**fields: Any):
    """이 블록 안의 모든 로그에 공통 필드를 붙인다."""
    token = _context.set({**(_context.get() or {}), **fields})
    try:
        yield
    finally:
        _context.reset(token)


def add_context(**fields: Any) -> None:
    """이미 열려 있는 bind 블록에 필드를 덧붙인다.

    라우팅 결과처럼 블록에 들어갈 때는 아직 모르는 값이 있다. bind가 만든 dict를
    그대로 고치므로 블록을 벗어나면 reset이 되돌린다.
    """
    current = _context.get()
    if current is not None:
        current.update(fields)


@contextmanager
def collect_timings():
    """이 블록 안 단계들의 소요시간을 모은다.

    단계마다 줄이 따로 남지만, 요청 하나가 어디서 시간을 썼는지 보려고 10줄을
    이어 붙이는 건 느리다. 마지막 요약 한 줄에 실어 보낸다.
    """
    collected: dict[str, int] = {}
    usage = {"input": 0, "output": 0}
    token = _timings.set(collected)
    usage_token = _usage_totals.set(usage)
    try:
        yield collected
    finally:
        _timings.reset(token)
        _usage_totals.reset(usage_token)


def note_llm_tokens(tokens: dict[str, int | None]) -> None:
    """이번 요청의 입력·출력 토큰을 모은다. 캐시는 입력에 포함돼 다시 더하지 않는다."""
    usage = _usage_totals.get()
    if usage is None:
        return
    for kind in ("input", "output"):
        amount = tokens.get(kind)
        if amount:
            usage[kind] = usage[kind] + int(amount)


def llm_token_total() -> int | None:
    usage = _usage_totals.get()
    if usage is None:
        return None
    return usage["input"] + usage["output"]


@contextmanager
def track_in_flight():
    """동시에 처리 중인 요청 수를 센다. 혼자 느린 건지 겹쳐서 느린 건지 가른다."""
    global _in_flight
    with _in_flight_lock:
        _in_flight += 1
        current = _in_flight
    try:
        yield current
    finally:
        with _in_flight_lock:
            _in_flight -= 1


def is_cold(scope: str) -> bool:
    """이번이 그 범위의 첫 작업인지. 호출 즉시 warm으로 넘긴다."""
    cold = scope not in _warm
    _warm.add(scope)
    return cold


@contextmanager
def stage(name: str, logger: logging.Logger, **fields: Any):
    """한 단계를 감싸 소요시간과 결과를 남긴다.

    성공·실패를 가리지 않고 항상 duration_ms를 남긴다. 실패 로그만 있으면
    "정상인데 느린" 구간을 영영 못 찾는다. 본문에서 yield된 dict에 담은 값이
    그대로 필드가 된다.
    """
    extra: dict[str, Any] = dict(fields)
    started = time.perf_counter()
    stage_token = _stage_name.set(name)
    try:
        with trace_step(name) as trace:
            try:
                yield extra
            except Exception as error:
                if trace is not None:
                    trace.update(
                        output={"outcome": "fail", "error_code": error_code(error)}
                    )
                raise
            else:
                if trace is not None:
                    trace.update(output={"outcome": "ok"})
    except Exception as error:
        code = error_code(error)
        logger.exception(
            "stage_done",
            extra={
                "stage": name,
                "duration_ms": _record(name, started),
                "outcome": "fail",
                "error_code": code,
                "error_type": type(error).__name__,
                "error": str(error),
                **extra,
            },
        )
        # 작업 전체 요약 한 줄에 어느 단계에서 죽었는지를 담기 위해 올려 보낸다.
        error.stage = name
        _observe_stage(name, "fail", _record(name, started) / 1000, code)
        raise
    finally:
        _stage_name.reset(stage_token)
    duration_ms = _record(name, started)
    _observe_stage(name, "ok", duration_ms / 1000)
    logger.info(
        "stage_done",
        extra={
            "stage": name,
            "duration_ms": _record(name, started),
            "outcome": "ok",
            "error_code": None,
            **extra,
        },
    )


def skipped(
    name: str, logger: logging.Logger, *, skip_reason: str, **fields: Any
) -> None:
    """앞 단계에서 막혀 아예 돌지 않은 단계를 남긴다.

    줄이 없으면 "느려서 안 찍힌 건지 건너뛴 건지" 구분할 수 없다. 이유는 기본값을
    두지 않는다. 기본값이 있으면 이유 없는 skipped 줄이 조용히 늘어난다.
    """
    _observe_stage(name, "skipped", 0)
    logger.info(
        "stage_done",
        extra={
            "stage": name,
            "duration_ms": 0,
            "outcome": "skipped",
            "skip_reason": skip_reason,
            "error_code": None,
            **fields,
        },
    )


def elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def _record(name: str, started: float) -> int:
    elapsed = elapsed_ms(started)
    timings = _timings.get()
    if timings is not None:
        timings[name] = elapsed
    return elapsed
