"""구조화 로그. 단계마다 걸린 시간과 결과를 한 줄씩 남긴다.

평문 로그는 CloudWatch Logs Insights가 필드로 쪼개지 못해 집계가 안 된다.
JSON 한 줄로 남기고, 한 요청에 속한 줄들은 trace_id·job_id로 묶는다.
"""

import json
import logging
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any, Final

from zipsai.errors import error_code

# 백엔드·embedding 로그와 같은 스키마로 조회하기 위한 공통 필드. 설정으로 뺄 값이
# 아니라 이 프로세스의 정체라서 여기가 단일 출처다.
SERVICE_NAME: Final = "ai-api"

# 로그 그룹마다 시각 기준이 다르면(ai-api는 KST, qdrant·mysql은 UTC) 같은 사건을
# 대조할 때마다 9시간을 암산해야 한다. 여기서 UTC로 고정한다.
_RESERVED = frozenset(logging.LogRecord("", 0, "", 0, "", None, None).__dict__) | {
    "message",
    "asctime",
    "taskName",
}

_context: ContextVar[dict[str, Any] | None] = ContextVar("log_context", default=None)
_timings: ContextVar[dict[str, int] | None] = ContextVar("stage_timings", default=None)

_in_flight = 0
_in_flight_lock = threading.Lock()

# 첫 요청은 docling·인코더·Qdrant 클라이언트를 만들며 수십 초가 더 걸린다.
# 이 값이 없으면 "파싱이 52초"라는 잘못된 결론으로 간다. 질의와 색인은 서로 다른
# 모델을 올리므로 플래그를 하나로 묶으면 먼저 온 쪽이 나머지의 첫 판을 가린다.
_warm: set[str] = set()


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(
                timespec="milliseconds"
            ),
            "level": record.levelname,
            "service": SERVICE_NAME,
            "logger": record.name,
            "event": record.getMessage(),
        }
        payload.update(
            {
                key: value
                for key, value in record.__dict__.items()
                if key not in _RESERVED
            }
        )
        if record.exc_info:
            payload["traceback"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


class _ContextFilter(logging.Filter):
    """공통 필드를 로그를 만든 시점에 붙인다.

    포매터에서 읽으면 출력 시점의 값을 보게 되어, 그 사이 컨텍스트를 벗어난
    로그는 trace_id를 잃는다.
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
        return "/health" not in record.getMessage()


def configure_logging(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    handler.addFilter(_ContextFilter())
    logging.basicConfig(level=level, handlers=[handler], force=True)

    # ai-api의 헬스체크는 embedding·qdrant를 실제로 찔러 httpx 로그를 두 줄 더 만든다.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").addFilter(_HealthFilter())
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
    token = _timings.set(collected)
    try:
        yield collected
    finally:
        _timings.reset(token)


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
    try:
        yield extra
    except Exception as error:
        logger.exception(
            "stage_done",
            extra={
                "stage": name,
                "duration_ms": _record(name, started),
                "outcome": "fail",
                "error_code": error_code(error),
                "error_type": type(error).__name__,
                "error": str(error),
                **extra,
            },
        )
        # 작업 전체 요약 한 줄에 어느 단계에서 죽었는지를 담기 위해 올려 보낸다.
        error.stage = name
        raise
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
