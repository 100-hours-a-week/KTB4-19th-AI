"""임베딩 서버 구조화 로그. ai-api 포매터와 같은 공통 필드를 낸다."""

import json
import logging
import os
import re
from datetime import UTC, datetime
from typing import Any, Final
from urllib.parse import urlparse

SERVICE_NAME: Final = "embedding"
COMMON_FIELDS: Final = frozenset(
    {"timestamp", "level", "service", "env", "version", "logger", "event"}
)
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
    re.compile(r"(?i)(api[-_]?key|authorization|secret)([\"'\s:=]+)[^\s\"']{6,}"),
    re.compile(r"\b[0-9a-fA-F]{32,}\b"),
    re.compile(r"\b[A-Za-z0-9+]{40,}={0,2}\b"),
    re.compile(r"01[016789]-?\d{3,4}-?\d{4}"),
    re.compile(r"\d{2,3}[가-힣]\d{4}"),
    re.compile(r"X-Amz-[A-Za-z0-9\-_]+=[^&\s]+"),
    re.compile(r"[가-힣]{2,}(?:\s+[가-힣0-9A-Za-z]{1,})+"),
)
_URL_PATTERN: Final = re.compile(r"https?://[^\s\"']+")
_REDACTED: Final = "[REDACTED]"
_RESERVED = frozenset(logging.LogRecord("", 0, "", 0, "", None, None).__dict__) | {
    "message",
    "asctime",
    "taskName",
}


def runtime_env() -> str | None:
    return os.getenv("LANGFUSE_TRACING_ENVIRONMENT") or None


def runtime_version() -> str | None:
    return os.getenv("ZIPSAI_VERSION") or None


def _scrub_url(url: str) -> str:
    parsed = urlparse(url)
    host = parsed.hostname
    if parsed.username or parsed.password or parsed.query or not host:
        return _REDACTED
    if re.search(r"/[A-Za-z0-9_\-]{16,}", parsed.path or ""):
        return f"{parsed.scheme}://{host}/[REDACTED]"
    return url


def scrub_text(value: str) -> tuple[str, bool]:
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
    if key in _SENSITIVE_KEYS:
        return _REDACTED, True
    if isinstance(value, str):
        return scrub_text(value)
    if isinstance(value, dict):
        redacted = False
        cleaned: dict[Any, Any] = {}
        for child_key, child in value.items():
            cleaned[child_key], child_redacted = scrub_value(str(child_key), child)
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
    def format(self, record: logging.LogRecord) -> str:
        redacted = False
        event, event_redacted = scrub_text(record.getMessage())
        redacted = redacted or event_redacted
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(
                timespec="milliseconds"
            ),
            "level": record.levelname,
            "service": SERVICE_NAME,
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


class _HealthFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return "/health" not in message and "/metrics" not in message


def configure_logging(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logging.basicConfig(level=level, handlers=[handler], force=True)
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        target = logging.getLogger(name)
        json_handler = logging.StreamHandler()
        json_handler.setFormatter(JsonFormatter())
        target.handlers = [json_handler]
        target.propagate = False
    logging.getLogger("uvicorn.access").addFilter(_HealthFilter())
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
