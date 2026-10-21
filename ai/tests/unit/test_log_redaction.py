import json
import logging

import pytest
from fastapi.testclient import TestClient

from zipsai.access import AccessLogMiddleware
from zipsai.main import app
from zipsai.observability import COMMON_FIELDS, SERVICE_NAME, JsonFormatter, scrub_text

_SENTENCE = "화장실 천장에서 물이 새요"


def _line(message: str, **extra: object) -> dict:
    record = logging.LogRecord(
        name="zipsai.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=message,
        args=(),
        exc_info=None,
    )
    for key, value in extra.items():
        setattr(record, key, value)
    return json.loads(JsonFormatter().format(record))


def test_formatter_emits_one_json_object_with_common_fields() -> None:
    payload = _line("request_started")

    assert COMMON_FIELDS <= payload.keys()
    assert payload["service"] == SERVICE_NAME
    assert payload["event"] == "request_started"
    assert payload["env"] is None or isinstance(payload["env"], str)
    assert payload["version"] is None or isinstance(payload["version"], str)


def test_formatter_redacts_secrets_phone_and_resident_text() -> None:
    payload = _line(
        f"leak {_SENTENCE}",
        error="sk-test-secret-key-value",
        header="Bearer xyz",
        phone="010-1234-5678",
        text=_SENTENCE,
    )
    rendered = json.dumps(payload, ensure_ascii=False)

    assert "sk-test" not in rendered
    assert "Bearer xyz" not in rendered
    assert "010-1234-5678" not in rendered
    assert _SENTENCE not in rendered
    assert payload["redacted"] is True
    assert payload["text"] == "[REDACTED]"


# embedding/tests/unit/test_logging.py와 같은 표다. 한쪽만 고치지 않는다.
_SCRUB_CASES = (
    ("sk-test-secret-key-value", "[REDACTED]", True),
    ("Bearer xyz", "[REDACTED]", True),
    ("010-1234-5678", "[REDACTED]", True),
    ("화장실 천장에서 물이 새요", "[REDACTED]", True),
    ("550e8400-e29b-41d4-a716-446655440000", "550e8400-e29b-41d4-a716-446655440000", False),
    ("0123456789abcdef0123456789abcdef", "[REDACTED]", True),
)


@pytest.mark.parametrize(("raw", "expected", "redacted"), _SCRUB_CASES)
def test_scrub_text_matches_the_shared_table(
    raw: str, expected: str, redacted: bool
) -> None:
    assert scrub_text(raw) == (expected, redacted)


def test_title_and_file_key_keep_sentences_and_hex_but_hide_secrets() -> None:
    title = "관리 규약 2026년 개정"
    path = "s3://zipsai-files/buildings/101/0123456789abcdef0123456789abcdef.pdf"
    kept = _line("job_accepted", doc_title=title, file_key_path=path)

    assert kept["doc_title"] == title
    assert kept["file_key_path"] == path
    assert "redacted" not in kept
    assert scrub_text(title) == ("[REDACTED]", True)
    assert "[REDACTED]" in scrub_text(path)[0]

    hidden = _line(
        "job_accepted",
        doc_title="관리 규약 Bearer abc123",
        file_key_path=(
            "s3://zipsai-files/buildings/101/Bearer abc123.pdf"
            "?X-Amz-Signature=secrettoken"
        ),
    )
    assert hidden["doc_title"] == "관리 규약 [REDACTED]"
    assert "Bearer" not in hidden["file_key_path"]
    assert "X-Amz-Signature" not in hidden["file_key_path"]
    assert hidden["redacted"] is True


def test_formatter_keeps_a_numeric_count_on_a_sensitive_key() -> None:
    payload = _line("stage_done", texts=3, text=["화장실 천장에서 물이 새요"])

    assert payload["texts"] == 3
    assert payload["text"] == "[REDACTED]"
    assert "물이 새요" not in json.dumps(payload, ensure_ascii=False)


def _access(caplog) -> list[logging.LogRecord]:
    return [record for record in caplog.records if record.name == "zipsai.access"]


def test_health_and_metrics_are_not_access_logged(caplog) -> None:
    client = TestClient(app)
    with caplog.at_level(logging.INFO):
        client.get("/health")
        client.get("/metrics")

    assert _access(caplog) == []


def test_http_access_log_is_json_and_omits_the_query(caplog) -> None:
    client = TestClient(app)
    with caplog.at_level(logging.INFO):
        client.get("/health?token=sekret")
        client.post("/api/v3/ai/converse", json={})

    access = _access(caplog)
    assert [record.getMessage() for record in access] == ["http_access"]
    assert access
    payload = json.loads(JsonFormatter().format(access[-1]))
    assert payload["event"] == "http_access"
    assert payload["path"] == "/api/v3/ai/converse"
    assert "sekret" not in json.dumps(payload)
    assert isinstance(payload["duration_ms"], int)
    assert AccessLogMiddleware is not None
