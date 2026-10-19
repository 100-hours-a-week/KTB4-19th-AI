import json
import logging

from fastapi.testclient import TestClient

from zipsai.access import AccessLogMiddleware
from zipsai.main import app
from zipsai.observability import COMMON_FIELDS, SERVICE_NAME, JsonFormatter

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
