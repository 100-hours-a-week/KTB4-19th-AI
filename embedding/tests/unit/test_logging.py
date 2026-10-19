import json
import logging

from fastapi.testclient import TestClient

from embedding.logging import COMMON_FIELDS, SERVICE_NAME, JsonFormatter
from embedding.main import app

_SENTENCE = "화장실 천장에서 물이 새요"


def _line(message: str, **extra: object) -> dict:
    record = logging.LogRecord(
        name="embedding.test",
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


def test_formatter_emits_the_same_common_fields() -> None:
    payload = _line("embed_request")

    assert COMMON_FIELDS <= payload.keys()
    assert payload["service"] == SERVICE_NAME
    assert payload["event"] == "embed_request"


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


def test_health_is_not_access_logged(caplog) -> None:
    with caplog.at_level(logging.INFO, logger="embedding.access"):
        TestClient(app).get("/health")

    assert caplog.records == []
