import logging
from pathlib import Path

from zipsai.indexing.reasons import RETRYABLE, reason_literals, retryable_for

_SRC = Path(__file__).resolve().parents[2] / "src" / "zipsai"


def test_known_codes_match_the_table():
    assert retryable_for(None) is False
    assert retryable_for("EMBEDDING_FAILED") is True
    assert retryable_for("CHUNK_TOO_LONG") is False
    assert retryable_for("EMBEDDING_REQUEST_INVALID") is False
    assert retryable_for("PARSE_FAILED") is False
    assert retryable_for("STALE_UPDATE") is False


def test_unknown_code_is_not_retryable(caplog):
    with caplog.at_level(logging.WARNING):
        assert retryable_for("NOT_A_CODE") is False

    assert any(record.message == "unknown_reason_code" for record in caplog.records)


def test_source_literals_are_in_the_table():
    missing = reason_literals(_SRC) - set(RETRYABLE)

    assert missing == set()
