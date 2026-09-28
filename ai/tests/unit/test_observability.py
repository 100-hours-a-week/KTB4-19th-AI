import json
import logging

import pytest

from zipsai.observability import (
    JsonFormatter,
    _ContextFilter,
    bind,
    collect_timings,
    is_cold,
    skipped,
    stage,
)

logger = logging.getLogger("zipsai.test")


def _records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [record for record in caplog.records if record.name == "zipsai.test"]


def test_stage_logs_duration_even_when_nothing_fails(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # 실패만 남기면 "정상인데 느린" 구간을 영영 못 찾는다.
    with caplog.at_level(logging.INFO), stage("parse", logger) as step:
        step["pages"] = 3

    record = _records(caplog)[0]
    assert record.stage == "parse"
    assert record.outcome == "ok"
    assert record.pages == 3
    assert record.duration_ms >= 0


def test_failing_stage_names_itself_on_the_exception(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # 작업 요약 한 줄에 어느 단계에서 죽었는지를 담으려면 예외가 단계를 실어야 한다.
    with (
        caplog.at_level(logging.INFO),
        pytest.raises(ValueError) as caught,
        stage("embed", logger),
    ):
        raise ValueError("boom")

    record = _records(caplog)[0]
    assert record.outcome == "fail"
    assert record.error_type == "ValueError"
    assert caught.value.stage == "embed"


def test_skipped_stage_is_distinguishable_from_a_missing_one(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        skipped("hybrid", logger)

    record = _records(caplog)[0]
    assert record.outcome == "skipped"
    assert record.duration_ms == 0


def test_timings_are_collected_for_the_request_summary(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO), collect_timings() as timings:
        with stage("gate", logger):
            pass
        with stage("hybrid", logger):
            pass

    assert sorted(timings) == ["gate", "hybrid"]


def test_bound_fields_are_attached_when_the_line_is_made(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # 포매터에서 읽으면 출력이 늦어질 때 trace_id를 잃는다. 여기서는 블록을
    # 빠져나온 뒤에 포맷해도 남아 있어야 한다.
    logger.addFilter(_ContextFilter())
    try:
        with (
            caplog.at_level(logging.INFO),
            bind(trace_id="t-1"),
            stage("encode", logger),
        ):
            pass
    finally:
        logger.filters.clear()

    payload = json.loads(JsonFormatter().format(_records(caplog)[0]))
    assert payload["trace_id"] == "t-1"
    assert payload["event"] == "stage_done"
    assert payload["ts"].endswith("+00:00")


def test_first_call_is_cold_and_the_next_is_not() -> None:
    # 첫 작업은 docling·인코더를 올리느라 느리다. 구분이 없으면 파싱이 느린 걸로 읽힌다.
    from zipsai import observability

    observability._warm = False
    assert is_cold() is True
    assert is_cold() is False
