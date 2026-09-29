import json
import logging

import pytest

from zipsai.errors import LlmTimeoutError
from zipsai.observability import (
    SERVICE_NAME,
    JsonFormatter,
    _ContextFilter,
    add_context,
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
    # 매핑에 없는 예외도 코드가 있어야 error_code 하나로 실패를 집계할 수 있다.
    assert record.error_code == "INTERNAL_ERROR"
    assert caught.value.stage == "embed"


def test_mapped_failure_carries_the_shared_error_code(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # 로그의 error_code와 응답 본문의 code가 같아야 한 필드로 이어 조회할 수 있다.
    with (
        caplog.at_level(logging.INFO),
        pytest.raises(LlmTimeoutError),
        stage("generate", logger),
    ):
        raise LlmTimeoutError("too slow")

    assert _records(caplog)[0].error_code == "MODEL_TIMEOUT"


def test_skipped_stage_is_distinguishable_from_a_missing_one(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        skipped("hybrid", logger, skip_reason="gate_blocked")

    record = _records(caplog)[0]
    assert record.outcome == "skipped"
    assert record.duration_ms == 0
    assert record.skip_reason == "gate_blocked"


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
    # 포매터에서 읽으면 출력이 늦어질 때 turn_id를 잃는다. 여기서는 블록을
    # 빠져나온 뒤에 포맷해도 남아 있어야 한다.
    logger.addFilter(_ContextFilter())
    try:
        with (
            caplog.at_level(logging.INFO),
            bind(turn_id="t-1"),
            stage("encode", logger),
        ):
            pass
    finally:
        logger.filters.clear()

    payload = json.loads(JsonFormatter().format(_records(caplog)[0]))
    assert payload["turn_id"] == "t-1"
    assert payload["event"] == "stage_done"
    assert payload["timestamp"].endswith("+00:00")
    # 백엔드·embedding 로그와 합쳐 볼 때 어느 서비스 줄인지 가리는 필드다.
    assert payload["service"] == SERVICE_NAME


def test_context_added_mid_block_reaches_the_later_lines(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # 라우팅 결과는 bind에 들어갈 때는 아직 모른다. 그 뒤 단계들이 물려받아야 한다.
    logger.addFilter(_ContextFilter())
    try:
        with caplog.at_level(logging.INFO), bind(intent_route=None):
            add_context(intent_route="knowledge")
            with stage("generate", logger):
                pass
    finally:
        logger.filters.clear()

    assert _records(caplog)[0].intent_route == "knowledge"


def test_first_call_is_cold_and_the_next_is_not() -> None:
    # 첫 작업은 docling·인코더를 올리느라 느리다. 구분이 없으면 파싱이 느린 걸로 읽힌다.
    from zipsai import observability

    observability._warm = set()
    assert is_cold("converse") is True
    assert is_cold("converse") is False


def test_each_scope_gets_its_own_first_call() -> None:
    # 플래그가 하나면 먼저 온 질의가 색인의 첫 판을 가려 docling 로딩을 놓친다.
    from zipsai import observability

    observability._warm = set()
    assert is_cold("converse") is True
    assert is_cold("indexing") is True
