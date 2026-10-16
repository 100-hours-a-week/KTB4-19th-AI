import tiktoken

from zipsai.contracts.converse import HistoryTurn, ImageObservation
from zipsai.history import (
    HISTORY_WINDOW,
    INTENT_HISTORY_TOKEN_LIMIT,
    INTENT_HISTORY_WINDOW,
    format_history,
    format_intent_history,
)


def _turn(index: int, text: str | None, role: str = "user") -> HistoryTurn:
    return HistoryTurn(message_id=f"msg-{index}", role=role, text=text)


def test_format_history_returns_placeholder_when_empty():
    assert format_history([]) == "없음"


def test_format_history_keeps_only_the_most_recent_turns():
    history = [_turn(index, f"발화 {index}") for index in range(HISTORY_WINDOW + 1)]

    formatted = format_history(history)

    assert len(formatted.splitlines()) == HISTORY_WINDOW
    assert "발화 0" not in formatted
    assert "발화 1" in formatted
    assert f"발화 {HISTORY_WINDOW}" in formatted


def test_format_history_marks_a_turn_that_carries_only_an_image():
    turn = _turn(0, None).model_copy(
        update={"images": [ImageObservation(attachmentId=123, summary=None)]}
    )

    formatted = format_history([turn])

    assert formatted == "user: [이미지 첨부]"


def test_format_history_includes_image_analysis_for_user_turn():
    turn = _turn(0, "이거 어디 버려요?").model_copy(
        update={
            "images": [
                ImageObservation(
                    attachmentId=123, summary="돼지껍데기", ocrText="음식물"
                )
            ]
        }
    )

    assert (
        format_history([turn]) == "user: 이거 어디 버려요? [사진: 돼지껍데기 / 음식물]"
    )


def test_format_history_labels_each_turn_with_its_role():
    formatted = format_history(
        [_turn(0, "물이 새요"), _turn(1, "어디에서 생긴 문제인가요?", role="assistant")]
    )

    assert formatted == "user: 물이 새요\nassistant: 어디에서 생긴 문제인가요?"


def test_format_intent_history_keeps_six_recent_turns_in_order():
    history = [
        _turn(index, f"발화 {index}") for index in range(INTENT_HISTORY_WINDOW + 2)
    ]

    formatted = format_intent_history(history)

    assert len(formatted.splitlines()) == INTENT_HISTORY_WINDOW
    assert "발화 0" not in formatted
    assert "발화 1" not in formatted
    assert "발화 2" in formatted
    assert f"발화 {INTENT_HISTORY_WINDOW + 1}" in formatted
    assert formatted.splitlines() == [f"user: 발화 {index}" for index in range(2, 8)]


def test_format_intent_history_caps_tokens_and_keeps_the_newest_turn():
    history = [
        _turn(0, "오래된 내용 " * 500),
        _turn(1, "최신 내용 <|endoftext|> " * 1000),
    ]

    formatted = format_intent_history(history)

    token_count = len(
        tiktoken.get_encoding("cl100k_base").encode(formatted, disallowed_special=())
    )
    assert token_count <= INTENT_HISTORY_TOKEN_LIMIT
    assert formatted.startswith("user: ")
    assert "최신 내용" in formatted
    assert "오래된 내용" not in formatted
