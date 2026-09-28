from zipsai.contracts.converse import HistoryTurn
from zipsai.history import HISTORY_WINDOW, format_history


def _turn(index: int, text: str | None, role: str = "user") -> HistoryTurn:
    return HistoryTurn(message_id=f"msg-{index}", role=role, text=text, image_urls=[])


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
    formatted = format_history([_turn(0, None)])

    assert formatted == "user: [이미지 첨부]"


def test_format_history_labels_each_turn_with_its_role():
    formatted = format_history(
        [_turn(0, "물이 새요"), _turn(1, "어디에서 생긴 문제인가요?", role="assistant")]
    )

    assert formatted == "user: 물이 새요\nassistant: 어디에서 생긴 문제인가요?"
