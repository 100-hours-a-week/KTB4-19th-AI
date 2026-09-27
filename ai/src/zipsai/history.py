from zipsai.contracts.converse import HistoryTurn


HISTORY_WINDOW = 5


def format_history(history: list[HistoryTurn]) -> str:
    recent = history[-HISTORY_WINDOW:]
    if not recent:
        return "없음"
    return "\n".join(f"{turn.role}: {turn.text or '[이미지 첨부]'}" for turn in recent)
