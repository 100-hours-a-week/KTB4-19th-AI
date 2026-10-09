from zipsai.contracts.converse import HistoryTurn

HISTORY_WINDOW = 5


def format_history(history: list[HistoryTurn]) -> str:
    recent = history[-HISTORY_WINDOW:]
    lines = []
    for turn in recent:
        content = turn.text or ""
        observations = [
            " / ".join(value for value in (image.summary, image.ocr_text) if value)
            for image in (turn.images or [])
        ]
        observations = [value for value in observations if value]
        if observations:
            content = f"{content} [사진: {'; '.join(observations)}]".strip()
        elif turn.images:
            content = f"{content} [이미지 첨부]".strip()
        lines.append(f"{turn.role}: {content or '[내용 없음]'}")
    return "\n".join(lines) or "없음"
