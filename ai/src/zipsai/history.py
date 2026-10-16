from functools import lru_cache

import tiktoken

from zipsai.contracts.converse import HistoryTurn

HISTORY_WINDOW = 5
INTENT_HISTORY_WINDOW = 6
INTENT_HISTORY_TOKEN_LIMIT = 500


@lru_cache(maxsize=1)
def _intent_history_encoding() -> tiktoken.Encoding:
    # shortcut: cl100k_base approximates the remote VLM tokenizer; use its tokenizer if available.
    return tiktoken.get_encoding("cl100k_base")


def _format_turn(turn: HistoryTurn) -> str:
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
    return f"{turn.role}: {content or '[내용 없음]'}"


def format_history(history: list[HistoryTurn]) -> str:
    recent = history[-HISTORY_WINDOW:]
    lines = [_format_turn(turn) for turn in recent]
    return "\n".join(lines) or "없음"


def format_intent_history(history: list[HistoryTurn]) -> str:
    lines = [_format_turn(turn) for turn in history[-INTENT_HISTORY_WINDOW:]]
    if not lines:
        return "없음"
    encoding = _intent_history_encoding()
    while (
        len(encoding.encode("\n".join(lines), disallowed_special=()))
        > INTENT_HISTORY_TOKEN_LIMIT
    ):
        if len(lines) > 1:
            lines.pop(0)
        else:
            tokens = encoding.encode(lines[0], disallowed_special=())[
                :INTENT_HISTORY_TOKEN_LIMIT
            ]
            lines[0] = encoding.decode_bytes(tokens).decode("utf-8", errors="ignore")
    return "\n".join(lines) or "없음"
