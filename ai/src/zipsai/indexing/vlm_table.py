"""VLM이 적은 " | " 줄을 청커가 읽는 표로 바꾼다. Paddle 출력에는 쓰지 않는다."""

from __future__ import annotations

import re

PIPE = " | "
_SEP = re.compile(r"^\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?$")


def cells_of(line: str) -> list[str]:
    parts = [part.strip() for part in line.split("|")]
    if parts and parts[0] == "":
        parts = parts[1:]
    if parts and parts[-1] == "":
        parts = parts[:-1]
    return parts


def markdown_row(line: str) -> str:
    return "| " + " | ".join(cells_of(line)) + " |"


def separator_row(cell_count: int) -> str:
    return "| " + " | ".join(["---"] * cell_count) + " |"


def is_separator(line: str) -> bool:
    return _SEP.fullmatch(line.strip()) is not None


def already_markdown(block: list[str]) -> bool:
    return all(line.strip().startswith("|") for line in block) and any(
        is_separator(line) for line in block
    )


def pipe_runs(lines: list[str]) -> list[tuple[int, int]]:
    runs = []
    index = 0
    while index < len(lines):
        if PIPE not in lines[index]:
            index += 1
            continue
        start = index
        while index < len(lines) and PIPE in lines[index]:
            index += 1
        if index - start >= 2:
            runs.append((start, index))
    return runs


def name_value_line(line: str) -> str:
    left, right = cells_of(line)
    return f"{left}: {right}"


def pair_table(block: list[str]) -> bool:
    return all(len(cells_of(line)) == 2 for line in block)


def markdown_block(block: list[str]) -> list[str]:
    rendered = [markdown_row(block[0]), separator_row(len(cells_of(block[0])))]
    rendered.extend(markdown_row(line) for line in block[1:])
    return rendered


def render_block(block: list[str]) -> list[str]:
    if already_markdown(block):
        return list(block)
    if pair_table(block):
        return [name_value_line(line) for line in block]
    rendered: list[str] = []
    wide: list[str] = []

    def flush() -> None:
        if len(wide) >= 2:
            rendered.extend(markdown_block(wide))
        elif len(wide) == 1:
            rendered.append(wide[0])
        wide.clear()

    for line in block:
        if len(cells_of(line)) == 2:
            flush()
            rendered.append(name_value_line(line))
            continue
        wide.append(line)
    flush()
    return rendered


def pipe_runs_to_markdown(text: str) -> str:
    """이어진 " | " 줄을 표로 바꾼 본문을 돌려준다. 나머지 줄은 그대로다."""

    lines = text.splitlines()
    blocks = {start: end for start, end in pipe_runs(lines)}
    out: list[str] = []
    index = 0
    while index < len(lines):
        end = blocks.get(index)
        if end is None:
            out.append(lines[index])
            index += 1
            continue
        out.extend(render_block(lines[index:end]))
        index = end
    suffix = "\n" if text.endswith("\n") else ""
    return "\n".join(out) + suffix
