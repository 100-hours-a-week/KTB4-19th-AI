import re
from dataclasses import dataclass

CHUNK_SIZE = 400
CHUNK_OVERLAP = 50

_ARTICLE_RE = re.compile(r"^제\s*\d+\s*조(?:의\s*\d+)?(?=[\s(（:]|$)")
_MARKDOWN_RE = re.compile(r"^#{1,6}\s+")
_ITEM_RE = re.compile(r"^(?:[①-⑳]|제\s*\d+\s*항)")
_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


@dataclass(frozen=True)
class Chunk:
    text: str
    page: int
    section: str | None


@dataclass(frozen=True)
class _Line:
    start: int
    end: int
    content: str


def _lines(text: str) -> list[_Line]:
    result: list[_Line] = []
    start = 0
    for line in text.splitlines(keepends=True):
        result.append(_Line(start, start + len(line), line.rstrip("\r\n")))
        start += len(line)
    if start < len(text):
        result.append(_Line(start, len(text), text[start:]))
    return result


def _section_name(line: str, article: bool) -> str:
    stripped = line.strip()
    if article:
        return stripped
    return _MARKDOWN_RE.sub("", stripped, count=1).strip()


def _section_ranges(text: str, lines: list[_Line]) -> list[tuple[int, int, str | None]]:
    article_lines = [line for line in lines if _ARTICLE_RE.match(line.content.strip())]
    if article_lines:
        boundaries = [
            (line.start, _section_name(line.content, True)) for line in article_lines
        ]
    else:
        heading_lines = [
            line for line in lines if _MARKDOWN_RE.match(line.content.strip())
        ]
        if not heading_lines:
            return [(0, len(text), None)]
        boundaries = [
            (line.start, _section_name(line.content, False)) for line in heading_lines
        ]

    sections: list[tuple[int, int, str | None]] = []
    first_start = boundaries[0][0]
    if first_start:
        sections.append((0, first_start, None))
    for index, (start, name) in enumerate(boundaries):
        end = boundaries[index + 1][0] if index + 1 < len(boundaries) else len(text)
        sections.append((start, end, name))
    return sections


def _trim_range(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _split_sentences(text: str, start: int, end: int) -> list[tuple[int, int, bool]]:
    ranges: list[tuple[int, int, bool]] = []
    cursor = start
    for match in _SPLIT_RE.finditer(text, start, end):
        if match.end() > cursor:
            ranges.append((cursor, match.end(), False))
        cursor = match.end()
    if cursor < end:
        ranges.append((cursor, end, False))
    return ranges or [(start, end, False)]


def _atoms(
    text: str, lines: list[_Line], start: int, end: int
) -> list[tuple[int, int, bool]]:
    table_spans: list[tuple[int, int]] = []
    index = 0
    relevant = [line for line in lines if start <= line.start < end]
    while index < len(relevant):
        line = relevant[index]
        if not line.content.startswith("|"):
            index += 1
            continue
        table_start = max(start, line.start)
        table_end = min(end, line.end)
        index += 1
        while index < len(relevant) and relevant[index].content.startswith("|"):
            table_end = min(end, relevant[index].end)
            index += 1
        table_spans.append((table_start, table_end))

    if not table_spans:
        return _split_sentences(text, start, end)

    atoms: list[tuple[int, int, bool]] = []
    cursor = start
    for table_start, table_end in table_spans:
        if cursor < table_start:
            atoms.extend(_split_sentences(text, cursor, table_start))
        atoms.append((table_start, table_end, True))
        cursor = table_end
    if cursor < end:
        atoms.extend(_split_sentences(text, cursor, end))
    return atoms


def _item_groups(
    text: str, lines: list[_Line], start: int, end: int
) -> list[tuple[int, int]]:
    item_starts = [
        line.start
        for line in lines
        if start <= line.start < end and _ITEM_RE.match(line.content.strip())
    ]
    if not item_starts:
        return [(start, end)]

    groups: list[tuple[int, int]] = []
    for index, item_start in enumerate(item_starts):
        group_start = start if index == 0 else item_start
        item_end = item_starts[index + 1] if index + 1 < len(item_starts) else end
        groups.append((group_start, item_end))
    return groups


def _group_ranges(
    text: str,
    lines: list[_Line],
    start: int,
    end: int,
    first_capacity: int,
) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    current_start: int | None = None
    current_end: int | None = None
    capacity = first_capacity

    def flush() -> None:
        nonlocal current_start, current_end, capacity
        if current_start is not None and current_end is not None:
            ranges.append((current_start, current_end))
            current_start = current_end = None
            capacity = CHUNK_SIZE - CHUNK_OVERLAP

    for atom_start, atom_end, is_table in _atoms(text, lines, start, end):
        if atom_start >= atom_end:
            continue
        if is_table:
            if current_start is not None and (
                current_end - current_start + atom_end - atom_start > capacity
            ):
                flush()
            if atom_end - atom_start > capacity:
                flush()
                ranges.append((atom_start, atom_end))
                capacity = CHUNK_SIZE - CHUNK_OVERLAP
            elif current_start is None:
                current_start, current_end = atom_start, atom_end
            else:
                current_end = atom_end
            continue

        remaining_start = atom_start
        while remaining_start < atom_end:
            if current_start is None:
                current_start, current_end = remaining_start, remaining_start
            room = capacity - (current_end - current_start)
            remaining_length = atom_end - remaining_start
            if remaining_length <= room:
                current_end = atom_end
                remaining_start = atom_end
                continue
            if current_end > current_start:
                flush()
                continue
            cut_end = current_start + capacity
            ranges.append((current_start, cut_end))
            remaining_start = cut_end
            current_start = current_end = None
            capacity = CHUNK_SIZE - CHUNK_OVERLAP

    flush()
    return ranges


def _page_for(text: str, page_map: list[int], start: int, end: int) -> int:
    position = start
    while position < end and text[position].isspace():
        position += 1
    if position >= end:
        position = start
    return page_map[position]


def chunk_pages(pages: list[dict[str, int | str]]) -> list[Chunk]:
    parts: list[str] = []
    page_map: list[int] = []
    for index, page in enumerate(pages):
        if index:
            parts.append("\n")
            page_map.append(page["page"])
        page_text = str(page["text"])
        parts.append(page_text)
        page_map.extend([page["page"]] * len(page_text))

    text = "".join(parts)
    if not text or not page_map:
        return []

    lines = _lines(text)
    chunks: list[Chunk] = []
    for section_start, section_end, section in _section_ranges(text, lines):
        section_start, section_end = _trim_range(text, section_start, section_end)
        if section_start >= section_end:
            continue
        if section_end - section_start <= CHUNK_SIZE:
            chunks.append(
                Chunk(
                    text=text[section_start:section_end],
                    page=_page_for(text, page_map, section_start, section_end),
                    section=section,
                )
            )
            continue

        section_chunk_count = 0
        previous_text = ""
        for group_start, group_end in _item_groups(
            text, lines, section_start, section_end
        ):
            group_start, group_end = _trim_range(text, group_start, group_end)
            if group_start >= group_end:
                continue
            first_capacity = (
                CHUNK_SIZE if section_chunk_count == 0 else CHUNK_SIZE - CHUNK_OVERLAP
            )
            for base_start, base_end in _group_ranges(
                text, lines, group_start, group_end, first_capacity
            ):
                base_text = text[base_start:base_end]
                if not base_text.strip():
                    continue
                overlap = (
                    "" if section_chunk_count == 0 else previous_text[-CHUNK_OVERLAP:]
                )
                chunks.append(
                    Chunk(
                        text=overlap + base_text,
                        page=_page_for(text, page_map, base_start, base_end),
                        section=section,
                    )
                )
                previous_text = chunks[-1].text
                section_chunk_count += 1
    return chunks
