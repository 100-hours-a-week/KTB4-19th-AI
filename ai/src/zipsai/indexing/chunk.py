import os
import re
import unicodedata
from dataclasses import dataclass

CHUNK_SIZE = 400
CHUNK_OVERLAP = 50
# 표는 이 글자 수까지만 묶는다. 512토큰인지는 테스트가 확인한다.
TABLE_CHUNK_CHARS = 400
TABLE_CHUNK_FORMAT = "fields"

_SEP = re.compile(r"^\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?$")
_DASHES = str.maketrans({"–": "-", "—": "-", "―": "-", "−": "-"})

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


def _table_format() -> str:
    if os.environ.get("TABLE_CHUNK_FORMAT", TABLE_CHUNK_FORMAT) == "fields":
        return "fields"
    return "markdown"


def _split_cells(line: str) -> list[str]:
    raw = line.strip().removeprefix("|").removesuffix("|")
    return [cell.strip() for cell in raw.split("|")]


def _fold_cell(cell: str) -> str:
    folded = unicodedata.normalize("NFKC", cell.translate(_DASHES))
    return "".join(char for char in folded if not char.isspace())


def _header_key(lines: list[str]) -> tuple[str, ...] | None:
    if len(lines) < 2 or not _SEP.fullmatch(lines[1].strip()):
        return None
    return tuple(_fold_cell(cell) for cell in _split_cells(lines[0]))


def _trailing_table(text: str) -> list[str] | None:
    lines = text.splitlines()
    end = len(lines)
    while end > 0 and not lines[end - 1].strip():
        end -= 1
    if end == 0 or not lines[end - 1].strip().startswith("|"):
        return None
    start = end
    while start > 0 and lines[start - 1].strip().startswith("|"):
        start -= 1
    return [lines[index].strip() for index in range(start, end)]


def _leading_table(text: str) -> tuple[list[str], str] | None:
    lines = text.splitlines()
    index = 0
    while index < len(lines) and not lines[index].strip():
        index += 1
    if index >= len(lines) or not lines[index].strip().startswith("|"):
        return None
    start = index
    while index < len(lines) and lines[index].strip().startswith("|"):
        index += 1
    suffix = "\n".join(lines[index:])
    block = [lines[line_index].strip() for line_index in range(start, index)]
    return block, suffix


def _body_without_header(lines: list[str], key: tuple[str, ...]) -> list[str]:
    body = []
    for line in lines[2:]:
        if tuple(_fold_cell(cell) for cell in _split_cells(line)) == key:
            continue
        body.append(line)
    return body


def _compose_continuation(body: list[str], suffix: str) -> str:
    suffix = suffix.lstrip("\n")
    if body and suffix:
        return "\n".join(body) + "\n" + suffix
    if body:
        return "\n".join(body)
    return suffix


def _join_continued_tables(pages: list[dict[str, int | str]]) -> list[dict[str, int | str]]:
    """쪽마다 반복된 같은 머리행을 빼고 한 표로 잇는다."""

    prepared = [dict(page) for page in pages]
    open_key: tuple[str, ...] | None = None
    for page in prepared:
        text = str(page.get("text", ""))
        if _trailing_table(text) is not None:
            text = text.rstrip()
            page["text"] = text
        continued = False
        if open_key is not None:
            leading = _leading_table(text)
            if leading is not None and _header_key(leading[0]) == open_key:
                body = _body_without_header(leading[0], open_key)
                page["text"] = _compose_continuation(body, leading[1])
                text = str(page["text"])
                continued = True
                if leading[1].strip():
                    open_key = _header_key(_trailing_table(leading[1]) or [])
                if not text.strip():
                    page["_drop"] = True
        if not continued and text.strip():
            open_key = _header_key(_trailing_table(text) or [])
    kept = []
    for page in prepared:
        if page.pop("_drop", False):
            continue
        kept.append(page)
    return kept


def _labels(section: str | None, caption: str | None) -> list[str]:
    labels: list[str] = []
    if section:
        labels.append(section)
    if not caption:
        return labels
    normalized = caption.strip()
    if _MARKDOWN_RE.match(normalized):
        normalized = _MARKDOWN_RE.sub("", normalized, count=1).strip()
    if normalized and normalized not in labels:
        labels.append(normalized)
    return labels


def _caption_before(lines: list[_Line], table_start: int) -> str | None:
    caption = None
    for line in lines:
        if line.start >= table_start:
            break
        content = line.content.strip()
        if content and not content.startswith("|"):
            caption = content
    return caption


def _fields_line(header: list[str], row: str) -> str:
    cells = _split_cells(row)
    width = max(len(header), len(cells))
    parts: list[str] = []
    for index in range(width):
        name = header[index] if index < len(header) else ""
        value = cells[index] if index < len(cells) else ""
        parts.append(f"{name}: {value}" if name else value)
    return " / ".join(parts)


def _render_piece(
    labels: list[str],
    header_lines: list[str],
    header_cells: list[str],
    rows: list[str],
    fmt: str,
) -> str:
    parts = list(labels)
    if fmt == "fields" and header_cells:
        parts.extend(_fields_line(header_cells, row) for row in rows)
    else:
        parts.extend(header_lines)
        parts.extend(rows)
    return "\n".join(parts)


def _pack_table(
    lines: list[_Line],
    start: int,
    end: int,
    section: str | None,
) -> list[tuple[str, int]]:
    rows = [
        (line.start, line.content.strip())
        for line in lines
        if start <= line.start < end and line.content.strip().startswith("|")
    ]
    if not rows:
        return []
    sep_at = next(
        (index for index, (_, content) in enumerate(rows) if _SEP.fullmatch(content)),
        None,
    )
    if sep_at is None or sep_at == 0:
        header_lines: list[str] = []
        header_cells: list[str] = []
        body = rows
    else:
        header_lines = [rows[sep_at - 1][1], rows[sep_at][1]]
        header_cells = _split_cells(rows[sep_at - 1][1])
        body = [
            row
            for index, row in enumerate(rows)
            if index not in {sep_at - 1, sep_at}
        ]
    labels = _labels(section, _caption_before(lines, start))
    fmt = _table_format()
    if not body:
        text = _render_piece(labels, header_lines, header_cells, [], fmt)
        return [(text, rows[0][0])] if text.strip() else []

    pieces: list[tuple[str, int]] = []
    current: list[tuple[int, str]] = []

    def render(chosen: list[tuple[int, str]]) -> str:
        return _render_piece(
            labels,
            header_lines,
            header_cells,
            [content for _, content in chosen],
            fmt,
        )

    for row in body:
        trial = current + [row]
        if current and len(render(trial)) > TABLE_CHUNK_CHARS:
            pieces.append((render(current), current[0][0]))
            current = [row]
        else:
            current = trial
    if current:
        pieces.append((render(current), current[0][0]))
    return pieces


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
    section: str | None,
) -> list[tuple]:
    ranges: list[tuple] = []
    current_start: int | None = None
    current_end: int | None = None
    capacity = first_capacity

    def flush() -> None:
        nonlocal current_start, current_end, capacity
        if current_start is not None and current_end is not None:
            ranges.append(("span", current_start, current_end))
            current_start = current_end = None
            capacity = CHUNK_SIZE - CHUNK_OVERLAP

    for atom_start, atom_end, is_table in _atoms(text, lines, start, end):
        if atom_start >= atom_end:
            continue
        if is_table:
            table_text = text[atom_start:atom_end].strip()
            # 표 조각은 앞뒤 본문 50자를 붙이지 않는다. 머리행을 반복하기 때문이다.
            if _table_format() == "fields" or len(table_text) > TABLE_CHUNK_CHARS:
                flush()
                for piece, pos in _pack_table(lines, atom_start, atom_end, section):
                    if piece.strip():
                        ranges.append(("piece", piece, pos))
                continue
            if current_start is not None and (
                current_end - current_start + atom_end - atom_start > capacity
            ):
                flush()
            if atom_end - atom_start > capacity:
                flush()
                ranges.append(("span", atom_start, atom_end))
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
            ranges.append(("span", current_start, cut_end))
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
    pages = _join_continued_tables(pages)
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
        if section_end - section_start <= CHUNK_SIZE and _table_format() != "fields":
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
            items = _group_ranges(
                text, lines, group_start, group_end, first_capacity, section
            )
            for index, item in enumerate(items):
                if item[0] == "piece":
                    _, piece, pos = item
                    if not str(piece).strip():
                        continue
                    chunks.append(
                        Chunk(
                            text=str(piece),
                            page=_page_for(text, page_map, pos, pos + 1),
                            section=section,
                        )
                    )
                    # 다음 본문이 표 끝을 겹쳐 가져가지 않게 비운다.
                    previous_text = ""
                    section_chunk_count += 1
                    continue
                _, base_start, base_end = item
                base_text = text[base_start:base_end]
                if not base_text.strip():
                    continue
                overlap = (
                    "" if section_chunk_count == 0 else previous_text[-CHUNK_OVERLAP:]
                )
                chunk_text = overlap + base_text
                caption = None
                if index + 1 < len(items) and items[index + 1][0] == "piece":
                    caption = _caption_before(lines, items[index + 1][2])
                # 글자 수로 지우지 않는다. 바로 뒤 표 라벨과 같은 제목만 뺀다.
                if caption and chunk_text.strip() == caption:
                    continue
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        page=_page_for(text, page_map, base_start, base_end),
                        section=section,
                    )
                )
                previous_text = chunk_text
                section_chunk_count += 1
    return chunks
