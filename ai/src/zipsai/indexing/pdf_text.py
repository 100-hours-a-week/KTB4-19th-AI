"""본문 PDF의 글자층과 표를 본다. 선이 있는 표는 pdfplumber, 판단만 여기서 한다."""

import re
import statistics
from dataclasses import dataclass

import pdfplumber

from zipsai.errors import PdfParseError

# 색인 검증의 garbage_ratio와 같은 식이다. 평가 코드를 서비스가 가져오지 않는다.
_CRUMB = re.compile(r"(?<![A-Za-z])[A-Za-z]{1,3}(?![A-Za-z])")
_CID = re.compile(r"\(cid:\d+\)")

GARBAGE_RATIO_MAX = 0.2
MIN_CHARS_PER_PAGE = 40
CID_RATIO_MAX = 0.05
BORDERLESS_MIN_COLUMNS = 3
BORDERLESS_MIN_GAP = 12.0
BORDERLESS_MIN_LINES = 3
LINE_CLUSTER_GAP = 3.0

_TABLE_SETTINGS = {"vertical_strategy": "lines", "horizontal_strategy": "lines"}


@dataclass(frozen=True)
class PdfInspection:
    page_texts: list[str]
    markdown_pages: list[str]
    tables_found: int
    ragged: bool
    borderless: bool


def _wansung(syllable: str) -> bool:
    try:
        syllable.encode("euc-kr")
    except UnicodeEncodeError:
        return False
    return True


def garbage_ratio(text: str) -> float:
    """완성형 밖 음절 비율 + 한글 문서에 섞인 3글자 이하 영문 토막 비율."""

    syllables = [char for char in text if "\uac00" <= char <= "\ud7a3"]
    if not syllables:
        return 0.0
    bad = sum(1 for syllable in syllables if not _wansung(syllable))
    crumbs = sum(len(token) for token in _CRUMB.findall(text))
    return bad / len(syllables) + crumbs / len(syllables)


def _cid_ratio(text: str) -> float:
    if not text:
        return 0.0
    cid_chars = sum(len(match.group()) for match in _CID.finditer(text))
    return cid_chars / len(text)


def classify_text_layer(page_texts: list[str]) -> str:
    """쪽 하나의 짧은 글은 문서 전체를 깨짐으로 보지 않는다.

    글자가 0인 쪽이 하나라도 있으면 none이 broken보다 우선한다.
    """

    if not page_texts:
        return "none"
    lengths = [len(text.strip()) for text in page_texts]
    if any(length == 0 for length in lengths):
        return "none"
    document = "\n".join(page_texts)
    if garbage_ratio(document) > GARBAGE_RATIO_MAX:
        return "broken"
    if statistics.median(lengths) < MIN_CHARS_PER_PAGE:
        return "broken"
    if _cid_ratio(document) > CID_RATIO_MAX:
        return "broken"
    return "ok"


def _layer_text(page: object) -> str:
    extracted = page.extract_text() or ""
    if extracted.strip():
        return extracted
    chars = page.chars or []
    if not chars:
        return ""
    return "".join(char.get("text") or "" for char in chars)


def _clean_rows(rows: list[list[object]]) -> list[list[str]]:
    cleaned: list[list[str]] = []
    for row in rows:
        cells: list[str] = []
        for cell in row:
            text = "" if cell is None else str(cell)
            cells.append(text.replace("\n", " ").replace("|", " ").strip())
        if any(cells):
            cleaned.append(cells)
    return cleaned


def table_is_ragged(rows: list[list[str]]) -> bool:
    return len({len(row) for row in rows}) > 1


def _markdown_table(rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(rows[0]) + " |",
        "| " + " | ".join("---" for _ in rows[0]) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows[1:])
    return "\n".join(lines)


def _outside_lines(page: object, bboxes: list[tuple]) -> list[tuple[float, str]]:
    kept = []
    for char in page.chars or []:
        center_x = (char["x0"] + char["x1"]) / 2
        center_y = (char["top"] + char["bottom"]) / 2
        inside = any(
            box[0] <= center_x <= box[2] and box[1] <= center_y <= box[3]
            for box in bboxes
        )
        if not inside:
            kept.append(char)
    if not kept:
        return []
    kept.sort(key=lambda char: (char["top"], char["x0"]))
    groups: list[list[dict]] = [[kept[0]]]
    for char in kept[1:]:
        if abs(char["top"] - groups[-1][-1]["top"]) <= LINE_CLUSTER_GAP:
            groups[-1].append(char)
        else:
            groups.append([char])
    lines: list[tuple[float, str]] = []
    for group in groups:
        group.sort(key=lambda char: char["x0"])
        parts: list[str] = []
        previous = None
        for char in group:
            if previous is not None and char["x0"] - previous["x1"] > 1.5:
                parts.append(" ")
            parts.append(char.get("text") or "")
            previous = char
        text = "".join(parts).strip()
        if text:
            lines.append((min(char["top"] for char in group), text))
    return lines


def _column_lines(page: object, bboxes: list[tuple]) -> int:
    kept = []
    for char in page.chars or []:
        center_x = (char["x0"] + char["x1"]) / 2
        center_y = (char["top"] + char["bottom"]) / 2
        inside = any(
            box[0] <= center_x <= box[2] and box[1] <= center_y <= box[3]
            for box in bboxes
        )
        if not inside:
            kept.append(char)
    if not kept:
        return 0
    kept.sort(key=lambda char: (char["top"], char["x0"]))
    groups: list[list[dict]] = [[kept[0]]]
    for char in kept[1:]:
        if abs(char["top"] - groups[-1][-1]["top"]) <= LINE_CLUSTER_GAP:
            groups[-1].append(char)
        else:
            groups.append([char])
    count = 0
    for group in groups:
        group.sort(key=lambda char: char["x0"])
        columns = 1
        previous = group[0]
        for char in group[1:]:
            if char["x0"] - previous["x1"] >= BORDERLESS_MIN_GAP:
                columns += 1
            previous = char
        if columns >= BORDERLESS_MIN_COLUMNS:
            count += 1
    return count


def inspect_pdf(path: object) -> PdfInspection:
    page_texts: list[str] = []
    markdown_pages: list[str] = []
    tables_found = 0
    ragged = False
    column_lines = 0
    with pdfplumber.open(path) as document:
        if not document.pages:
            raise PdfParseError(f"Document has no pages: {path}")
        for page in document.pages:
            page_texts.append(_layer_text(page))
            found = page.find_tables(table_settings=_TABLE_SETTINGS) or []
            events: list[tuple[float, int, str]] = []
            bboxes: list[tuple] = []
            for table in found:
                cleaned = _clean_rows(table.extract() or [])
                if not cleaned:
                    continue
                ragged = ragged or table_is_ragged(cleaned)
                tables_found += 1
                box = tuple(table.bbox)
                bboxes.append(box)
                events.append((box[1], 1, _markdown_table(cleaned)))
            column_lines += _column_lines(page, bboxes)
            for top, text in _outside_lines(page, bboxes):
                events.append((top, 0, text))
            events.sort(key=lambda item: (item[0], item[1]))
            markdown_pages.append("\n".join(item[2] for item in events))
    borderless = tables_found == 0 and column_lines >= BORDERLESS_MIN_LINES
    return PdfInspection(
        page_texts=page_texts,
        markdown_pages=markdown_pages,
        tables_found=tables_found,
        ragged=ragged,
        borderless=borderless,
    )
