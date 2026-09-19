import logging
import re
from collections import defaultdict
from dataclasses import dataclass

logger = logging.getLogger(__name__)

HOLD_RATIO = 0.30

_ARTICLE_RE = re.compile(r"^제\s*\d+\s*조(?:의\s*\d+)?(?=[\s(（:]|$)")
_MARKDOWN_RE = re.compile(r"^#{1,6}\s")
_ITEM_RE = re.compile(r"^(?:[①-⑳]|\d+\.|제\s*\d+\s*항)")
_PAGE_NUMBER_RE = re.compile(r"^(?:-\s*\d+\s*-|\d+\s*/\s*\d+|페이지\s*\d+|\d+)$")
_TOC_MARKER_RE = re.compile(r"^(?:목차|차례)\s*[:：]$")
_TOC_LEADER_RE = re.compile(r"(?:\.{2,}|·{2,}|…{2,})(?:\s*\d+)?\s*$")
_TOC_TRAILING_PAGE_RE = re.compile(r"\s+\d+\s*$")
_UNCHANGED_RE = re.compile(
    r"^(?:\d{4}년\s*\d{1,2}월\s*\d{1,2}일|"
    r"\d{1,4}호\s+\[(?:이름|전화번호)\]|\[(?:이름|전화번호)\])$"
)


@dataclass(frozen=True)
class CleaningResult:
    pages: list[dict[str, int | str]]
    removed_ratio: float


def _is_protected(line: str) -> bool:
    stripped = line.strip()
    return bool(
        _ARTICLE_RE.match(stripped)
        or _MARKDOWN_RE.match(stripped)
        or _ITEM_RE.match(stripped)
        or _UNCHANGED_RE.fullmatch(stripped)
    )


def _is_toc_marker(line: str) -> bool:
    return line == "목차" or line == "차례" or bool(_TOC_MARKER_RE.fullmatch(line))


def _is_toc_line(line: str) -> bool:
    return bool(_TOC_LEADER_RE.search(line) or _TOC_TRAILING_PAGE_RE.search(line))


def _normalize(text: str) -> str:
    return re.sub(
        r"\n{3,}", "\n\n", "\n".join(line.rstrip() for line in text.splitlines())
    )


def clean_pages(pages: list[dict[str, int | str]]) -> CleaningResult:
    repeated_pages: defaultdict[str, set[int | str]] = defaultdict(set)
    for page in pages:
        for line in str(page["text"]).splitlines():
            stripped = line.strip()
            if stripped:
                repeated_pages[stripped].add(page["page"])

    repeated_lines = {
        line for line, page_numbers in repeated_pages.items() if len(page_numbers) >= 3
    }
    cleaned_pages: list[dict[str, int | str]] = []
    nonspace_before = 0
    nonspace_after = 0

    for page in pages:
        text = str(page["text"])
        nonspace_before += sum(not character.isspace() for character in text)
        toc = False
        kept_lines: list[str] = []
        for line in text.splitlines():
            stripped = line.strip()
            protected = _is_protected(line)

            if _is_toc_marker(stripped):
                toc = True
                continue
            if toc:
                if _is_toc_line(stripped):
                    if protected:
                        kept_lines.append(line)
                    continue
                toc = False
            if not protected and (
                stripped in repeated_lines or _PAGE_NUMBER_RE.fullmatch(stripped)
            ):
                continue
            kept_lines.append(line)

        cleaned_text = _normalize("\n".join(kept_lines))
        nonspace_after += sum(not character.isspace() for character in cleaned_text)
        cleaned_page = dict(page)
        cleaned_page["text"] = cleaned_text
        cleaned_pages.append(cleaned_page)

    removed_ratio = (
        0.0 if nonspace_before == 0 else 1 - (nonspace_after / nonspace_before)
    )
    return CleaningResult(pages=cleaned_pages, removed_ratio=removed_ratio)


def apply_cleaning(job_id: str, pages: list[dict[str, int | str]]) -> CleaningResult:
    result = clean_pages(pages)
    if result.removed_ratio > HOLD_RATIO:
        # 너무 많이 지웠으면 정제를 포기한다. 쪽번호가 섞이는 편이
        # 본문이 잘린 채 색인되는 것보다 낫다.
        logger.info(
            "cleaning_skipped job_id=%s removed_ratio=%.3f",
            job_id,
            result.removed_ratio,
        )
        return CleaningResult(pages=[dict(page) for page in pages], removed_ratio=0.0)

    logger.info("cleaning job_id=%s removed_ratio=%.3f", job_id, result.removed_ratio)
    return result
