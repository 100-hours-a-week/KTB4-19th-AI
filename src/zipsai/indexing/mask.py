import logging
import re
from collections import Counter
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class PiiKind(str, Enum):
    EMAIL = "이메일"
    RESIDENT_ID = "주민등록번호"
    PHONE = "전화번호"
    ACCOUNT = "계좌번호"
    CAR_PLATE = "차량번호"
    UNIT_NAME = "이름"


@dataclass(frozen=True)
class PiiDetection:
    page: int
    kind: PiiKind
    matched: str
    start: int
    end: int


@dataclass(frozen=True)
class MaskingResult:
    pages: list[dict[str, int | str]]
    detections: list[PiiDetection]


_NOT_NAME = frozenset(
    {
        "관리실",
        "관리인",
        "관리소",
        "관리자",
        "경비실",
        "경비원",
        "사무실",
        "사무소",
        "입주민",
        "입주자",
        "거주자",
        "세입자",
        "임차인",
        "임대인",
        "세대",
        "세대주",
        "호실",
        "보일러",
        "계단",
        "복도",
        "옥상",
        "현관",
        "창고",
        "화장실",
        "앞에",
        "옆에",
        "뒤에",
        "아래",
        "위에",
        "근처",
        "또는",
        "그리고",
        "내지",
        "사이",
        "전체",
        "모두",
        "전부",
        "각각",
        "기준",
        "이상",
        "이하",
        "부터",
        "까지",
        "이전",
        "이후",
        "포함",
        "제외",
        "해당",
    }
)

_DETECTORS = (
    (
        PiiKind.EMAIL,
        re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    ),
    (
        PiiKind.RESIDENT_ID,
        # 뒷자리 첫 숫자: 1~4 내국인, 5~8 외국인 등록번호. 둘 다 탐지한다.
        re.compile(r"(?<![\d-])\d{6}-[1-8]\d{6}(?![\d-])"),
    ),
    (
        PiiKind.PHONE,
        re.compile(
            r"(?<![\d-])(?:01[016789][-\.\s]?\d{3,4}[-\.\s]?\d{4}|0\d{1,2}-\d{3,4}-\d{4})(?![\d-])"
        ),
    ),
    (
        PiiKind.ACCOUNT,
        re.compile(r"(?<![\d-])\d{2,6}-\d{2,6}(?:-\d{2,8})?(?![\d-])"),
    ),
    (
        PiiKind.CAR_PLATE,
        re.compile(
            r"(?<!\d)\d{2,3}[가나다라마거너더러머버서어저고노도로모보소오조구누두루무부수우주바사아자배하허호]\d{4}(?!\d)"
        ),
    ),
    (
        PiiKind.UNIT_NAME,
        re.compile(r"(?<![\d-])(\d{1,4}호)(\s+)([가-힣]{2,4})"),
    ),
)


def mask_pages(pages: list[dict[str, int | str]]) -> MaskingResult:
    masked_pages = [dict(page) for page in pages]
    detections: list[PiiDetection] = []

    for page in masked_pages:
        page_number = page["page"]
        text = page["text"]
        claimed: list[tuple[int, int]] = []
        replacements: list[tuple[PiiDetection, str]] = []

        for kind, detector in _DETECTORS:
            for match in detector.finditer(text):
                start, end = match.span()
                if any(
                    start < claimed_end and end > claimed_start
                    for claimed_start, claimed_end in claimed
                ):
                    continue
                matched = match.group()
                if (
                    kind is PiiKind.ACCOUNT
                    and not 10 <= sum(char.isdigit() for char in matched) <= 14
                ):
                    continue
                if kind is PiiKind.UNIT_NAME and any(
                    match.group(3).startswith(not_name) for not_name in _NOT_NAME
                ):
                    continue

                detection = PiiDetection(
                    page=page_number,
                    kind=kind,
                    matched=matched,
                    start=start,
                    end=end,
                )
                replacement = (
                    f"{match.group(1)}{match.group(2)}[{kind.value}]"
                    if kind is PiiKind.UNIT_NAME
                    else f"[{kind.value}]"
                )
                claimed.append((start, end))
                replacements.append((detection, replacement))

        for detection, replacement in sorted(
            replacements, key=lambda item: item[0].start, reverse=True
        ):
            text = text[: detection.start] + replacement + text[detection.end :]
            detections.append(detection)
        page["text"] = text

    detections.sort(key=lambda detection: (detection.page, detection.start))
    return MaskingResult(pages=masked_pages, detections=detections)


def apply_masking(job_id: str, pages: list[dict[str, int | str]]) -> MaskingResult:
    # 탐지돼도 멈추지 않는다. 라벨로 치환된 본문을 그대로 색인하고
    # 무엇이 걸렸는지는 로그로만 남긴다.
    result = mask_pages(pages)
    counts = Counter(detection.kind.value for detection in result.detections)
    logger.info("pii_masking job_id=%s counts=%s", job_id, dict(counts))
    return result
