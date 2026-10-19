import re
from dataclasses import dataclass
from typing import Literal

from zipsai.contracts.converse import ComplaintDraft, ImageAnalysis

Axis = Literal["safety", "progression", "function", "scope"]
UrgencyLevel = Literal["urgent", "priority", "normal"]
Confidence = Literal["high", "low"]

_AXIS_WEIGHTS: dict[Axis, int] = {
    "safety": 45,
    "progression": 25,
    "function": 15,
    "scope": 15,
}

_URGENT_THRESHOLD = 70
_PRIORITY_THRESHOLD = 40


def _compile(*patterns: str) -> tuple[re.Pattern[str], ...]:
    return tuple(re.compile(p) for p in patterns)


@dataclass(frozen=True)
class AxisPatterns:
    strong: tuple[re.Pattern[str], ...]
    weak: tuple[re.Pattern[str], ...]


SAFETY_PATTERNS = AxisPatterns(
    strong=_compile(
        r"불(이|길)?\s*(나|붙|번지)",
        r"화재",
        r"타는\s*냄새",
        r"불꽃",
        r"감전",
        r"노출된?\s*전선",
        r"전선.{0,5}(벗겨|드러나)",
        r"가스\s*(냄새|누출|새)",
        r"붕괴",
        r"무너",
    ),
    weak=_compile(
        r"위험(할|해)\s*보",
        r"감전.{0,6}(될|수도)",
        r"탈\s*것\s*같",
        r"연기",
    ),
)
# ponytail: 부정 처리 없음. 검증 단계에서 "없다고 했는데 올라간" 오탐 사례가
# 실제로 쌓이면 어절 단위 윈도 부정 검사로 보강.

PROGRESSION_PATTERNS = AxisPatterns(
    strong=_compile(
        r"계속\s*(새|흐르|번지|심해|악화)",
        r"점점\s*(심해|악화|커지|번지)",
        r"빠르게\s*(번지|악화|퍼지)",
    ),
    weak=_compile(
        r"아직도",
        r"여전히",
        r"또\s*(새|흐르)",
    ),
)

FUNCTION_PATTERNS = AxisPatterns(
    strong=_compile(
        r"단수",
        r"정전",
        r"난방.{0,4}(안\s*되|끊기|고장)",
        r"배수.{0,4}(안\s*되|막히|역류)",
        r"전기.{0,4}(안\s*들어오|끊기|나가)",
    ),
    weak=_compile(
        r"(물|난방|전기).{0,4}(약해|미지근|불안정)",
    ),
)

SCOPE_PATTERNS = AxisPatterns(
    strong=_compile(
        r"여러\s*세대",
        r"전체\s*(세대|층)",
        r"공용",
    ),
    weak=_compile(
        r"복도",
        r"계단",
        r"옥상",
        r"지하\s*주차장",
        r"엘리베이터",
    ),
)

_AXIS_PATTERNS: dict[Axis, AxisPatterns] = {
    "safety": SAFETY_PATTERNS,
    "progression": PROGRESSION_PATTERNS,
    "function": FUNCTION_PATTERNS,
    "scope": SCOPE_PATTERNS,
}

# 안전 축 만점 조건(화재·감전·가스누출·붕괴)과 상향 규칙의 "즉시 위해" 조건이
# 점수 스펙에서 같은 목록을 가리키므로, 1차 구현에서는 패턴 집합을 하나로 합친다.
_ESCALATION_PATTERNS = SAFETY_PATTERNS.strong


@dataclass(frozen=True)
class UrgencyResult:
    score: int
    level: UrgencyLevel
    is_urgent: bool
    confidence: Confidence
    unknown_axes: list[Axis]
    escalated: bool


def _axis_text(draft: ComplaintDraft, image_analysis: ImageAnalysis | None) -> str:
    parts = [draft.symptom or ""]
    if image_analysis:
        for image in image_analysis.images:
            if image.summary:
                parts.append(image.summary)
            if image.ocr_text:
                parts.append(image.ocr_text)
    return " ".join(parts)


def _scope_text(draft: ComplaintDraft, image_analysis: ImageAnalysis | None) -> str:
    return f"{draft.location or ''} {_axis_text(draft, image_analysis)}"


def _axis_score(axis: Axis, text: str) -> int | None:
    """None은 unknown이다. 0점(위험 없음 확인)은 부정 처리가 없는 1차 구현에서는
    도달하지 않는 분기라 여기서 만들지 않는다."""
    patterns = _AXIS_PATTERNS[axis]
    weight = _AXIS_WEIGHTS[axis]
    if any(pattern.search(text) for pattern in patterns.strong):
        return weight
    if any(pattern.search(text) for pattern in patterns.weak):
        return weight // 2
    return None


def _grade(score: int, unknown_axes: list[Axis]) -> UrgencyLevel:
    forced_priority = "safety" in unknown_axes or "progression" in unknown_axes
    if score >= _URGENT_THRESHOLD:
        return "urgent"
    if score >= _PRIORITY_THRESHOLD or forced_priority:
        return "priority"
    return "normal"


def score_urgency(
    draft: ComplaintDraft, image_analysis: ImageAnalysis | None
) -> UrgencyResult:
    general_text = _axis_text(draft, image_analysis)
    axis_texts: dict[Axis, str] = {
        "safety": general_text,
        "progression": general_text,
        "function": general_text,
        "scope": _scope_text(draft, image_analysis),
    }

    scores = {axis: _axis_score(axis, text) for axis, text in axis_texts.items()}
    unknown_axes = [axis for axis, value in scores.items() if value is None]
    total = sum(value for value in scores.values() if value is not None)

    escalated = any(pattern.search(general_text) for pattern in _ESCALATION_PATTERNS)
    level: UrgencyLevel = "urgent" if escalated else _grade(total, unknown_axes)

    return UrgencyResult(
        score=total,
        level=level,
        is_urgent=level == "urgent",
        confidence="low" if unknown_axes else "high",
        unknown_axes=unknown_axes,
        escalated=escalated,
    )
