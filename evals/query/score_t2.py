"""T2 표본과 집계. 네트워크와 모델 호출은 없다.

표본은 질의-채점표 §4-3이다. 답있음 건물 45, 구역 15, 답없음 40.
표 B 20문항은 그 100개 안에 항상 넣는다. 나머지 자리는 id 순으로 채운다.
"""

from __future__ import annotations

from dataclasses import dataclass

from score import GoldItem, nfc

# 질의-채점표 §3의 대표 20문항. 표에 나온 순서를 유지한다.
TABLE_B: tuple[str, ...] = (
    "b001-q08",
    "b001-q13",
    "b001-q14",
    "b002-q09",
    "b004-q06",
    "b004-q09",
    "b006-q08",
    "b006-q18",
    "b001-q03",
    "b003-q03",
    "b007-q04",
    "b010-q04",
    "b001-q20",
    "b002-q18",
    "b004-q21",
    "b005-q14",
    "b005-q18",
    "b008-q22",
    "b003-q01",
    "b006-q02",
)

ANSWERABLE_BUILDING = 45
ANSWERABLE_ZONE = 15
UNANSWERABLE = 40


@dataclass(frozen=True)
class AnswerRecord:
    item_id: str
    question: str
    answerable: bool
    scope: str
    outcome: str
    reply: str
    answer_chars: int | None
    citation_ids: tuple[str, ...]
    retrieved_ids: tuple[str, ...]
    unsupported: tuple[str, ...] | None
    judge_failed: bool


def _take(pool: list[GoldItem], answerable: bool, scope: str | None, count: int) -> list[GoldItem]:
    chosen = [
        item
        for item in pool
        if item.answerable is answerable and (scope is None or item.scope == scope)
    ]
    chosen.sort(key=lambda item: item.id)
    if len(chosen) < count:
        label = scope or "unanswerable"
        raise ValueError(f"{label} 문항이 {len(chosen)}개라 {count}개를 채울 수 없습니다")
    return chosen[:count]


def select_sample(items: list[GoldItem]) -> list[GoldItem]:
    by_id = {item.id: item for item in items}
    missing = [item_id for item_id in TABLE_B if item_id not in by_id]
    if missing:
        raise ValueError(f"표 B 문항이 없습니다: {', '.join(missing)}")
    forced = [by_id[item_id] for item_id in TABLE_B]
    forced_ids = set(TABLE_B)
    rest = [item for item in items if item.id not in forced_ids]

    building = [item for item in forced if item.answerable and item.scope == "building"]
    zone = [item for item in forced if item.answerable and item.scope == "zone"]
    absent = [item for item in forced if not item.answerable]
    building += _take(rest, True, "building", ANSWERABLE_BUILDING - len(building))
    zone += _take(rest, True, "zone", ANSWERABLE_ZONE - len(zone))
    absent += _take(rest, False, None, UNANSWERABLE - len(absent))
    return building + zone + absent


def _ids(values: tuple[str, ...]) -> set[str]:
    return {nfc(value) for value in values if value}


def aggregate(rows: list[AnswerRecord]) -> dict[str, object]:
    answerable = [row for row in rows if row.answerable]
    declined = [row for row in answerable if row.outcome == "declined"]
    answered = [row for row in rows if row.outcome == "answered"]
    cited = [
        row
        for row in answered
        if _ids(row.citation_ids) == _ids(row.retrieved_ids) and _ids(row.retrieved_ids)
    ]
    lengths = [row.answer_chars for row in rows if row.answer_chars is not None]
    judged = [row for row in answered if row.unsupported is not None and not row.judge_failed]
    failed = [row for row in answered if row.judge_failed]
    claims = sum(len(row.unsupported or ()) for row in judged)
    return {
        "n": len(rows),
        "answerable": len(answerable),
        "G2": (len(declined) / len(answerable)) if answerable else None,
        "G2_declined": len(declined),
        "G5": (len(cited) / len(answered)) if answered else None,
        "G5_n": len(answered),
        "G6": max(lengths) if lengths else None,
        "G1": claims,
        "G1_judged": len(judged),
        "G1_failed": len(failed),
        "G1_provisional": bool(failed) or len(judged) != len(answered),
    }


def _fmt(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def format_report(metrics: dict[str, object], rows: list[AnswerRecord]) -> str:
    lines = [
        "T2 답변 채점",
        "R2가 목표에 못 미치면 아래 숫자는 적기만 하고 합격으로 판정하지 않습니다.",
        f"표본 {metrics['n']} · 답있음 {metrics['answerable']}",
        (
            f"G2 오회피율 {_fmt(metrics['G2'])}  "
            f"({metrics['G2_declined']}/{metrics['answerable']})  목표 0.050"
        ),
        f"G5 인용 정합성 {_fmt(metrics['G5'])}  (n={metrics['G5_n']})  목표 1.000",
        f"G6 답변 길이 최대 {_fmt(metrics['G6'])}자  목표 800",
    ]
    if metrics["G1_provisional"]:
        lines.append(
            f"G1 환각 {_fmt(metrics['G1'])}건  잠정  "
            f"(판정 {metrics['G1_judged']}  실패 {metrics['G1_failed']})  "
            "표 B 20건을 사람이 보기 전에는 확정하지 않습니다."
        )
    else:
        lines.append(
            f"G1 환각 {_fmt(metrics['G1'])}건  잠정  "
            "표 B 20건을 사람이 보기 전에는 확정하지 않습니다."
        )
    for row in rows:
        if row.outcome == "answered" and not row.judge_failed and row.unsupported:
            lines.append(f"G1 문항 {row.item_id}  {len(row.unsupported)}건")
    lines.append("표 B")
    by_id = {row.item_id: row for row in rows}
    for item_id in TABLE_B:
        row = by_id.get(item_id)
        if row is None:
            lines.append(f"{item_id}  표본에 없음")
            continue
        reply = " ".join(row.reply.split())
        lines.append(f"{item_id}  {row.outcome}  {reply}")
    return "\n".join(lines) + "\n"
