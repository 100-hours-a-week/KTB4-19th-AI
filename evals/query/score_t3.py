"""T3 집계. 네트워크와 모델 호출은 없다.

C1은 route가 knowledge인 문항 수 ÷ 전체 수다.
실패한 호출은 분자에서 빠지고 분모에는 남는다.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RouteRecord:
    item_id: str
    route: str | None
    failed: bool


def aggregate(rows: list[RouteRecord]) -> dict[str, object]:
    knowledge = [row for row in rows if row.route == "knowledge" and not row.failed]
    complaint = [row for row in rows if row.route == "complaint" and not row.failed]
    clarify = [row for row in rows if row.route == "clarify" and not row.failed]
    failed = [row for row in rows if row.failed]
    return {
        "n": len(rows),
        "knowledge": len(knowledge),
        "complaint": len(complaint),
        "clarify": len(clarify),
        "failed": len(failed),
        "C1": (len(knowledge) / len(rows)) if rows else None,
    }


def _fmt(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def format_report(metrics: dict[str, object]) -> str:
    lines = [
        "T3 길 고르기",
        (
            f"C1 knowledge 재현율 {_fmt(metrics['C1'])}  "
            f"({metrics['knowledge']}/{metrics['n']})  목표 0.950"
        ),
        (
            f"knowledge {metrics['knowledge']} · "
            f"complaint {metrics['complaint']} · "
            f"clarify {metrics['clarify']} · "
            f"실패 {metrics['failed']}"
        ),
    ]
    return "\n".join(lines) + "\n"
