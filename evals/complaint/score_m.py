"""M1·M2·M4·M5·M6·M8·M9 집계. 네트워크와 모델 호출은 없다.

위치·증상·유형은 앞뒤 공백을 뺀 완전 일치다. 빈 기대는 빈 결과만 맞다.
M8은 빈 기대를 결과가 채운 필드 수다.
실패한 호출은 맞춘 수에서 빠지고 분모에는 남는다.

M3(LLM 재질문 채택률)과 M7(추출 재시도율)은 로컬 38문항 벤치마크에 그 신호가
없어 여기서 집계하지 않는다(A-200-00 §2·§4) — M3는 운영 로그로만, M7은
로컬 러너 아직 미구현.
"""

from __future__ import annotations

from dataclasses import dataclass


def blank(value: object) -> str | None:
    if value is None:
        return None
    text = str(getattr(value, "value", value)).strip()
    return text or None


def same(got: object, expected: object) -> bool:
    return blank(got) == blank(expected)


@dataclass(frozen=True)
class ComplaintRow:
    item_id: str
    location_ok: bool
    symptom_ok: bool
    type_ok: bool
    location_invented: bool
    symptom_invented: bool
    issue_type: str | None
    failed: bool


@dataclass(frozen=True)
class PhotoRow:
    item_id: str
    ocr_ok: bool
    summary_ok: bool
    failed: bool


@dataclass(frozen=True)
class FailRow:
    item_id: str
    notified: bool
    failed: bool


def ocr_same(got: object, expected: object) -> bool:
    return (blank(got) or "") == (blank(expected) or "")


def summary_ok(summary: object, must_include: list[str], must_not: list[str]) -> bool:
    text = blank(summary) or ""
    return all(phrase in text for phrase in must_include) and all(
        phrase not in text for phrase in must_not
    )


def aggregate_complaints(rows: list[ComplaintRow]) -> dict[str, object]:
    n = len(rows)
    location_ok = sum(1 for row in rows if row.location_ok and not row.failed)
    symptom_ok = sum(1 for row in rows if row.symptom_ok and not row.failed)
    type_ok = sum(1 for row in rows if row.type_ok and not row.failed)
    invented = sum(
        (row.location_invented + row.symptom_invented) for row in rows if not row.failed
    )
    other = sum(1 for row in rows if row.issue_type == "other" and not row.failed)
    return {
        "n": n,
        "M1": ((location_ok + symptom_ok) / (n * 2)) if n else None,
        "location_ok": location_ok,
        "symptom_ok": symptom_ok,
        "M2": (type_ok / n) if n else None,
        "type_ok": type_ok,
        "M8": invented,
        "M9": (other / n) if n else None,
        "other": other,
        "failed": sum(1 for row in rows if row.failed),
    }


def aggregate_photos(rows: list[PhotoRow], fails: list[FailRow]) -> dict[str, object]:
    n = len(rows)
    ocr_ok = sum(1 for row in rows if row.ocr_ok and not row.failed)
    summary_hits = sum(1 for row in rows if row.summary_ok and not row.failed)
    fail_n = len(fails)
    notified = sum(1 for row in fails if row.notified and not row.failed)
    return {
        "n": n,
        "M4": (ocr_ok / n) if n else None,
        "ocr_ok": ocr_ok,
        "M5": (summary_hits / n) if n else None,
        "summary_ok": summary_hits,
        "fail_n": fail_n,
        "M6": (notified / fail_n) if fail_n else None,
        "notified": notified,
        "failed": sum(1 for row in rows if row.failed)
        + sum(1 for row in fails if row.failed),
    }


def _fmt(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def format_complaint_report(metrics: dict[str, object]) -> str:
    lines = [
        "민원 글",
        (
            f"M1 필수 필드 {_fmt(metrics['M1'])}  "
            f"(위치 {metrics['location_ok']}+증상 {metrics['symptom_ok']}"
            f"/{int(metrics['n']) * 2})"
        ),
        f"M2 택소노미 {_fmt(metrics['M2'])}  ({metrics['type_ok']}/{metrics['n']})",
        f"M8 필드 환각 {metrics['M8']}건",
        f"M9 other 도피율 {_fmt(metrics['M9'])}  ({metrics['other']}/{metrics['n']})",
        f"실패 {metrics['failed']}",
    ]
    return "\n".join(lines) + "\n"


def format_photo_report(metrics: dict[str, object]) -> str:
    lines = [
        "사진 읽기",
        f"M4 OCR {_fmt(metrics['M4'])}  ({metrics['ocr_ok']}/{metrics['n']})",
        f"M5 요약 {_fmt(metrics['M5'])}  ({metrics['summary_ok']}/{metrics['n']})",
        (
            f"M6 실패 고지 {_fmt(metrics['M6'])}  "
            f"({metrics['notified']}/{metrics['fail_n']})"
        ),
        f"실패 {metrics['failed']}",
    ]
    return "\n".join(lines) + "\n"
