"""T1 집계. 네트워크와 모델 호출은 없다.

채점식은 설계-docs/질의-채점표.md §4-4.
doc_id는 §4-2다. 확장자를 뗀 골드셋 파일명 앞에 건물 코드를 붙인다.
스캔본·사진의 디스크 파일명은 문서 제목으로 따로 붙는다. 그 이름과 doc_id는 다르다.
섹션(## 제목)은 채점하지 않는다. PDF로 구우면서 빠진다.
"""

from __future__ import annotations

import json
import re
import statistics
import unicodedata
from dataclasses import dataclass
from pathlib import Path

# make_inputs.safe 와 같다. 제목의 가운데점·공백·슬래시를 파일명에서 밑줄로 바꾼다.
_TITLE_UNSAFE = re.compile(r"[ ·/]+")
_INTENT = re.compile(r"[A-Z][A-Z0-9_]*")

EXPECTED_QUESTIONS = 235
EXPECTED_ANSWERABLE = 135
EXPECTED_UNANSWERABLE = 100
EXPECTED_BUILDING_QUESTIONS = 97
EXPECTED_FILES = 51

REPRODUCE = Path(__file__).resolve().parents[1] / "synthetic" / "2-reproduce"
BUILD = Path(__file__).resolve().parents[1] / "synthetic" / "1-dataset" / "build"

TARGETS = {
    "R2": 0.95,
    "R3": 0.70,
    "R4": 0.85,
    "R5": 0.90,
    "R6": 0.70,
    "R7": 2,
}


def nfc(value: str) -> str:
    return unicodedata.normalize("NFC", value)


def safe_title(name: str) -> str:
    return _TITLE_UNSAFE.sub("_", name).strip("_")


def kind_of(intent: list[str]) -> str:
    codes = set(intent)
    if "PARSE_NO_TEXT_LAYER" in codes:
        return "scan"
    if "PARSE_IMAGE_ONLY" in codes:
        return "photo"
    if "CHUNK_TABLE_WHOLE" in codes:
        return "table"
    return "body"


def build_name(building_code: str, md_file: str, title: str, file_type: str, kind: str) -> str:
    if kind == "scan":
        return f"{building_code}-{safe_title(title)}-스캔본.pdf"
    if kind == "photo":
        return f"{building_code}-{safe_title(title)}-사진.{file_type}"
    return f"{building_code}-{Path(md_file).stem}.pdf"


def doc_id_of(building_code: str, md_file: str) -> str:
    return nfc(f"{building_code}-{Path(md_file).stem}")


@dataclass(frozen=True)
class ManifestDoc:
    building_code: str
    building_id: int
    doc_id: str
    title: str
    md_file: str
    build_name: str
    kind: str


@dataclass(frozen=True)
class GoldItem:
    id: str
    building_code: str
    building_id: int
    question: str
    answerable: bool
    scope: str
    expected_doc_id: str | None
    kind: str | None


@dataclass(frozen=True)
class Scored:
    item_id: str
    answerable: bool
    scope: str
    kind: str | None
    passed: bool
    top_score: float | None
    expected_doc_id: str | None
    requested_building_id: str
    hit_doc_ids: tuple[str, ...]
    hit_building_ids: tuple[str, ...]


def _field(line: str, key: str) -> str | None:
    prefix = f"{key}:"
    stripped = line.strip()
    if not stripped.startswith(prefix):
        return None
    return stripped[len(prefix) :].strip()


def _manifest_doc(building_code: str, building_id: int, raw: dict[str, str]) -> ManifestDoc:
    intent = _INTENT.findall(raw.get("intent", ""))
    kind = kind_of(intent)
    md_file = raw["file"]
    title = raw["title"]
    return ManifestDoc(
        building_code=building_code,
        building_id=building_id,
        doc_id=doc_id_of(building_code, md_file),
        title=title,
        md_file=md_file,
        build_name=nfc(
            build_name(building_code, md_file, title, raw.get("file_type", ""), kind)
        ),
        kind=kind,
    )


def load_manifest(reproduce: Path) -> list[ManifestDoc]:
    """building.yaml에서 색인에 필요한 칸만 읽는다. PyYAML에 의존하지 않는다."""
    docs: list[ManifestDoc] = []
    for path in sorted((reproduce / "buildings").glob("b*/building.yaml")):
        building_id: int | None = None
        building_code: str | None = None
        raw_docs: list[dict[str, str]] = []
        current: dict[str, str] | None = None
        in_documents = False
        for raw in path.read_text(encoding="utf-8").splitlines():
            if not raw.strip() or raw.lstrip().startswith("#"):
                continue
            indent = len(raw) - len(raw.lstrip(" "))
            if indent == 0:
                if current is not None:
                    raw_docs.append(current)
                    current = None
                in_documents = raw.strip().startswith("documents:")
                if (value := _field(raw, "building_id")) is not None:
                    building_id = int(value)
                elif (value := _field(raw, "building_code")) is not None:
                    building_code = value
                continue
            if not in_documents:
                continue
            stripped = raw.strip()
            if stripped.startswith("- file:"):
                if current is not None:
                    raw_docs.append(current)
                current = {"file": stripped.split(":", 1)[1].strip()}
                continue
            if current is None:
                continue
            for key in ("document_title", "file_type", "intent"):
                if (value := _field(stripped, key)) is not None:
                    stored = "title" if key == "document_title" else key
                    current[stored] = value
        if current is not None:
            raw_docs.append(current)
        if building_id is None or not building_code:
            raise ValueError(f"building_id 또는 building_code가 없습니다: {path}")
        docs.extend(_manifest_doc(building_code, building_id, raw) for raw in raw_docs)
    return docs


def load_gold(reproduce: Path, docs: list[ManifestDoc]) -> tuple[list[GoldItem], list[str]]:
    by_id = {doc.doc_id: doc for doc in docs}
    buildings = {doc.building_code: doc.building_id for doc in docs}
    items: list[GoldItem] = []
    errors: list[str] = []
    for path in sorted((reproduce / "buildings").glob("b*/goldset.jsonl")):
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            code = row["building_code"]
            citation = row.get("expected_citation") or None
            expected = None
            kind = None
            if citation:
                expected = doc_id_of(code, citation["file"])
                matched = by_id.get(expected)
                if matched is None:
                    errors.append(f"{path.name}:{line_no} 정답 문서 없음 {expected}")
                else:
                    kind = matched.kind
            elif row["answerable"]:
                errors.append(f"{path.name}:{line_no} 답있음인데 정답 문서가 없습니다")
            if code not in buildings:
                errors.append(f"{path.name}:{line_no} 건물 코드 없음 {code}")
            items.append(
                GoldItem(
                    id=row["id"],
                    building_code=code,
                    building_id=buildings.get(code, 0),
                    question=row["question"],
                    answerable=bool(row["answerable"]),
                    scope=row["scope"],
                    expected_doc_id=expected,
                    kind=kind,
                )
            )
    return items, errors


def build_files(build: Path) -> dict[str, Path]:
    return {
        nfc(path.name): path
        for path in build.iterdir()
        if path.is_file() and not path.name.startswith(".")
    }


def check_inputs(reproduce: Path, build: Path) -> list[str]:
    docs = load_manifest(reproduce)
    items, errors = load_gold(reproduce, docs)
    errors = list(errors)
    if len(docs) != EXPECTED_FILES:
        errors.append(f"문서 {len(docs)}건 (기대 {EXPECTED_FILES})")
    doc_ids = [doc.doc_id for doc in docs]
    if len(doc_ids) != len(set(doc_ids)):
        errors.append("doc_id가 겹칩니다")
    names = build_files(build) if build.is_dir() else {}
    if not build.is_dir():
        errors.append(f"build 디렉터리가 없습니다: {build}")
    for doc in docs:
        if doc.build_name not in names:
            errors.append(f"파일 없음 {doc.build_name}")
    extra = sorted(set(names) - {doc.build_name for doc in docs})
    for name in extra:
        errors.append(f"매니페스트에 없는 파일 {name}")
    answerable = [item for item in items if item.answerable]
    unanswerable = [item for item in items if not item.answerable]
    building = [item for item in answerable if item.scope == "building"]
    if len(items) != EXPECTED_QUESTIONS:
        errors.append(f"문항 {len(items)} (기대 {EXPECTED_QUESTIONS})")
    if len(answerable) != EXPECTED_ANSWERABLE:
        errors.append(f"답있음 {len(answerable)} (기대 {EXPECTED_ANSWERABLE})")
    if len(unanswerable) != EXPECTED_UNANSWERABLE:
        errors.append(f"답없음 {len(unanswerable)} (기대 {EXPECTED_UNANSWERABLE})")
    if len(building) != EXPECTED_BUILDING_QUESTIONS:
        errors.append(f"건물고유 {len(building)} (기대 {EXPECTED_BUILDING_QUESTIONS})")
    ids = [item.id for item in items]
    if len(ids) != len(set(ids)):
        errors.append("문항 id가 겹칩니다")
    return errors


def _rank(expected: str | None, hit_doc_ids: tuple[str, ...]) -> int | None:
    if expected is None:
        return None
    wanted = nfc(expected)
    for index, doc_id in enumerate(hit_doc_ids, start=1):
        if nfc(doc_id) == wanted:
            return index
    return None


def _rate(hits: int, total: int) -> float | None:
    if total == 0:
        return None
    return hits / total


def aggregate(rows: list[Scored]) -> dict[str, object]:
    answerable = [row for row in rows if row.answerable]
    unanswerable = [row for row in rows if not row.answerable]
    building = [row for row in answerable if row.scope == "building"]
    passed_answerable = [row for row in answerable if row.passed]

    violations = 0
    for row in building:
        wanted = row.requested_building_id
        violations += sum(1 for building_id in row.hit_building_ids if building_id != wanted)

    recalled = 0
    for row in answerable:
        if _rank(row.expected_doc_id, row.hit_doc_ids) is not None:
            recalled += 1

    passed_recalled = 0
    reciprocal: list[float] = []
    distinct: list[int] = []
    for row in passed_answerable:
        rank = _rank(row.expected_doc_id, row.hit_doc_ids)
        if rank is not None:
            passed_recalled += 1
            reciprocal.append(1 / rank)
        else:
            reciprocal.append(0)
        distinct.append(len({nfc(doc_id) for doc_id in row.hit_doc_ids if doc_id}))

    blocked = sum(1 for row in unanswerable if not row.passed)
    slices: dict[str, dict[str, int]] = {}
    for kind in ("body", "table", "scan", "photo"):
        group = [row for row in answerable if row.kind == kind]
        slices[kind] = {
            "n": len(group),
            "passed": sum(1 for row in group if row.passed),
            "recalled": sum(
                1 for row in group if _rank(row.expected_doc_id, row.hit_doc_ids) is not None
            ),
        }

    return {
        "n": len(rows),
        "answerable": len(answerable),
        "unanswerable": len(unanswerable),
        "building": len(building),
        "R1": violations,
        "R2": _rate(len(passed_answerable), len(answerable)),
        "R3": _rate(blocked, len(unanswerable)),
        "R4": _rate(recalled, len(answerable)),
        "R5": _rate(passed_recalled, len(passed_answerable)),
        "R6": (sum(reciprocal) / len(reciprocal)) if reciprocal else None,
        "R7": statistics.median(distinct) if distinct else None,
        "passed_answerable": len(passed_answerable),
        "slices": slices,
    }


def _fmt(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def format_report(metrics: dict[str, object], *, index_lines: list[str] | None = None) -> str:
    judge = metrics["R1"] == 0
    lines = ["T1 검색 채점"]
    if index_lines:
        lines.extend(index_lines)
    lines.append(
        f"문항 {metrics['n']} · 답있음 {metrics['answerable']} · "
        f"답없음 {metrics['unanswerable']} · 건물고유 {metrics['building']}"
    )
    lines.append(f"R1 격리 위반 {_fmt(metrics['R1'])}건  목표 0")
    if not judge:
        lines.append("R1이 0이 아니라 R2 이하는 숫자만 남기고 판정하지 않습니다.")
    labels = {
        "R2": "게이트 통과율(답있음)",
        "R3": "게이트 차단율(답없음)",
        "R4": "Recall@5 (파이프라인)",
        "R5": "Recall@5 (검색만)",
        "R6": "MRR@5",
        "R7": "고유 문서 수 중앙값",
    }
    for key, label in labels.items():
        value = metrics[key]
        target = TARGETS[key]
        mark = ""
        if judge and value is not None:
            ok = value >= target
            mark = "  통과" if ok else "  미달"
        elif not judge:
            mark = "  판정 안 함"
        lines.append(f"{key} {label} {_fmt(value)}  목표 {_fmt(target)}{mark}")
    lines.append(
        f"R5·R6·R7 분모는 게이트를 통과한 답있음 {metrics['passed_answerable']}문항입니다."
    )
    lines.append("슬라이스  답있음 기준. recalled는 상위 5개에 정답 문서가 있는 수.")
    names = {"body": "본문 PDF", "table": "표", "scan": "스캔본", "photo": "사진"}
    slices = metrics["slices"]
    assert isinstance(slices, dict)
    for kind, label in names.items():
        cell = slices[kind]
        lines.append(
            f"{label}  n={cell['n']}  게이트 통과 {cell['passed']}  정답 문서 {cell['recalled']}"
        )
    return "\n".join(lines) + "\n"
