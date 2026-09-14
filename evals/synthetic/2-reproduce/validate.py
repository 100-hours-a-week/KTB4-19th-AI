#!/usr/bin/env python3
"""합성데이터 제약 검증. 인자 없이 실행하면 전부 돌린다.

  --lengths    A1 document_title 20자 / room_no 5자 / expected_answer 200자
  --citations  A2 expected_citation이 실존 파일·섹션을 가리키는가
  --scope      A3 scope 라벨 존재 + 같은 구역 건물끼리 고유값이 겹치지 않는가
  --scale      A4 건물 10 / 호실 합계 280~320 / 구역 4
"""
import json
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
fails = []


def fail(check, msg):
    fails.append(f"[{check}] {msg}")


def buildings():
    for d in sorted((ROOT / "buildings").glob("b*")):
        f = d / "building.yaml"
        if f.exists():
            yield d, yaml.safe_load(f.read_text(encoding="utf-8"))


def goldset(d):
    f = d / "goldset.jsonl"
    if not f.exists():
        return
    for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if line:
            try:
                yield i, json.loads(line)
            except json.JSONDecodeError as e:
                fail("json", f"{f.name}:{i} 파싱 실패 — {e}")


def check_lengths():
    for d, b in buildings():
        fmt = str(b.get("room_no_format", ""))
        if len(fmt) > 5:
            fail("lengths", f"{d.name} room_no_format {len(fmt)}자 — Rooms.room_no VARCHAR(5) 초과")
        for doc in b.get("documents", []):
            t = doc.get("document_title", "")
            if len(t) > 20:
                fail("lengths", f"{d.name} '{t}' {len(t)}자 — document_title VARCHAR(20) 초과")
        for ln, item in goldset(d):
            a = item.get("expected_answer") or ""
            if len(a) > 200:
                fail("lengths", f"{d.name}:{ln} expected_answer {len(a)}자 — Messages.content VARCHAR(200) 초과")


def check_citations():
    for d, b in buildings():
        for ln, item in goldset(d):
            cit = item.get("expected_citation")
            if not item.get("answerable"):
                if cit is not None:
                    fail("citations", f"{d.name}:{ln} answerable=false인데 citation이 있음")
                continue
            if not cit:
                fail("citations", f"{d.name}:{ln} answerable=true인데 citation이 없음")
                continue
            path = d / cit.get("file", "")
            if not path.exists():
                fail("citations", f"{d.name}:{ln} 파일 없음 — {cit.get('file')}")
                continue
            section = cit.get("section", "")
            heads = re.findall(r"^#{2,3}\s+(.+?)\s*$", path.read_text(encoding="utf-8"), re.M)
            if section not in heads:
                fail("citations", f"{d.name}:{ln} 섹션 없음 — '{section}' in {cit.get('file')} (있는 섹션: {heads})")


SPECIFIC_KEYS = ["parking", "monthly_fee", "laundry", "parcel", "entrance"]


def check_scope():
    by_zone = {}
    for d, b in buildings():
        by_zone.setdefault(b.get("zone_id"), []).append((d.name, b))
        for ln, item in goldset(d):
            s = item.get("scope")
            if s not in ("building", "zone"):
                fail("scope", f"{d.name}:{ln} scope 라벨이 없거나 잘못됨 — {s!r}")

    # 건물 고유값이 겹치면 격리 검증이 무의미해진다.
    # 검색은 구역을 모르므로 같은 구역 안이 아니라 10개 건물 전체에서 유일해야 한다.
    for key in SPECIFIC_KEYS:
        seen = {}
        for items in by_zone.values():
            for name, b in items:
                v = (b.get("specifics") or {}).get(key)
                if v is None:
                    fail("scope", f"{name} specifics.{key} 누락")
                    continue
                if v in seen:
                    fail("scope", f"{seen[v]}와 {name}의 {key}가 동일 — '{v}' (격리 검증 무효)")
                seen[v] = name


def check_scale():
    bs = list(buildings())
    if len(bs) != 10:
        fail("scale", f"건물 {len(bs)}개 — 10개여야 함")
    total = sum(b.get("room_count", 0) for _, b in bs)
    if not 280 <= total <= 320:
        fail("scale", f"호실 합계 {total} — 280~320이어야 함 (초기규모설계 약 300)")
    zone_files = sorted((ROOT / "zones").glob("z*.yaml"))
    if len(zone_files) != 4:
        fail("scale", f"구역 {len(zone_files)}개 — 4개여야 함")
    declared = {}
    for f in zone_files:
        z = yaml.safe_load(f.read_text(encoding="utf-8"))
        for code in z.get("buildings", []):
            declared[code] = z["zone_id"]
    for d, b in bs:
        want, got = declared.get(b.get("building_code")), b.get("zone_id")
        if want != got:
            fail("scale", f"{d.name} 구역 불일치 — zones는 {want}, building.yaml은 {got}")
    print(f"  건물 {len(bs)} / 호실 {total} / 구역 {len(zone_files)}")


CHECKS = {
    "--lengths": check_lengths,
    "--citations": check_citations,
    "--scope": check_scope,
    "--scale": check_scale,
}


def main():
    args = [a for a in sys.argv[1:] if a != "--all"] or list(CHECKS)
    for a in args:
        if a not in CHECKS:
            sys.exit(f"모르는 인자: {a}\n{__doc__}")
        print(f"{a}")
        CHECKS[a]()
    if fails:
        print(f"\nFAIL {len(fails)}건")
        for f in fails:
            print(f"  {f}")
        sys.exit(1)
    print("\nPASS")


if __name__ == "__main__":
    main()
