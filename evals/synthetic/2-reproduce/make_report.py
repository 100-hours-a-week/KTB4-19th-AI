#!/usr/bin/env python3
"""데이터셋 현황을 docs/데이터셋-안내.md 로 뽑는다.

수치를 손으로 적지 않기 위한 스크립트다. 데이터가 바뀌면 다시 돌리면 된다.
    python3 make_report.py
"""
import collections
import json
import re
from datetime import date
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent                    # 2-reproduce/
OUT = ROOT.parent / "1-dataset" / "데이터셋-안내.md"        # 완성물은 1-dataset/ 으로 나간다

TOPIC = [
    ("쓰레기·분리수거", r"쓰레기|분리|재활용|종량제|폐기물|음식물|페트|캔"),
    ("주차", r"주차|차량|견인"),
    ("세탁", r"세탁|건조"),
    ("관리비", r"관리비|납부|연체"),
    ("택배·우편", r"택배|보관함|우편"),
    ("출입·현관", r"현관|비밀번호|카드키|지문|열쇠|출입"),
    ("점검·공사", r"점검|청소|물탱크|정화조|소방|승강기|도시가스|동파|계량기"),
    ("소음", r"소음|조용|시끄"),
    ("반려동물", r"반려|강아지|고양이|애완"),
    ("공용시설", r"옥상|계단|복도|창고|자전거"),
    ("계약·퇴실", r"계약|연장|갱신|퇴실|보증금|청소비"),
    ("설비수리", r"에어컨|보일러|방충망|누수|고장|수리|도배"),
    ("행정", r"전입신고|서류"),
    ("인터넷", r"인터넷|통신사|회선"),
    ("흡연", r"흡연|담배"),
]


def topic(q):
    for name, pat in TOPIC:
        if re.search(pat, q):
            return name
    return "기타"


def collect():
    out = []
    for d in sorted((ROOT / "buildings").glob("b*")):
        b = yaml.safe_load((d / "building.yaml").read_text(encoding="utf-8"))
        gs = [json.loads(l) for l in (d / "goldset.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        kinds = collections.Counter()
        for doc in b["documents"]:
            it = set(doc["intent"])
            kinds["scan" if "PARSE_NO_TEXT_LAYER" in it
                  else "photo" if "PARSE_IMAGE_ONLY" in it else "text"] += 1
        out.append({
            "b": b, "gs": gs, "kinds": kinds,
            "yes": collections.Counter(topic(x["question"]) for x in gs if x["answerable"]),
            "no": collections.Counter(topic(x["question"]) for x in gs if not x["answerable"]),
            "n_yes": sum(1 for x in gs if x["answerable"]),
            "n_no": sum(1 for x in gs if not x["answerable"]),
            "n_zone": sum(1 for x in gs if x["scope"] == "zone"),
        })
    return out


def topics(c):
    return " · ".join(f"{k} {v}" for k, v in c.most_common())


def main():
    rows = collect()
    tot = lambda f: sum(f(r) for r in rows)
    qa = [json.loads(l) for l in (ROOT / "path-b/qa-cards.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

    L = [
        "# 합성데이터 안내",
        "",
        f"생성일 {date.today()} · `python3 make_report.py` 로 다시 뽑을 수 있다.",
        "",
        "순천향대 인근 4구역의 원룸형 빌라·다가구주택 10개동을 대상으로 만든 "
        "건물관리 문서 RAG용 합성데이터다. 문서만이 아니라 **정답셋을 함께** 들고 있다. "
        "정답셋이 없으면 검색 품질이 좋아졌는지 나빠졌는지 잴 수가 없다.",
        "",
        "## 1. 건물별 구성",
        "",
        "| code | 구역 | 유형 | 준공 | 호실 | 문서 | 답있음 | 답없음 | 구역질문 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        b = r["b"]
        L.append(f"| {b['building_code']} | {b['zone_id']} | {b['building_type']} | {b['built_year']} | "
                 f"{b['room_count']} | {len(b['documents'])} | {r['n_yes']} | {r['n_no']} | {r['n_zone']} |")
    L.append(f"| **합계** | **4구역** | | | **{tot(lambda r: r['b']['room_count'])}** | "
             f"**{tot(lambda r: len(r['b']['documents']))}** | **{tot(lambda r: r['n_yes'])}** | "
             f"**{tot(lambda r: r['n_no'])}** | **{tot(lambda r: r['n_zone'])}** |")
    L += [
        "",
        "호실 합계는 초기 서비스 규모 설계의 「10개 건물 × 평균 30호실 ≈ 300명」에 맞췄다.",
        "문서가 3~4건뿐인 건물은 실수가 아니라 **문서화가 부실한 건물**을 일부러 넣은 것이다.",
        "",
        "## 2. 건물별 — 답이 있는 것 / 없는 것",
        "",
        "숫자는 문항 수다.",
        "",
        "| 건물 | 답이 **있는** 주제 | 답이 **없는** 주제 |",
        "|---|---|---|",
    ]
    for r in rows:
        b = r["b"]
        L.append(f"| {b['building_code']} {b['building_name']} | {topics(r['yes'])} | {topics(r['no'])} |")
    L += [
        "",
        "### 읽는 법",
        "",
        "**답있음은 건물마다 다르다.** 그 건물 문서에 실제로 적힌 것만 물어본다. "
        "세탁실이 없는 건물은 세탁 문항이 없고, 안내문이 많은 신축은 점검 문항이 많다.",
        "",
        "**답없음은 10개 건물이 거의 같다.** 계약·보증금·전입신고·설비수리 책임·인터넷 사업자·흡연 구역은 "
        "원룸 관리인이 A4 안내문에 쓰지 않는 주제라 어느 건물에서 물어도 문서에 답이 없다. "
        "실제로 입주민이 가장 자주 묻는데 문서에는 없는 것들이기도 하다.",
        "",
        "**「점검」이 양쪽에 다 있는 이유** — 그 건물이 공지한 점검은 답이 있고, "
        "공지하지 않은 점검을 물으면 답이 없다. 같은 주제라도 문서에 적혔는지로 갈린다.",
        "",
        "### 두 구분이 잡아내는 실패",
        "",
        "| 구분 | AI가 해야 하는 일 | 실패 모습 | 그다음 |",
        "|---|---|---|---|",
        "| 답있음 | 맞는 청크를 찾아 그 근거로 답하기 | 답이 있는데 못 찾음 → 불필요한 민원 접수 | 끝 |",
        "| 답없음 | **모른다고 말하고** 민원 접수를 제안 (`has_sufficient_evidence = false`) | "
        "없는 답을 지어냄(환각) → 잘못된 안내 | 질의 카드 → 관리자 답변 → 재색인 |",
        "",
        f"답없음 {tot(lambda r: r['n_no'])}건 중 관리자 답변이 달린 **{len(qa)}건**만 "
        f"`path-b/qa-cards.jsonl`에 있다. 그중 {sum(1 for x in qa if x['note'])}건은 "
        "「확인 후 연락드리겠습니다」처럼 내용이 없어 색인 제외 대상이다. "
        f"나머지 {tot(lambda r: r['n_no']) - len(qa)}건은 파일에 없다 — "
        "**관리자가 답하지 않으면 루프가 멈춘다**는 상태를 데이터로 재현한 것이다.",
        "",
        "## 3. 건물별 문서 종류",
        "",
        "원본은 모두 Markdown이고, `make_inputs.py`가 파싱 입력 파일로 변환한다.",
        "",
        "| 건물 | 본문 | 스캔본 | 사진공지 | 합계 |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        k = r["kinds"]
        L.append(f"| {r['b']['building_code']} | {k['text']} | {k['scan']} | {k['photo']} | "
                 f"{sum(k.values())} |")
    L.append(f"| **합계** | **{tot(lambda r: r['kinds']['text'])}** | "
             f"**{tot(lambda r: r['kinds']['scan'])}** | **{tot(lambda r: r['kinds']['photo'])}** | "
             f"**{tot(lambda r: sum(r['kinds'].values()))}** |")
    L += [
        "",
        "| 종류 | 변환 경로 | 결과 | 무엇을 시험하나 |",
        "|---|---|---|---|",
        "| 본문 | md → **PDF** | `.pdf` (텍스트 레이어 있음) | 정상 파싱 |",
        "| 스캔본 | md → PDF → **PNG** → PDF | `.pdf` (텍스트 레이어 **없음**) | 스캔 문서 OCR |",
        "| 사진공지 | md → PDF → **JPEG·PNG** | `.jpg` `.png` | 사진으로 찍어 올린 공지 |",
        "",
        "사진공지는 PDF로 돌아가지 않는다. 입주민이 휴대폰으로 찍어 올린 상황이라 이미지에서 끝나고, 색인은 docling OCR이 맡는다.",
        "스캔본은 쪽마다 따로 굽고 다시 합친다 — `sips`가 PDF 첫 장만 변환하기 때문이다.",
        "",
        "## 재현",
        "",
        "```bash",
        "python3 make_inputs.py      # build/ 에 파싱 입력 생성 + 자체 검사",
        "python3 validate.py         # 데이터셋 제약 검사",
        "python3 make_report.py      # 이 문서 갱신",
        "```",
        "",
        "`make_inputs.py`와 `make_report.py`는 reportlab·pypdf가 필요해 `.venv` 파이썬으로 실행한다.",
        "",
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L), encoding="utf-8")
    print(f"{OUT} — {len(L)}줄")


if __name__ == "__main__":
    main()
