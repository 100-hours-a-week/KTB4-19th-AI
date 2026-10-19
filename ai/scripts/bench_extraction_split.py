"""reply를 같은 호출에 합친 원래 구조(A) vs 추출/후속질문을 분리한 현재 구조(B)를
A/low, B/low+low, B/low+medium(현재 서비스)을 비교한다.
세 조건의 시작 순서를 순환하고 추출 결과와 답변을 JSONL 및 리뷰 Markdown으로 저장한다.

A: COMPLAINT_PROMPT + reply 필드를 되살린 스키마, 1콜.
B: 현재 COMPLAINT_PROMPT(5키, reply 없음) 1콜 + (location/symptom 중 하나라도
   비어 있으면) FOLLOW_UP_PROMPT 1콜 — node.py의 실제 분기(_missing_fields)를
   그대로 흉내낸다.

실행: cd ai && uv run python scripts/bench_extraction_split.py
"""

import argparse
import copy
import json
import statistics
import time
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate

from zipsai.complaint.node import (
    _EXTRACTION_SCHEMA,
    _Extraction,
    _merge_complaint_draft,
    _missing_fields,
    _settle_switch,
)
from zipsai.complaint.prompts import (
    COMPLAINT_PROMPT,
    COMPLAINT_SYSTEM_PROMPT,
    COMPLAINT_USER_TEMPLATE,
    FOLLOW_UP_PROMPT,
)
from zipsai.contracts.converse import ComplaintDraft, ComplaintSwitch, HistoryTurn
from zipsai.history import format_history
from zipsai.integrations.llm import generate_text, strip_json_code_fence

RUNS_PER_CASE = 10
TODAY = "2026-10-02"

# --- 변형 A: reply를 되살린 원래 프롬프트/스키마 ---

_REPLY_INSTRUCTIONS = (
    "- reply: a short, natural Korean follow-up question asking about EXACTLY ONE field — "
    'whichever of "location"/"symptom" is still unknown overall, looking at the EXISTING draft '
    "below together with what you just extracted this turn, not just this turn's message. Ask "
    'about the first still-unknown field in the order "location", then "symptom". Never ask '
    'about two fields in one turn. Empty string "" if both are already known. One short, '
    "friendly sentence, no lists. Never ask about issue_type or occurred_at. "
    'When complaint_switch is "accept", ignore the existing draft entirely and judge only against '
    "the new complaint — the old draft is being discarded. "
    'Output "" when complaint_switch is "ask" — the service writes that question itself.\n\n'
)

_ORIGINAL_SYSTEM_PROMPT = COMPLAINT_SYSTEM_PROMPT.replace(
    "Output contract: output ONLY a JSON object with exactly these five keys",
    _REPLY_INSTRUCTIONS
    + "Output contract: output ONLY a JSON object with exactly these six keys",
).replace('"occurred_at": null}}', '"occurred_at": null, "reply": ""}}')

assert _ORIGINAL_SYSTEM_PROMPT != COMPLAINT_SYSTEM_PROMPT, (
    "reply 패치가 적용되지 않음 — COMPLAINT_SYSTEM_PROMPT가 바뀌어 .replace() 대상이 안 맞는 듯"
)

PROMPT_A = ChatPromptTemplate.from_messages(
    [("system", _ORIGINAL_SYSTEM_PROMPT), ("user", COMPLAINT_USER_TEMPLATE)]
)

_SCHEMA_A = copy.deepcopy(_EXTRACTION_SCHEMA)
_SCHEMA_A["json_schema"]["schema"]["properties"]["reply"] = {"type": "string"}
_SCHEMA_A["json_schema"]["schema"]["required"].append("reply")

CASES = [
    ("빈 초안 + 텍스트", None, []),
    (
        "위치 물은 뒤 답변",
        ComplaintDraft(issue_type="leak", symptom="천장에서 물이 떨어져요"),
        [
            HistoryTurn(
                message_id="u1",
                role="user",
                text="화장실 천장에서 물이 떨어져요",
                image_urls=[],
            ),
            HistoryTurn(
                message_id="a1",
                role="assistant",
                text="어디에서 생긴 문제인가요?",
                image_urls=[],
            ),
        ],
    ),
]
TEXTS = {"빈 초안 + 텍스트": "화장실이요", "위치 물은 뒤 답변": "화장실이요"}


def call_a(
    draft: ComplaintDraft | None,
    history: list[HistoryTurn],
    text: str,
    effort: str | None = "low",
) -> dict:
    messages = PROMPT_A.format_messages(
        today=TODAY,
        conversation_history=format_history(history),
        complaint_draft=draft,
        message_text=text,
    )
    started = time.perf_counter()
    raw = generate_text(
        str(messages[0].content),
        str(messages[1].content),
        response_format=_SCHEMA_A,
        **({"reasoning_effort": effort} if effort else {}),
    )
    elapsed = time.perf_counter() - started
    data = json.loads(strip_json_code_fence(raw))
    reply = data.pop("reply", "")
    return {"seconds": elapsed, "extraction": data, "reply": reply, "calls": 1}


def call_b(
    draft: ComplaintDraft | None,
    history: list[HistoryTurn],
    text: str,
    effort: str | None = "low",
    reply_effort: str | None = None,
) -> dict:
    messages = COMPLAINT_PROMPT.format_messages(
        today=TODAY,
        conversation_history=format_history(history),
        complaint_draft=draft,
        message_text=text,
    )
    started = time.perf_counter()
    raw = generate_text(
        str(messages[0].content),
        str(messages[1].content),
        response_format=_EXTRACTION_SCHEMA,
        **({"reasoning_effort": effort} if effort else {}),
    )
    data = json.loads(strip_json_code_fence(raw))
    extracted = dict(data)
    switch = ComplaintSwitch(data.pop("complaint_switch", "same"))
    extraction = _settle_switch(
        _Extraction(draft=ComplaintDraft(**data), switch=switch), draft
    )
    if extraction.switch == ComplaintSwitch.ASK:
        return {
            "seconds": time.perf_counter() - started,
            "extraction": extracted,
            "reply": "",
            "calls": 1,
        }
    previous = None if extraction.switch == ComplaintSwitch.ACCEPT else draft
    merged = _merge_complaint_draft(previous, extraction.draft)
    missing = _missing_fields(merged)
    reply = ""
    if missing:
        asked = missing[0]
        fu_messages = FOLLOW_UP_PROMPT.format_messages(
            asked=asked,
            location=merged.location or "없음",
            symptom=merged.symptom or "없음",
        )
        reply = generate_text(
            str(fu_messages[0].content),
            str(fu_messages[1].content),
            **(
                {"reasoning_effort": reply_effort or effort}
                if reply_effort or effort
                else {}
            ),
        )
    return {
        "seconds": time.perf_counter() - started,
        "extraction": extracted,
        "reply": reply,
        "calls": 1 + bool(missing),
    }


def write_review(rows: list[dict], path: Path) -> None:
    def cell(value):
        return (
            str(value)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("|", "&#124;")
            .replace("\n", "<br>")
        )

    names = list(dict.fromkeys(r["variant"] for r in rows))
    lines = [
        "# 추출·답변 직접 리뷰",
        "",
        "A/low = 통합 1콜 low. B/low+low = 추출 low·답변 low. B/low+medium = 추출 low·답변 medium(현재 설정).",
        "",
        "같은 입력 2개를 반복한 제한적 비교다. 답변 칸은 모델 생성 원문이며 최종 서비스 응답과 다를 수 있다. 빈 답변은 후속 질문 생략을 뜻한다. A가 불필요한 질문을 생성해도 실제 서비스가 완성된 초안에서 버릴 수 있다.",
        "",
        "|조건|성공/전체|평균 초|중앙값 초|p95 초|",
        "|---|---:|---:|---:|---:|",
    ]
    for name in names:
        selected = [r for r in rows if r["variant"] == name]
        times = [r["seconds"] for r in selected if "error" not in r]
        if times:
            p95 = (
                statistics.quantiles(times, n=20, method="inclusive")[18]
                if len(times) > 1
                else times[0]
            )
            lines.append(
                f"|{name}|{len(times)}/{len(selected)}|{statistics.mean(times):.2f}|{statistics.median(times):.2f}|{p95:.2f}|"
            )
    for label, draft, history in CASES:
        lines += [
            "",
            f"## {label}",
            "",
            f"입력: {TEXTS[label]}",
            "",
            "기존 초안:",
            "```json",
            json.dumps(
                draft.model_dump(mode="json") if draft else None,
                ensure_ascii=False,
                indent=2,
            ),
            "```",
            "",
            "이전 대화:",
            "```text",
            format_history(history) or "(없음)",
            "```",
            "",
        ]
        lines += ["|조건|평균 초|중앙값 초|", "|---|---:|---:|"]
        for name in names:
            times = [
                r["seconds"]
                for r in rows
                if r["case"] == label and r["variant"] == name and "error" not in r
            ]
            if times:
                lines.append(
                    f"|{name}|{statistics.mean(times):.2f}|{statistics.median(times):.2f}|"
                )
        lines += [
            "",
            "B의 두 조건 모두 추출은 low다. 답변 호출이 생략되는 입력에서는 답변 effort 차이가 적용되지 않는다.",
            "",
        ]
        runs = sorted({r["run"] for r in rows if r["case"] == label})
        for run in runs:
            lines += [
                f"### 반복 {run}",
                "",
                "|조건|초|호출 수|추출 JSON|생성 답변 원문|",
                "|---|---:|---:|---|---|",
            ]
            for name in names:
                r = next(
                    r
                    for r in rows
                    if (r["case"], r["run"], r["variant"]) == (label, run, name)
                )
                lines.append(
                    f"|{name}|{r['seconds']:.2f}|{r.get('calls', '—')}|{cell(json.dumps(r.get('extraction'), ensure_ascii=False))}|{cell(r.get('error') or r.get('reply') or '(빈 문자열 — 질문 생략)')}|"
                )
            lines += ["", "리뷰 메모: ", ""]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="A/low vs B/low+low vs B/low+medium")
    parser.add_argument("--runs", type=int, default=RUNS_PER_CASE)
    parser.add_argument(
        "--output", type=Path, default=Path("bench_extraction_three.jsonl")
    )
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")
    variants = [
        ("A/low", call_a, ("low",)),
        ("B/low+low", call_b, ("low", "low")),
        ("B/low+medium", call_b, ("low", "medium")),
    ]
    rows = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        for label, draft, history in CASES:
            for run in range(1, args.runs + 1):
                offset = (run - 1) % len(variants)
                for name, fn, efforts in variants[offset:] + variants[:offset]:
                    started = time.perf_counter()
                    row = {
                        "case": label,
                        "run": run,
                        "variant": name,
                        "extraction_effort": efforts[0],
                        "reply_effort": efforts[-1],
                    }
                    try:
                        row.update(fn(draft, history, TEXTS[label], *efforts))
                    except (ValueError, RuntimeError) as error:
                        row.update(
                            seconds=time.perf_counter() - started,
                            error=type(error).__name__,
                        )
                    rows.append(row)
                    output.write(json.dumps(row, ensure_ascii=False) + "\n")
                    output.flush()
                    print(json.dumps(row, ensure_ascii=False), flush=True)
    write_review(rows, args.output.with_suffix(".md"))
    print(f"리뷰: {args.output.with_suffix('.md')}")


if __name__ == "__main__":
    main()
