"""Compare the current one-call extraction with a sparse event prototype.

Run from the repository root: PYTHONPATH=.:ai/src ai/.venv/bin/python ai/scripts/bench_complaint_events.py
This is an offline probe, not a replacement for the production handler.
"""

import json
import statistics
import time
from pathlib import Path

from zipsai.complaint.node import (
    _EXTRACTION_SCHEMA,
    _Extraction,
    _merge_complaint_draft,
    _missing_fields,
    _settle_switch,
)
from zipsai.complaint.prompts import COMPLAINT_PROMPT
from zipsai.contracts.converse import ComplaintDraft, ComplaintSwitch, ConverseRequest
from zipsai.history import format_history
from zipsai.integrations.llm import generate_text, strip_json_code_fence
from evals.complaint.score import read_jsonl, score_cases

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "work" / "bench-complaint-events-20261004-v3"
IDS = (
    "text-001", "text-003", "text-021", "text-022", "text-023", "text-024",
    "text-029", "text-031", "text-033", "text-034", "text-035", "text-036",
    "text-037", "text-038", "text-060", "text-099", "text-100", "text-101",
    "text-102", "text-103", "text-104", "text-105", "text-106", "text-110",
)

EVENT_SYSTEM = """You update one resident facility complaint from the CURRENT Korean message.
Return only JSON matching the schema. `changes` is a sparse list of values newly stated or
explicitly corrected in this turn; never copy an unchanged value from the existing draft.
Use the draft and recent conversation only to resolve references, corrections, photo
confirmations, and a pending complaint switch. Never invent facts.

`issue_type` must be exactly one of water_supply, drain, leak, heating, electricity,
mold, pest, facility, noise, other, or null. Set null if this turn does not establish
a new type. `changes` covers location (where), symptom (the actual problem in the
resident's words), and occurred_at (ISO date when first noticed; resolve relative
dates using today's date). Never put issue_type in `changes`.
heating means room heating, electricity means power, facility means a broken fixture;
mold/pest/noise are literal, and other is a real complaint outside those categories.
Water not coming out is water_supply; water not draining is drain; water escaping is leak.
Only a reply of not knowing/refusing to a previous location question sets location to 모름.
Never set symptom to 모름. A confirmed prior photo observation can set symptom. On accepting
a pending switch, copy the new symptom between 「」 in the previous assistant question.

switch is same by default. Use ask only for a clearly different symptom from the existing
complaint; use accept only when the resident accepts a pending switch. A bare answer or
correction is same. With no existing symptom, always same. An issue_type difference alone
does not trigger a switch. If switch=accept, the old draft will be discarded.

Produce one short, natural Korean question about only the first field still missing after
applying changes to the draft: location before symptom. If neither is missing, or switch=ask,
set asked_field=null and question="". For switch=accept, judge missing against the new
complaint only. Do not ask for issue_type or date. Do not include a photo preamble.
"""

EVENT_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "complaint_events",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "switch": {"type": "string", "enum": [s.value for s in ComplaintSwitch]},
                "issue_type": {"type": ["string", "null"], "enum": ["water_supply", "drain", "leak", "heating", "electricity", "mold", "pest", "facility", "noise", "other", None]},
                "changes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "field": {"type": "string", "enum": ["location", "symptom", "occurred_at"]},
                            "value": {"type": "string"},
                        },
                        "required": ["field", "value"],
                        "additionalProperties": False,
                    },
                },
                "asked_field": {"type": ["string", "null"], "enum": ["location", "symptom", None]},
                "question": {"type": "string"},
            },
            "required": ["switch", "issue_type", "changes", "asked_field", "question"],
            "additionalProperties": False,
        },
    },
}

FALLBACK = {"location": "어디에서 생긴 문제인가요?", "symptom": "어떤 불편 증상인지 알려주세요."}


def _call(case: dict, variant: str) -> dict:
    request = ConverseRequest.model_validate({"turn_id": case["id"], **case["request"]})
    if variant == "current":
        messages = COMPLAINT_PROMPT.format_messages(
            today=case["today"],
            conversation_history=format_history(request.conversation_history),
            complaint_draft=request.complaint_draft,
            message_text=request.message.text or "",
        )
        system, user, schema = str(messages[0].content), str(messages[1].content), _EXTRACTION_SCHEMA
    else:
        system, schema = EVENT_SYSTEM, EVENT_SCHEMA
        user = (
            f"Today (Asia/Seoul): {case['today']}\n"
            f"Existing draft: {request.complaint_draft.model_dump_json(ensure_ascii=False) if request.complaint_draft else 'null'}\n"
            f"Recent conversation: {format_history(request.conversation_history[-2:])}\n"
            f"Current message: {request.message.text or ''}"
        )
    usage = {}
    started = time.perf_counter()
    raw = generate_text(system, user, response_format=schema, usage_sink=usage)
    seconds = time.perf_counter() - started
    data = json.loads(strip_json_code_fence(raw))
    if variant == "current":
        updates = {key: data[key] for key in ("issue_type", "location", "symptom", "occurred_at")}
        candidate = data["reply"]
        candidate_field = None
        switch = ComplaintSwitch(data["complaint_switch"])
    else:
        updates = {"issue_type": data["issue_type"]} if data["issue_type"] else {}
        for change in data["changes"]:
            if change["field"] in updates:
                raise ValueError(f"duplicate event: {change['field']}")
            updates[change["field"]] = change["value"]
        candidate = data["question"]
        candidate_field = data["asked_field"]
        switch = ComplaintSwitch(data["switch"])
    delta = ComplaintDraft(**updates)
    settled = _settle_switch(_Extraction(draft=delta, switch=switch), request.complaint_draft)
    switch = settled.switch
    if switch == ComplaintSwitch.ASK:
        draft = request.complaint_draft
        missing = _missing_fields(draft)
        reply = f"{draft.symptom} 건은 접어두고 「{delta.symptom}」으로 전환할까요?"
        asked = None
        source = "switch_ask"
    else:
        previous = None if switch == ComplaintSwitch.ACCEPT else request.complaint_draft
        draft = _merge_complaint_draft(previous, settled.draft)
        missing = _missing_fields(draft)
        asked = missing[0] if missing else None
        if not asked:
            draft = draft if draft.issue_type else draft.model_copy(update={"issue_type": "other"})
            reply, source = "민원 정보를 확인했습니다. 접수할 내용을 확인해 주세요.", "complete"
        elif candidate.strip() and (variant == "current" or candidate_field == asked):
            reply, source = candidate.strip(), "llm"
        else:
            reply, source = FALLBACK[asked], "fixed"
    return {
        "id": case["id"], "bucket": "complaint", "variant": variant, "seconds": seconds,
        "usage": usage, "raw": data, "delta": delta.model_dump(mode="json"),
        "draft": draft.model_dump(mode="json"), "missing_fields": missing,
        "asked_field": asked, "state": "collecting" if missing else None,
        "reply": reply, "reply_source": source, "switch": switch.value,
        "candidate_field": candidate_field,
    }


def main() -> None:
    cases_by_id = {case["id"]: case for case in read_jsonl(ROOT / "evals/complaint/complaint_cases.jsonl")}
    cases = [cases_by_id[id_] for id_ in IDS]
    OUT.mkdir(parents=True, exist_ok=False)
    rows = []
    with (OUT / "predictions.jsonl").open("w") as file:
        for index, case in enumerate(cases):
            for variant in (("current", "events") if index % 2 == 0 else ("events", "current")):
                try:
                    row = _call(case, variant)
                except Exception as error:
                    row = {"id": case["id"], "variant": variant, "error": {"type": type(error).__name__, "message": str(error)}}
                rows.append(row)
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
                file.flush()
                print(f"{case['id']} {variant}: {row.get('seconds', 'ERROR')}", flush=True)
                if "error" in row:
                    raise RuntimeError(f"experiment stopped: {row['error']}")
    summary = {}
    for variant in ("current", "events"):
        selected = [row for row in rows if row["variant"] == variant]
        times = [row["seconds"] for row in selected]
        score = score_cases(cases, selected)
        summary[variant] = {
            "count": len(selected), "mean_seconds": statistics.mean(times),
            "median_seconds": statistics.median(times),
            "p95_seconds": statistics.quantiles(times, n=20, method="inclusive")[18],
            "mean_input_tokens": statistics.mean(row["usage"].get("input_tokens", 0) for row in selected),
            "mean_output_tokens": statistics.mean(row["usage"].get("output_tokens", 0) for row in selected),
            "mean_reasoning_tokens": statistics.mean(row["usage"].get("reasoning_tokens", 0) for row in selected),
            "score": score,
        }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
