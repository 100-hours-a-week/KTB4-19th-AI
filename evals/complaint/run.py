"""Call the current models on the fixed complaint evaluation cases.

Run from repository root with PYTHONPATH=.:ai/src and ai/.venv/bin/python.
"""

import argparse
import base64
import hashlib
import json
import mimetypes
import subprocess
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

from zipsai.complaint import node
from zipsai.complaint.prompts import VLM_ANALYSIS_PROMPT
from zipsai.contracts.converse import ComplaintDraft, ConverseRequest
from zipsai.errors import LlmRateLimitedError, LlmUnavailableError
from zipsai.integrations.vlm import analyze_images
from zipsai.orchestration import intent
from zipsai.orchestration.graph import select_entry_node
from zipsai.orchestration.intent import classify_intent
from zipsai.settings import get_settings

from evals.complaint.score import input_fingerprint, read_jsonl

ROOT = Path(__file__).parent
REPO = ROOT.parent.parent


def _frozen_datetime(today: str) -> type[datetime]:
    class FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime.fromisoformat(today + "T12:00:00+09:00").astimezone(tz or ZoneInfo("Asia/Seoul"))

    return FrozenDateTime


def run_complaint_case(case: dict) -> dict:
    request = ConverseRequest.model_validate(case["request"])
    original_generate = node.generate_text
    raw_outputs: list[str] = []

    def record_generate(*args, **kwargs):
        output = original_generate(*args, **kwargs)
        raw_outputs.append(output)
        return output

    with patch.object(node, "datetime", _frozen_datetime(case["today"])), patch.object(node, "generate_text", record_generate):
        extracted, reply, llm_missing = node._extract_complaint_fields_and_reply(request)
    # The production handler merges the same parsed result. Reuse it without a second model call.
    with patch.object(node, "_extract_complaint_fields_and_reply", return_value=(extracted, reply, llm_missing)):
        result = node.handle_complaint(request)
    route_result = result["result"]
    return {
        "id": case["id"],
        "bucket": "complaint",
        "delta": extracted.model_dump(mode="json"),
        "draft": route_result.complaint_draft.model_dump(mode="json"),
        "missing_fields": route_result.missing_fields,
        "asked_field": route_result.missing_fields[0] if route_result.missing_fields else None,
        "state": result["complaint_state"].value if result["complaint_state"] else None,
        "reply": result["reply"],
        "llm_missing": sorted(llm_missing),
        "raw_outputs": raw_outputs,
    }


def run_route_case(case: dict) -> dict:
    request = ConverseRequest.model_validate(case["request"])
    entry = select_entry_node({"request": request})
    raw_outputs: list[str] = []
    if entry == "complaint":
        route = "complaint"
    else:
        original_generate = intent.generate_text

        def record_generate(*args, **kwargs):
            output = original_generate(*args, **kwargs)
            raw_outputs.append(output)
            return output

        with patch.object(intent, "generate_text", record_generate):
            route = classify_intent({"request": request})["route"].value
    return {"id": case["id"], "bucket": "routing", "entry_node": entry, "route": route, "raw_outputs": raw_outputs}


def run_photo_case(case: dict) -> dict:
    image_path = ROOT / case["image_path"]
    mime = mimetypes.guess_type(image_path)[0] or "image/png"
    data_url = f"data:{mime};base64,{base64.b64encode(image_path.read_bytes()).decode('ascii')}"
    analysis = analyze_images([data_url], VLM_ANALYSIS_PROMPT)
    return {
        "id": case["id"],
        "bucket": "vlm",
        "images": [
            {"summary": image.summary, "ocr_text": image.ocr_text}
            for image in analysis.images
        ],
    }


def _cases(bucket: str) -> list[tuple[dict, object]]:
    selected: list[tuple[dict, object]] = []
    if bucket in ("complaint", "all"):
        selected.extend((case, run_complaint_case) for case in read_jsonl(ROOT / "complaint_cases.jsonl"))
        selected.extend((turn, run_complaint_case) for conversation in read_jsonl(ROOT / "complaint_conversations.jsonl") for turn in conversation["turns"])
    if bucket in ("routing", "all"):
        selected.extend((case, run_route_case) for case in read_jsonl(ROOT / "routing_cases.jsonl"))
    photo_gold = ROOT / "photo_cases.jsonl"
    if bucket in ("vlm", "all") and photo_gold.is_file():
        selected.extend((case, run_photo_case) for case in read_jsonl(photo_gold))
    if len({case["id"] for case, _ in selected}) != len(selected):
        raise ValueError("duplicate gold case IDs")
    return selected


def validate_cases(selected: list[tuple[dict, object]]) -> int:
    for case, runner in selected:
        if runner is run_complaint_case:
            ConverseRequest.model_validate(case["request"])
            ComplaintDraft.model_validate(case["expected"]["delta"])
            draft = ComplaintDraft.model_validate(case["expected"]["draft"])
            missing = node._missing_fields(draft)
            if case["expected"]["missing_fields"] != missing or case["expected"]["asked_field"] != (missing[0] if missing else None):
                raise ValueError(f"inconsistent complaint gold: {case['id']}")
        elif runner is run_route_case:
            request = ConverseRequest.model_validate(case["request"])
            if select_entry_node({"request": request}) != case["expected"]["entry_node"]:
                raise ValueError(f"inconsistent route entry: {case['id']}")
        else:
            image_path = ROOT / case["image_path"]
            if not image_path.is_file():
                raise ValueError(f"missing photo fixture: {case['id']}")
            if Path(case["image_path"]).parts[:2] == ("photos", "candidates"):
                if case.get("review_status") != "approved":
                    raise ValueError(f"candidate photo not approved: {case['id']}")
                if case.get("image_sha256") != hashlib.sha256(image_path.read_bytes()).hexdigest():
                    raise ValueError(f"candidate photo hash mismatch: {case['id']}")
                if (
                    not case.get("device")
                    or (type(case.get("anomaly")) is not bool and case.get("anomaly") != "unknown")
                    or case.get("source_type") != "synthetic"
                    or "gold_ocr" not in case
                    or not isinstance(case["gold_ocr"], (str, type(None)))
                    or case["gold_ocr"] == ""
                    or not isinstance(case.get("key_tokens"), list)
                    or not isinstance(case.get("required_facts"), list)
                    or not case["required_facts"]
                    or not isinstance(case.get("forbidden"), list)
                    or not isinstance(case.get("refs"), list)
                    or not case["refs"]
                    or not isinstance(case.get("tags"), list)
                ):
                    raise ValueError(f"candidate photo incomplete gold: {case['id']}")
    return len(selected)


def _meta(bucket: str, count: int) -> dict:
    settings = get_settings()
    prompt_files = [REPO / "ai/src/zipsai/complaint/prompts.py", REPO / "ai/src/zipsai/orchestration/prompts.py"]
    prompt_hash = hashlib.sha256(b"".join(path.read_bytes() for path in prompt_files)).hexdigest()
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=REPO, capture_output=True, text=True, check=True).stdout)
    return {
        "bucket": bucket,
        "cases": count,
        "model": settings.llm_model,
        "vlm_model": settings.vlm_model,
        "prompt_sha256": prompt_hash,
        "input_sha256": input_fingerprint(bucket),
        "git_sha": sha,
        "dirty_worktree": dirty,
        "created_at": datetime.now(ZoneInfo("UTC")).isoformat(),
        "cost": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", choices=("complaint", "routing", "vlm", "all"), default="all")
    parser.add_argument("--out", type=Path, help="output directory; default is timestamped under evals/complaint/runs")
    parser.add_argument("--validate-only", action="store_true", help="check gold data without model calls")
    args = parser.parse_args()
    selected = _cases(args.bucket)
    if args.bucket == "vlm" and not selected:
        raise ValueError("no approved VLM gold cases; label candidates before running VLM evaluation")
    validate_cases(selected)
    if args.validate_only:
        print(f"validated {len(selected)} cases")
        return
    run_dir = args.out or ROOT / "runs" / datetime.now(ZoneInfo("UTC")).strftime("%Y%m%dT%H%M%SZ")
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "meta.json").write_text(json.dumps(_meta(args.bucket, len(selected)), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    errors: list[str] = []
    with (run_dir / "predictions.jsonl").open("w", encoding="utf-8") as file:
        for case, runner in selected:
            stop = False
            try:
                prediction = runner(case)
            except Exception as error:  # noqa: BLE001 - preserve every failed case in the run artifact
                bucket = {run_complaint_case: "complaint", run_route_case: "routing", run_photo_case: "vlm"}[runner]
                prediction = {"id": case["id"], "bucket": bucket, "error": {"type": type(error).__name__, "message": str(error)}}
                errors.append(case["id"])
                stop = isinstance(error, (LlmRateLimitedError, LlmUnavailableError))
            file.write(json.dumps(prediction, ensure_ascii=False) + "\n")
            file.flush()
            if stop:
                break
    print(run_dir)
    if errors:
        raise SystemExit(f"model calls failed for {len(errors)} cases: {', '.join(errors)}")


if __name__ == "__main__":
    main()
