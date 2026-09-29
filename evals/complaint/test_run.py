import json
from hashlib import sha256
from pathlib import Path

import pytest

from evals.complaint.run import (
    _cases,
    run_complaint_case,
    run_photo_case,
    run_route_case,
    validate_cases,
)

ROOT = Path(__file__).parent


def test_run_complaint_calls_model_once_and_scores_raw_delta(monkeypatch):
    case = json.loads((ROOT / "complaint_cases.jsonl").read_text(encoding="utf-8").splitlines()[0])
    calls = []

    def fake_model(system_prompt, user_prompt):
        calls.append(user_prompt)
        return json.dumps({"issue_type": "water_supply", "location": "주방", "symptom": "수도에서 물이 전혀 안 나와요", "occurred_at": None, "missing": [], "reply": ""}, ensure_ascii=False)

    monkeypatch.setattr("zipsai.complaint.node.generate_text", fake_model)
    result = run_complaint_case(case)
    assert len(calls) == 1
    assert "2026-09-28" in calls[0]
    assert result["delta"]["issue_type"] == "water_supply"
    assert result["draft"]["issue_type"] == "water_supply"
    assert result["missing_fields"] == []


def test_route_graph_bypass_does_not_call_model(monkeypatch):
    cases = [json.loads(line) for line in (ROOT / "routing_cases.jsonl").read_text(encoding="utf-8").splitlines()]
    case = next(row for row in cases if row["expected"]["entry_node"] == "complaint")
    monkeypatch.setattr("zipsai.orchestration.intent.generate_text", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("model called")))
    result = run_route_case(case)
    assert result["route"] == "complaint"
    assert result["entry_node"] == "complaint"


def test_all_gold_cases_validate_before_model_calls():
    assert validate_cases(_cases("all")) == 158
    assert _cases("vlm") == []


def test_candidate_photo_needs_approval_and_matching_hash():
    image = ROOT / "photos/candidates/AO-01.jpg"
    case = {"id": "candidate_ao_01", "image_path": "photos/candidates/AO-01.jpg"}
    with pytest.raises(ValueError, match="not approved"):
        validate_cases([(case, run_photo_case)])
    case.update(review_status="approved", image_sha256="wrong")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_cases([(case, run_photo_case)])
    case["image_sha256"] = sha256(image.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="incomplete gold"):
        validate_cases([(case, run_photo_case)])
    case.update(device="보일러", anomaly="unknown", source_type="synthetic", gold_ocr=None, key_tokens=[], required_facts=[], forbidden=[], refs=[], tags=[])
    with pytest.raises(ValueError, match="incomplete gold"):
        validate_cases([(case, run_photo_case)])
    case["anomaly"] = False
    case["required_facts"] = ["보일러"]
    case["refs"] = ["보일러가 보인다."]
    assert validate_cases([(case, run_photo_case)]) == 1
    case["anomaly"] = "unknown"
    assert validate_cases([(case, run_photo_case)]) == 1
