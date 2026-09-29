import json
import sys

import pytest

from evals.complaint.baseline import compare, main, select_metrics


def test_baseline_compare_respects_metric_direction():
    baseline = {"classification": {"macro_f1": 0.8, "null_overfill_rate": 0.1}}
    current = {"classification": {"macro_f1": 0.75, "null_overfill_rate": 0.15}}
    failures = compare(select_metrics(baseline), select_metrics(current), tolerance=0.03)
    assert {item["metric"] for item in failures} == {"classification.macro_f1", "classification.null_overfill_rate"}


def test_baseline_uses_ocr_precision_and_recall_not_keyword_proxy():
    scores = {
        "classification": {"water_type_accuracy": 0.8},
        "extraction": {"symptom_keyword_recall": 0.9},
        "vlm": {"key_token_f1": 0.7, "key_token_recall": 1.0},
    }
    selected = select_metrics(scores)
    assert "classification.water_type_accuracy" in selected
    assert "vlm.key_token_f1" in selected
    assert "vlm.key_token_recall" not in selected
    assert "extraction.symptom_keyword_recall" not in selected


def test_baseline_compare_requires_measured_metric():
    with pytest.raises(ValueError, match="missing baseline"):
        compare({}, {"classification.macro_f1": 0.8})
    with pytest.raises(ValueError, match="missing current"):
        compare({"classification.macro_f1": 0.8, "routing.macro_f1": 0.8}, {"classification.macro_f1": 0.8})


def test_baseline_rejects_nonfinite_scores():
    with pytest.raises(ValueError, match="finite"):
        compare({"classification.macro_f1": 0.8}, {"classification.macro_f1": float("nan")})


def test_compare_rejects_changed_gold_even_when_scores_match(tmp_path, monkeypatch):
    provenance = {"bucket": "complaint", "gold_sha256": "new", "input_sha256": "same", "scorer_sha256": "same"}
    (tmp_path / "scores.json").write_text(json.dumps({"classification": {"macro_f1": 0.8}, "provenance": provenance}))
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"metrics": {"classification.macro_f1": 0.8}, "provenance": {**provenance, "gold_sha256": "old"}}))
    monkeypatch.setattr(sys, "argv", ["baseline", "compare", str(tmp_path), "--baseline", str(baseline)])
    with pytest.raises(ValueError, match="gold_sha256 differs"):
        main()
