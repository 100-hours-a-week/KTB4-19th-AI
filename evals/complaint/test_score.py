import pytest

from evals.complaint.score import (
    score_cases,
    score_photos,
    score_routes,
    validate_buckets,
)


def test_score_separates_model_delta_from_final_draft():
    case = {
        "id": "t1",
        "request": {"complaint_draft": {"issue_type": "leak", "location": "화장실", "symptom": "물이 샘", "occurred_at": None, "image_urls": []}},
        "expected": {
            "delta": {"issue_type": None, "location": None, "symptom": None, "occurred_at": None},
            "origin": {"issue_type": "history", "location": "history", "symptom": "history", "occurred_at": "absent"},
            "draft": {"issue_type": "leak", "location": "화장실", "symptom": "물이 샘", "occurred_at": None, "image_urls": []},
            "missing_fields": [],
            "asked_field": None,
            "symptom_keywords": [],
        },
    }
    correct = {"id": "t1", "bucket": "complaint", "delta": case["expected"]["delta"], "draft": case["expected"]["draft"], "missing_fields": [], "state": None}
    inflated = {**correct, "delta": {**correct["delta"], "issue_type": "leak", "location": "화장실"}}

    assert score_cases([case], [correct])["extraction"]["delta_exact"] == 1
    result = score_cases([case], [inflated])
    assert result["extraction"]["delta_exact"] < 1
    assert result["extraction"]["final_draft_exact"] == 1
    assert result["extraction"]["over_extraction_rate"] > 0
    assert result["extraction"]["history_reextraction_rate"] > 0


def test_score_rejects_missing_predictions():
    with pytest.raises(ValueError, match="missing.*t1"):
        score_cases([{"id": "t1", "expected": {}}], [])


def test_routing_reports_fatal_cross_route_error_and_skips_graph_bypass():
    cases = [
        {"id": "a", "expected": {"route": "complaint", "entry_node": "classify_intent"}},
        {"id": "b", "expected": {"route": "complaint", "entry_node": "complaint"}},
    ]
    predictions = [
        {"id": "a", "route": "knowledge", "entry_node": "classify_intent"},
        {"id": "b", "route": "complaint", "entry_node": "complaint"},
    ]
    result = score_routes(cases, predictions)
    assert result["classifier_cases"] == 1
    assert result["fatal_cross_route_rate"] == 1
    assert result["entry_accuracy"] == 1


def test_photo_score_catches_missing_code_and_false_ocr():
    cases = [
        {"id": "p1", "device": "세탁기", "key_tokens": ["LE"], "required_facts": ["세탁기", "LE"], "forbidden": ["누수"], "gold_ocr": "LE", "anomaly": True},
        {"id": "p2", "device": "벽", "key_tokens": [], "required_facts": ["벽"], "forbidden": ["곰팡이"], "gold_ocr": None, "anomaly": False},
    ]
    predictions = [
        {"id": "p1", "images": [{"summary": "세탁기에 오류가 보임", "ocr_text": "L1"}]},
        {"id": "p2", "images": [{"summary": "벽에 곰팡이", "ocr_text": "E4"}]},
    ]
    result = score_photos(cases, predictions)
    assert result["key_token_recall"] == 0
    assert result["forbidden_phrase_rate"] == 0.5
    assert result["no_text_false_positive_rate"] == 1
    assert result["ocr_exact"] == 0


def test_all_run_rejects_missing_prediction_bucket():
    with pytest.raises(ValueError, match="missing buckets"):
        validate_buckets("all", [{"id": "a", "bucket": "complaint"}])


def test_all_run_requires_only_current_gold_buckets():
    validate_buckets("all", [{"id": "a", "bucket": "complaint"}, {"id": "b", "bucket": "routing"}])


def test_classification_uses_fixed_labels_and_water_slice():
    cases = []
    predictions = []
    for index, (gold, predicted) in enumerate((("water_supply", "drain"), ("drain", "drain"), ("leak", "leak"))):
        expected = {"delta": {"issue_type": gold}, "draft": {"issue_type": gold}, "missing_fields": [], "asked_field": None}
        cases.append({"id": str(index), "expected": expected})
        predictions.append({"id": str(index), "delta": {"issue_type": predicted}, "draft": {"issue_type": gold}, "missing_fields": []})
    result = score_cases(cases, predictions)["classification"]
    assert result["macro_f1"] == 0.1515  # (2/3 + 1) / 11
    assert result["water_type_accuracy"] == 0.6667
    assert len(result["per_class"]) == 11
    assert result["per_class"]["water_supply"] == {"cases": 1, "tp": 0, "fp": 0, "fn": 1, "precision": None, "recall": 0, "f1": 0}
    assert result["per_class"]["drain"]["fp"] == 1


def test_next_missing_slot_uses_only_missing_turns():
    cases = [
        {"id": "need", "expected": {"delta": {}, "draft": {"issue_type": None}, "missing_fields": ["location", "symptom"], "asked_field": "location"}},
        {"id": "done", "expected": {"delta": {}, "draft": {"issue_type": None}, "missing_fields": [], "asked_field": None}},
    ]
    predictions = [
        {"id": "need", "delta": {}, "draft": {}, "missing_fields": ["location", "symptom"], "asked_field": "symptom"},
        {"id": "done", "delta": {}, "draft": {}, "missing_fields": [], "asked_field": None},
    ]
    result = score_cases(cases, predictions)["dialogue"]
    assert result["next_missing_slot_cases"] == 1
    assert result["next_missing_slot_accuracy"] == 0
    assert score_cases(cases[1:], predictions[1:])["dialogue"]["next_missing_slot_accuracy"] is None


def test_cross_route_denominator_excludes_other_gold_routes():
    cases = [
        {"id": "a", "expected": {"route": "complaint", "entry_node": "classify_intent"}},
        {"id": "b", "expected": {"route": "clarify", "entry_node": "classify_intent"}},
    ]
    predictions = [
        {"id": "a", "route": "knowledge", "entry_node": "classify_intent"},
        {"id": "b", "route": "complaint", "entry_node": "classify_intent"},
    ]
    result = score_routes(cases, predictions)
    assert result["classifier_cases"] == 2
    assert result["fatal_cross_route_cases"] == 1
    assert result["fatal_cross_route_rate"] == 1


def test_route_macro_f1_uses_all_three_routes_on_slices():
    cases = [{"id": "a", "expected": {"route": "complaint", "entry_node": "classify_intent"}}]
    predictions = [{"id": "a", "route": "complaint", "entry_node": "classify_intent"}]
    assert score_routes(cases, predictions)["macro_f1"] == 0.3333


def test_photo_key_tokens_are_ocr_only_boundary_matched_and_count_false_positives():
    cases = [
        {"id": "a", "device": "기기", "key_tokens": ["E1"], "required_facts": [], "forbidden": [], "gold_ocr": "E1"},
        {"id": "b", "device": "기기", "key_tokens": [], "required_facts": [], "forbidden": [], "gold_ocr": None},
    ]
    predictions = [
        {"id": "a", "images": [{"summary": "E1", "ocr_text": "ERROR E10"}]},
        {"id": "b", "images": [{"summary": "", "ocr_text": "L1"}]},
    ]
    result = score_photos(cases, predictions)
    assert (result["key_token_tp"], result["key_token_fp"], result["key_token_fn"]) == (0, 2, 1)
    assert result["key_token_precision"] == 0
    assert result["key_token_recall"] == 0
    assert result["key_token_f1"] == 0


def test_photo_key_token_micro_scores_include_hits_and_extra_codes():
    cases = [{"id": "a", "device": "기기", "key_tokens": ["LE", "01:20"], "required_facts": [], "forbidden": [], "gold_ocr": "LE 01:20"}]
    predictions = [{"id": "a", "images": [{"summary": "", "ocr_text": "ERROR LE 01:20 E10"}]}]
    result = score_photos(cases, predictions)
    assert (result["key_token_tp"], result["key_token_fp"], result["key_token_fn"]) == (2, 1, 0)
    assert result["key_token_precision"] == 0.6667
    assert result["key_token_recall"] == 1
    assert result["key_token_f1"] == 0.8


def test_photo_fixture_type_changes_when_generated_photo_is_added():
    case = {"id": "a", "device": "기기", "key_tokens": [], "required_facts": [], "forbidden": [], "gold_ocr": None, "source_type": "synthetic", "tags": []}
    prediction = {"id": "a", "images": [{"summary": "", "ocr_text": None}]}
    assert score_photos([case], [prediction])["fixture_type"] == "synthetic_photos"


def test_photo_key_token_score_ignores_ordinary_display_words():
    case = {"id": "a", "device": "기기", "key_tokens": ["E4"], "required_facts": [], "forbidden": [], "gold_ocr": "ON E4"}
    prediction = {"id": "a", "images": [{"summary": "", "ocr_text": "ON E4"}]}
    result = score_photos([case], [prediction])
    assert (result["key_token_tp"], result["key_token_fp"], result["key_token_fn"]) == (1, 0, 0)
