"""Score saved complaint predictions without calling a model.

Run from the repository root: PYTHONPATH=.:ai/src ai/.venv/bin/python -m evals.complaint.score RUN_DIR
"""

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent
SLOTS = ("issue_type", "location", "symptom", "occurred_at")
ISSUE_TYPES = ("water_supply", "drain", "leak", "heating", "electricity", "mold", "pest", "facility", "noise", "other", "null")
ROUTES = ("complaint", "knowledge", "clarify")
WATER_TYPES = {"water_supply", "drain", "leak"}
LOCATION_ALIASES = {"욕실": "화장실", "부엌": "주방", "화장실겸욕실": "화장실"}
FACT_ALIASES = {"에러": ("에러", "오류", "ERROR")}
BUCKETS = {"complaint", "routing"}


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def validate_buckets(requested: str, predictions: list[dict]) -> None:
    available = BUCKETS | ({"vlm"} if (ROOT / "photo_cases.jsonl").is_file() else set())
    expected = available if requested == "all" else {requested}
    actual = {row["bucket"] for row in predictions}
    missing = expected - actual
    extra = actual - expected
    if missing or extra:
        raise ValueError(f"missing buckets: {sorted(missing)}; extra buckets: {sorted(extra)}")


def _gold_files(bucket: str) -> list[Path]:
    files = []
    if bucket in ("complaint", "all"):
        files.extend((ROOT / "complaint_cases.jsonl", ROOT / "complaint_conversations.jsonl"))
    if bucket in ("routing", "all"):
        files.append(ROOT / "routing_cases.jsonl")
    if bucket in ("vlm", "all") and (ROOT / "photo_cases.jsonl").is_file():
        files.append(ROOT / "photo_cases.jsonl")
        files.extend(sorted(ROOT / row["image_path"] for row in read_jsonl(ROOT / "photo_cases.jsonl")))
    return files


def gold_fingerprint(bucket: str) -> str:
    digest = hashlib.sha256()
    for path in _gold_files(bucket):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def input_fingerprint(bucket: str) -> str:
    inputs: list[dict] = []
    if bucket in ("complaint", "all"):
        cases = read_jsonl(ROOT / "complaint_cases.jsonl")
        cases.extend(turn for row in read_jsonl(ROOT / "complaint_conversations.jsonl") for turn in row["turns"])
        inputs.extend({"id": row["id"], "today": row["today"], "request": row["request"]} for row in cases)
    if bucket in ("routing", "all"):
        inputs.extend({"id": row["id"], "request": row["request"]} for row in read_jsonl(ROOT / "routing_cases.jsonl"))
    if bucket in ("vlm", "all") and (ROOT / "photo_cases.jsonl").is_file():
        inputs.extend({"id": row["id"], "image_sha256": hashlib.sha256((ROOT / row["image_path"]).read_bytes()).hexdigest()} for row in read_jsonl(ROOT / "photo_cases.jsonl"))
    return hashlib.sha256(json.dumps(inputs, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def _indexed(rows: list[dict], what: str) -> dict[str, dict]:
    result = {row["id"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"duplicate {what} IDs")
    return result


def _paired(cases: list[dict], predictions: list[dict]) -> list[tuple[dict, dict]]:
    gold = _indexed(cases, "gold")
    pred = _indexed(predictions, "prediction")
    missing = gold.keys() - pred.keys()
    extra = pred.keys() - gold.keys()
    if missing or extra:
        raise ValueError(f"missing predictions: {sorted(missing)}; extra predictions: {sorted(extra)}")
    failed = [key for key, row in pred.items() if row.get("error")]
    if failed:
        raise ValueError(f"model errors for cases: {failed}")
    return [(case, pred[case["id"]]) for case in cases]


def _value(slot: str, value: object) -> object:
    if value is None:
        return None
    if slot == "occurred_at":
        return str(value)[:10]
    if slot == "location":
        compact = "".join(str(value).split())
        return LOCATION_ALIASES.get(compact, compact)
    if slot == "symptom":
        return "".join(str(value).split())
    return value


def _ratio(numerator: float, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def _macro_f1(gold: list[str], pred: list[str], labels: tuple[str, ...] | None = None) -> float | None:
    if not gold:
        return None
    labels = labels or tuple(set(gold) | set(pred))
    scores = []
    for label in labels:
        tp = sum(g == p == label for g, p in zip(gold, pred, strict=True))
        fp = sum(g != label and p == label for g, p in zip(gold, pred, strict=True))
        fn = sum(g == label and p != label for g, p in zip(gold, pred, strict=True))
        scores.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0)
    return round(sum(scores) / len(scores), 4)


def score_cases(cases: list[dict], predictions: list[dict]) -> dict:
    pairs = _paired(cases, predictions)
    gold_types = [case["expected"]["delta"].get("issue_type") or "null" for case, _ in pairs]
    pred_types = [pred["delta"].get("issue_type") or "null" for _, pred in pairs]
    null_cases = sum(label == "null" for label in gold_types)
    null_filled = sum(g == "null" and p != "null" for g, p in zip(gold_types, pred_types, strict=True))
    per_class = {}
    for label in ISSUE_TYPES:
        tp = sum(g == p == label for g, p in zip(gold_types, pred_types, strict=True))
        fp = sum(g != label and p == label for g, p in zip(gold_types, pred_types, strict=True))
        fn = sum(g == label and p != label for g, p in zip(gold_types, pred_types, strict=True))
        per_class[label] = {"cases": tp + fn, "tp": tp, "fp": fp, "fn": fn,
                            "precision": _ratio(tp, tp + fp), "recall": _ratio(tp, tp + fn),
                            "f1": _ratio(2 * tp, 2 * tp + fp + fn) or 0}
    water_cases = sum(g in WATER_TYPES for g in gold_types)
    water_correct = sum(g in WATER_TYPES and g == p for g, p in zip(gold_types, pred_types, strict=True))

    delta_correct = 0
    draft_correct = 0
    structured_correct = 0
    field_correct: Counter[str] = Counter()
    empty_slots = 0
    over_extracted = 0
    history_slots = 0
    history_reextracted = 0
    keyword_scores: list[float] = []
    missing_correct = 0
    asked_correct = 0
    missing_turns = 0
    state_correct = 0
    reply_nonempty = 0
    llm_missing_cases = 0
    llm_missing_correct = 0
    for case, pred in pairs:
        expected = case["expected"]
        delta_matches = {
            slot: _value(slot, expected["delta"].get(slot)) == _value(slot, pred["delta"].get(slot))
            for slot in SLOTS
        }
        delta_correct += all(delta_matches.values())
        field_correct.update(slot for slot, match in delta_matches.items() if match)
        for slot in ("location", "symptom", "occurred_at"):
            if expected["delta"].get(slot) is None:
                empty_slots += 1
                over_extracted += pred["delta"].get(slot) is not None
        for slot in SLOTS:
            if expected.get("origin", {}).get(slot) == "history":
                history_slots += 1
                history_reextracted += pred["delta"].get(slot) is not None
        draft_correct += all(
            _value(slot, expected["draft"].get(slot)) == _value(slot, pred["draft"].get(slot))
            for slot in SLOTS
        ) and expected["draft"].get("image_urls", []) == pred["draft"].get("image_urls", [])
        structured_correct += all(
            _value(slot, expected["draft"].get(slot)) == _value(slot, pred["draft"].get(slot))
            for slot in ("issue_type", "location", "occurred_at")
        ) and bool(expected["draft"].get("symptom")) == bool(pred["draft"].get("symptom"))
        keywords = expected.get("symptom_keywords", [])
        if keywords:
            symptom = str(pred["delta"].get("symptom") or "")
            keyword_scores.append(sum(word in symptom for word in keywords) / len(keywords))
        missing_correct += expected["missing_fields"] == pred["missing_fields"]
        if expected["missing_fields"]:
            missing_turns += 1
            asked_correct += expected["missing_fields"][0] == pred.get("asked_field")
        state_correct += pred.get("state") == ("collecting" if expected["missing_fields"] else None)
        reply_nonempty += bool(str(pred.get("reply") or "").strip())
        if "llm_missing" in pred:
            llm_missing_cases += 1
            llm_missing_correct += set(expected["missing_fields"]) == set(pred["llm_missing"])

    return {
        "classification": {
            "cases": len(pairs),
            "macro_f1": round(sum(row["f1"] for row in per_class.values()) / len(ISSUE_TYPES), 4) if pairs else None,
            "per_class": per_class,
            "water_type_cases": water_cases,
            "water_type_accuracy": _ratio(water_correct, water_cases),
            "accuracy": _ratio(sum(g == p for g, p in zip(gold_types, pred_types, strict=True)), len(pairs)),
            "null_overfill_rate": _ratio(null_filled, null_cases),
            "final_type_accuracy": _ratio(sum(case["expected"]["draft"]["issue_type"] == pred["draft"].get("issue_type") for case, pred in pairs), len(pairs)),
        },
        "extraction": {
            "cases": len(pairs),
            "delta_exact": _ratio(delta_correct, len(pairs)),
            "final_draft_exact": _ratio(draft_correct, len(pairs)),
            "final_structured_accuracy": _ratio(structured_correct, len(pairs)),
            "slot_accuracy": {slot: _ratio(field_correct[slot], len(pairs)) for slot in SLOTS},
            "over_extraction_rate": _ratio(over_extracted, empty_slots),
            "history_reextraction_rate": _ratio(history_reextracted, history_slots),
            "symptom_keyword_recall": round(sum(keyword_scores) / len(keyword_scores), 4) if keyword_scores else None,
            "symptom_keyword_cases": len(keyword_scores),
        },
        "dialogue": {
            "missing_fields_accuracy": _ratio(missing_correct, len(pairs)),
            "next_missing_slot_cases": missing_turns,
            # Structural asked_field only; reply wording/meaning needs separate review.
            "next_missing_slot_accuracy": _ratio(asked_correct, missing_turns),
            "state_accuracy": _ratio(state_correct, len(pairs)),
            "reply_nonempty_rate": _ratio(reply_nonempty, len(pairs)),
            "llm_missing_accuracy": _ratio(llm_missing_correct, llm_missing_cases),
        },
    }


def score_routes(cases: list[dict], predictions: list[dict]) -> dict:
    pairs = _paired(cases, predictions)
    classifier = [(case, pred) for case, pred in pairs if case["expected"]["entry_node"] == "classify_intent"]
    gold = [case["expected"]["route"] for case, _ in classifier]
    pred = [row["route"] for _, row in classifier]
    cross = sum({g, p} == {"complaint", "knowledge"} for g, p in zip(gold, pred, strict=True))
    return {
        "cases": len(pairs),
        "classifier_cases": len(classifier),
        "macro_f1": _macro_f1(gold, pred, ROUTES),
        "accuracy": _ratio(sum(g == p for g, p in zip(gold, pred, strict=True)), len(classifier)),
        "fatal_cross_route_cases": sum(g in {"complaint", "knowledge"} for g in gold),
        "fatal_cross_route_rate": _ratio(cross, sum(g in {"complaint", "knowledge"} for g in gold)),
        "entry_accuracy": _ratio(sum(case["expected"]["entry_node"] == row["entry_node"] for case, row in pairs), len(pairs)),
        "end_to_end_route_accuracy": _ratio(sum(case["expected"]["route"] == row["route"] for case, row in pairs), len(pairs)),
    }


def score_conversations(conversations: list[dict], predictions: list[dict]) -> dict:
    turns = [turn for conversation in conversations for turn in conversation["turns"]]
    pairs = _paired(turns, predictions)
    by_id = {case["id"]: pred for case, pred in pairs}
    final_correct = 0
    completion_correct = 0
    for conversation in conversations:
        final = conversation["turns"][-1]
        actual = by_id[final["id"]]["draft"]
        final_correct += all(_value(slot, final["expected"]["draft"].get(slot)) == _value(slot, actual.get(slot)) for slot in SLOTS)
        completed = next((index for index, turn in enumerate(conversation["turns"], 1) if not by_id[turn["id"]]["missing_fields"]), None)
        completion_correct += completed == conversation["minimum_completion_turns"]
    return {
        "conversations": len(conversations),
        "teacher_forced_final_exact": _ratio(final_correct, len(conversations)),
        "completion_turn_accuracy": _ratio(completion_correct, len(conversations)),
    }


def _edit_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for index, char in enumerate(left, 1):
        current = [index]
        for other_index, other in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[other_index] + 1, previous[other_index - 1] + (char != other)))
        previous = current
    return previous[-1]


def _ocr(value: object) -> str:
    return "".join(str(value or "").upper().split())


def _key_tokens(value: object, gold_tokens: set[str]) -> set[str]:
    ocr = str(value or "").upper()
    words = re.findall(r"[A-Z0-9]+(?:[:.][A-Z0-9]+)*", ocr)
    marked_codes = set(re.findall(r"(?:ERROR|ERR|에러)\s*[:：]?\s*([A-Z]{2,4})\b", ocr))
    # Unmarked alphabetic words such as ON/TEMP are ordinary OCR, not extra error codes.
    return {word for word in words if any(char.isdigit() for char in word) or word in gold_tokens or word in marked_codes}


def score_photos(cases: list[dict], predictions: list[dict]) -> dict:
    pairs = _paired(cases, predictions)
    fixture_types = {
        "synthetic_drawings" if "synthetic_drawing" in case.get("tags", [])
        else "synthetic_photos" if case.get("source_type") == "synthetic"
        else case.get("source_type", "unknown")
        for case, _ in pairs
    }
    count_ok = 0
    forbidden_hits = 0
    device_hits = 0
    required_hit = 0
    required_total = 0
    token_tp = 0
    token_fp = 0
    token_fn = 0
    ocr_exact = 0
    ocr_chars = 0
    ocr_errors = 0
    text_cases = 0
    no_text = 0
    false_text = 0
    for case, pred in pairs:
        images = pred.get("images", [])
        count_ok += len(images) == 1
        observation = images[0] if len(images) == 1 else {}
        summary = str(observation.get("summary") or "")
        actual_ocr = _ocr(observation.get("ocr_text"))
        forbidden_hits += any(phrase in summary for phrase in case["forbidden"])
        device_hits += case["device"] in summary
        for fact in case["required_facts"]:
            required_total += 1
            required_hit += any(alias in summary for alias in FACT_ALIASES.get(fact, (fact,)))
        gold_tokens = {token.upper() for token in case["key_tokens"]}
        predicted_tokens = _key_tokens(observation.get("ocr_text"), gold_tokens)
        token_tp += len(gold_tokens & predicted_tokens)
        token_fp += len(predicted_tokens - gold_tokens)
        token_fn += len(gold_tokens - predicted_tokens)
        if case["gold_ocr"] is None:
            no_text += 1
            false_text += bool(actual_ocr)
        else:
            text_cases += 1
            gold_ocr = _ocr(case["gold_ocr"])
            ocr_exact += actual_ocr == gold_ocr
            ocr_chars += len(gold_ocr)
            ocr_errors += _edit_distance(gold_ocr, actual_ocr)
    return {
        "cases": len(pairs),
        "fixture_type": next(iter(fixture_types)) if len(fixture_types) == 1 else "mixed",
        "image_count_accuracy": _ratio(count_ok, len(pairs)),
        "required_fact_recall": _ratio(required_hit, required_total),
        "device_mention_rate_literal": _ratio(device_hits, len(pairs)),
        "key_token_tp": token_tp,
        "key_token_fp": token_fp,
        "key_token_fn": token_fn,
        "key_token_precision": _ratio(token_tp, token_tp + token_fp),
        "key_token_recall": _ratio(token_tp, token_tp + token_fn),
        "key_token_f1": _ratio(2 * token_tp, 2 * token_tp + token_fp + token_fn),
        "forbidden_phrase_rate": _ratio(forbidden_hits, len(pairs)),
        "ocr_exact": _ratio(ocr_exact, text_cases),
        "ocr_cer": _ratio(ocr_errors, ocr_chars),
        "no_text_false_positive_rate": _ratio(false_text, no_text),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir
    predictions = read_jsonl(run_dir / "predictions.jsonl")
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    bucket = meta["bucket"]
    validate_buckets(bucket, predictions)
    if len(predictions) != meta["cases"]:
        raise ValueError(f"prediction count {len(predictions)} != run metadata {meta['cases']}")
    current_input_hash = input_fingerprint(bucket)
    if current_input_hash != meta["input_sha256"]:
        raise ValueError("evaluation inputs changed since prediction run; rerun the model")
    by_bucket: dict[str, list[dict]] = {}
    for row in predictions:
        by_bucket.setdefault(row["bucket"], []).append(row)
    scores: dict[str, object] = {}
    if "complaint" in by_bucket:
        standalone = read_jsonl(ROOT / "complaint_cases.jsonl")
        conversations = read_jsonl(ROOT / "complaint_conversations.jsonl")
        text_scores = score_cases(standalone + [turn for row in conversations for turn in row["turns"]], by_bucket["complaint"])
        scores.update(text_scores)
        conversation_ids = {turn["id"] for row in conversations for turn in row["turns"]}
        scores["conversations"] = score_conversations(conversations, [row for row in by_bucket["complaint"] if row["id"] in conversation_ids])
    if "routing" in by_bucket:
        scores["routing"] = score_routes(read_jsonl(ROOT / "routing_cases.jsonl"), by_bucket["routing"])
    if "vlm" in by_bucket:
        scores["vlm"] = score_photos(read_jsonl(ROOT / "photo_cases.jsonl"), by_bucket["vlm"])
    if not scores:
        raise ValueError("no supported prediction buckets")
    scores["provenance"] = {
        "bucket": bucket,
        "gold_sha256": gold_fingerprint(bucket),
        "input_sha256": current_input_hash,
        "scorer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (run_dir / "scores.json").write_text(json.dumps(scores, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(scores, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
