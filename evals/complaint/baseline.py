"""Save a measured baseline or compare a scored run with it.

Usage: PYTHONPATH=. ai/.venv/bin/python -m evals.complaint.baseline init RUN_DIR
       PYTHONPATH=. ai/.venv/bin/python -m evals.complaint.baseline compare RUN_DIR
"""

import argparse
import json
import math
from pathlib import Path

BASELINE = Path(__file__).parent / "baseline.json"
DIRECTIONS = {
    "classification.macro_f1": "up",
    "classification.water_type_accuracy": "up",
    "classification.null_overfill_rate": "down",
    "extraction.final_structured_accuracy": "up",
    "extraction.over_extraction_rate": "down",
    "extraction.history_reextraction_rate": "down",
    "dialogue.missing_fields_accuracy": "up",
    "dialogue.state_accuracy": "up",
    "conversations.completion_turn_accuracy": "up",
    "routing.macro_f1": "up",
    "routing.fatal_cross_route_rate": "down",
    "vlm.key_token_f1": "up",
    "vlm.ocr_cer": "down",
    "vlm.no_text_false_positive_rate": "down",
}


def select_metrics(scores: dict) -> dict[str, float]:
    selected = {}
    for name in DIRECTIONS:
        section, metric = name.split(".", 1)
        value = scores.get(section, {}).get(metric)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if not math.isfinite(value):
                raise ValueError(f"metric {name} must be finite")
            selected[name] = float(value)
    return selected


def compare(baseline: dict[str, float], current: dict[str, float], tolerance: float = 0.03) -> list[dict]:
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    if any(not math.isfinite(value) for value in (*baseline.values(), *current.values())):
        raise ValueError("baseline and current metrics must be finite")
    missing = current.keys() - baseline.keys()
    if missing:
        raise ValueError(f"missing baseline metrics: {sorted(missing)}")
    disappeared = baseline.keys() - current.keys()
    if disappeared:
        raise ValueError(f"missing current metrics: {sorted(disappeared)}")
    failures = []
    for name, value in current.items():
        previous = baseline[name]
        direction = DIRECTIONS[name]
        regressed = value < previous - tolerance if direction == "up" else value > previous + tolerance
        if regressed:
            failures.append({"metric": name, "baseline": previous, "current": value, "direction": direction})
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("init", "compare"))
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument("--tolerance", type=float, default=0.03)
    args = parser.parse_args()
    scores = json.loads((args.run_dir / "scores.json").read_text(encoding="utf-8"))
    metrics = select_metrics(scores)
    if not metrics:
        raise ValueError("run has no baseline metrics")
    if args.action == "init":
        meta = json.loads((args.run_dir / "meta.json").read_text(encoding="utf-8"))
        payload = {"source_run": str(args.run_dir), "meta": meta, "provenance": scores["provenance"], "metrics": metrics}
        with args.baseline.open("x", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
            file.write("\n")
        print(args.baseline)
        return
    baseline_data = json.loads(args.baseline.read_text(encoding="utf-8"))
    baseline_scope = baseline_data["provenance"]["bucket"]
    current_scope = scores["provenance"]["bucket"]
    if baseline_scope != current_scope:
        raise ValueError(f"baseline scope {baseline_scope} != run scope {current_scope}")
    for field in ("gold_sha256", "input_sha256", "scorer_sha256"):
        if baseline_data["provenance"][field] != scores["provenance"][field]:
            raise ValueError(f"baseline {field} differs; rescore or rerun before comparison")
    failures = compare(baseline_data["metrics"], metrics, args.tolerance)
    print(json.dumps({"status": "FAIL" if failures else "PASS", "failures": failures}, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
