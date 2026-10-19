"""Run without API calls: uv run python scripts/check_bench_extraction_split.py."""

import contextlib
import io
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import bench_extraction_split as bench


def main():
    response = '{"complaint_switch":"same","location":"화장실","symptom":null}'
    with patch.object(bench, "generate_text", return_value=response) as generate:
        row = bench.call_b(bench.CASES[1][1], bench.CASES[1][2], "화장실이요", None)
        assert row["calls"] == 1, "existing draft was ignored"
        assert generate.call_count == 1
        assert "reasoning_effort" not in generate.call_args.kwargs
    with patch.object(
        bench, "generate_text", side_effect=[response, "어떤 증상인가요?"]
    ) as generate:
        row = bench.call_b(None, [], "화장실이요", "low")
        assert row["calls"] == 2 and row["reply"] == "어떤 증상인가요?"
        assert all(
            c.kwargs["reasoning_effort"] == "low" for c in generate.call_args_list
        )

    with patch.object(
        bench, "generate_text", side_effect=[response, "어떤 증상인가요?"]
    ) as generate:
        bench.call_b(None, [], "화장실이요", "medium", "low")
        assert [c.kwargs["reasoning_effort"] for c in generate.call_args_list] == [
            "medium",
            "low",
        ]

    order = []

    def a(*args):
        order.append(("A", args[3:]))
        return {"seconds": 2.0}

    def b(*args):
        order.append(("B", args[3:]))
        return {"seconds": 1.0}

    output = io.StringIO()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "results.jsonl"
        with (
            patch.object(bench, "call_a", a),
            patch.object(bench, "call_b", b),
            patch("sys.argv", ["bench", "--runs", "3", "--output", str(path)]),
            contextlib.redirect_stdout(output),
        ):
            bench.main()
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(rows) == 18
        review = path.with_suffix(".md").read_text()
        assert "생성 답변 원문" in review and "기존 초안" in review
        assert "B/low+medium" in review
    variants = [("A", ("low",)), ("B", ("low", "low")), ("B", ("low", "medium"))]
    expected = [item for i in range(3) for item in variants[i:] + variants[:i]]
    assert order == expected * len(bench.CASES), order
    print("bench checks passed")


if __name__ == "__main__":
    main()
