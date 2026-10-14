"""Manually check the SSE contract of the running AI API.

Usage:
    uv run python scripts/streaming_smoke.py payload.json
"""

import argparse
import json
from pathlib import Path

import httpx


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("payload", type=Path, help="JSON request body for converse")
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8000/api/v3/ai/converse/stream",
    )
    args = parser.parse_args()

    payload = json.loads(args.payload.read_text(encoding="utf-8"))
    event_types: list[str] = []
    complete_response: dict[str, object] | None = None

    with httpx.stream("POST", args.url, json=payload, timeout=None) as response:
        response.raise_for_status()
        if not response.headers.get("content-type", "").startswith(
            "text/event-stream"
        ):
            raise RuntimeError("response is not text/event-stream")

        for line in response.iter_lines():
            if not line.startswith("data: "):
                continue
            event = json.loads(line.removeprefix("data: "))
            event_types.append(event["type"])
            if event["type"] == "complete":
                response = event.get("response")
                if isinstance(response, dict):
                    complete_response = response
            print(event)

    if not event_types or event_types[-1] != "done":
        raise RuntimeError(f"stream ended without done event: {event_types}")

    if "error" in event_types:
        raise RuntimeError(f"stream returned an error event: {event_types}")

    if "complete" not in event_types or complete_response is None:
        raise RuntimeError(
            "successful stream must include complete event: " f"{event_types}"
        )

    if event_types.index("complete") > event_types.index("done"):
        raise RuntimeError("complete event must arrive before done")

    if complete_response.get("code") != "ai_response_success":
        raise RuntimeError("complete.response.code is not ai_response_success")

    response_data = complete_response.get("data")
    if not isinstance(response_data, dict):
        raise TypeError("complete.response.data is missing")

    required_fields = {"route", "reply", "result", "meta"}
    missing_fields = required_fields - response_data.keys()
    if missing_fields:
        raise RuntimeError(
            "complete.response.data is missing fields: "
            f"{', '.join(sorted(missing_fields))}"
        )

    print("complete response validated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
