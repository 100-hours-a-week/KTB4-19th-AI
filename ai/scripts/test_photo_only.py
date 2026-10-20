"""Run a photo-only complaint request against the local Docker API."""

import json
import os
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from uuid import uuid4

image_url = os.environ.get("PRESIGNED_URL")
if not image_url:
    raise SystemExit("PRESIGNED_URL 환경변수를 먼저 설정하세요.")

trace_id = f"photo-only-{uuid4().hex[:8]}"
payload = {
    "building_id": 101,
    "room_no": "301",
    "resident_id": "log-test",
    "conversation_id": trace_id,
    "trace_id": trace_id,
    "current_route": "complaint",
    "current_complaint_state": "collecting",
    "message": {"message_id": "photo-1", "text": None, "image_urls": [image_url]},
    "conversation_history": [],
    "complaint_draft": None,
}
request = Request(
    "http://127.0.0.1:8000/api/v3/ai/converse",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)
try:
    print("trace_id:", trace_id, flush=True)
    response = urlopen(request, timeout=120)
except HTTPError as error:
    response = error

with response:
    result = json.load(response)
    print("HTTP:", response.status)
    print("code:", result.get("code", result.get("error", {}).get("code")))
    if response.status >= 400:
        raise SystemExit(1)
    print("reply:", result["data"]["reply"])
    print("missing:", result["data"]["result"]["missing_fields"])
    analyzed = result["data"]["result"]["image_analysis"] is not None
    print("photo:", "analyzed" if analyzed else "failed")
    if not analyzed:
        raise SystemExit(1)
