"""이미지 분기·대표이미지·draft 병합을 로컬 docker API로 눈으로 확인한다.

사용:
  docker compose up -d
  uv run python ai/scripts/test_image_scenarios.py

.env의 PRESIGNED_URL_1~3을 읽어 시나리오를 순서대로 돌리고, 응답 핵심 필드만
찍는다. 정답을 채점하지 않는다 — 배포 전 "분기/병합이 안 깨졌는지" 눈으로
보는 스모크 테스트다.

한 메시지당 이미지 1장 제한(기존 3장 → 1장)이 적용된 뒤라, 여러 장을 보내는
흐름은 한 메시지에 몰아넣지 않고 턴을 나눠 보낸다.
"""

import json
import os
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from uuid import uuid4

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

BASE_URL = f"http://127.0.0.1:{os.environ.get('AI_PORT', '8000')}"
URLS = [os.environ[f"PRESIGNED_URL_{i}"] for i in range(1, 4)]

DRAFT_KEYS = {
    "issue_type", "location", "symptom", "occurred_at", "attachment_ids",
    "issueType", "location", "symptom", "occurredAt", "attachmentIds",
}


def converse(payload: dict) -> dict:
    request = Request(
        f"{BASE_URL}/api/v3/ai/converse",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        response = urlopen(request, timeout=300)
    except HTTPError as error:
        response = error
    with response:
        body = json.load(response)
        if response.status >= 400:
            print("  HTTP", response.status, json.dumps(body, ensure_ascii=False)[:300])
            return {}
        return body


def image_attachment(attachment_id: int, url: str) -> dict:
    return {"attachmentId": attachment_id, "url": url}


def base_request(*, text, images, history, draft, route, state) -> dict:
    trace = f"scn-{uuid4().hex[:8]}"
    return {
        "building_id": 101,
        "room_no": "301",
        "resident_id": "image-scenario-test",
        "conversation_id": trace,
        "turn_id": trace,
        "trace_id": trace,
        "current_route": route,
        "current_complaint_state": state,
        "message": {"message_id": trace, "text": text, "images": images},
        "conversation_history": history,
        "complaint_draft": draft,
    }


def report(label: str, body: dict) -> dict:
    print(f"\n=== {label} ===")
    if not body:
        return {}
    data = body["data"]
    result = data["result"]
    print("route:", data["route"], "| next_state:", data.get("next_complaint_state"))
    print("reply:", data["reply"][:120])
    print("missing_fields:", result["missing_fields"])
    analysis = result.get("image_analysis")
    if analysis:
        for img in analysis["images"]:
            print(f"  image[{img['attachmentId']}] summary: {(img.get('summary') or '')[:80]}")
    draft = result.get("complaint_draft")
    if draft:
        print("draft (raw keys):", json.dumps(draft, ensure_ascii=False))
    return result


def strip_draft(draft: dict | None) -> dict | None:
    if not draft:
        return None
    return {k: v for k, v in draft.items() if k in DRAFT_KEYS}


def run_turn(label: str, *, text, images, prev_req=None, prev_body=None) -> tuple[dict, dict]:
    """prev_req/prev_body가 있으면 그 결과를 이어받아 다음 턴을 보낸다 (멀티턴 체이닝)."""
    if prev_req is None:
        req = base_request(
            text=text, images=images, history=[], draft=None,
            route="complaint", state="collecting",
        )
    else:
        prev_result = prev_body["data"]["result"]
        prev_analysis = prev_result.get("image_analysis") or {"images": []}
        history = [
            *prev_req["conversation_history"],
            {
                "message_id": prev_req["message"]["message_id"],
                "role": "user",
                "text": prev_req["message"]["text"],
                "images": prev_analysis["images"],
            },
            {
                "message_id": f"{prev_req['turn_id']}-assistant",
                "role": "assistant",
                "text": prev_body["data"]["reply"],
            },
        ]
        req = base_request(
            text=text, images=images, history=history,
            draft=strip_draft(prev_result.get("complaint_draft")),
            route="complaint",
            state=prev_body["data"].get("next_complaint_state"),
        )
    body = converse(req)
    report(label, body)
    return req, body


def main() -> None:
    # 1. 이미지 1개 단독
    run_turn("1. 이미지 1개 단독", text=None, images=[image_attachment(1, URLS[0])])

    # 2. 신규 제약 확인: 한 메시지에 이미지 2장을 넣으면 거부되는가
    body = converse(
        base_request(
            text=None,
            images=[image_attachment(1, URLS[0]), image_attachment(2, URLS[1])],
            history=[], draft=None, route="complaint", state="collecting",
        )
    )
    print("\n=== 2. 한 메시지에 이미지 2장 (제약 위반 시도) ===")
    if body:
        print("  주의: 거부되지 않고 200으로 처리됨 — 계약에 아직 1장 제한이 없음")
        report("2-결과", body)
    else:
        print("  422/4xx로 거부됨 (제약 적용 확인)")

    # 3. 텍스트+이미지 1개 (단일턴)
    run_turn(
        "3. 텍스트+이미지 1개",
        text="화장실 세면대 아래에서 물이 새요",
        images=[image_attachment(1, URLS[0])],
    )

    # 4. 멀티턴: 사진만 1장씩 3턴에 걸쳐 전송 → 누적/대표이미지 재계산 확인
    req, body = run_turn("4-1. 사진만 (1/3)", text=None, images=[image_attachment(1, URLS[0])])
    req, body = run_turn("4-2. 사진만 (2/3)", text=None, images=[image_attachment(2, URLS[1])], prev_req=req, prev_body=body)
    run_turn("4-3. 사진만 (3/3)", text=None, images=[image_attachment(3, URLS[2])], prev_req=req, prev_body=body)

    # 5. 멀티턴: 텍스트+사진1장으로 완성 → 사진 1장 더 추가
    req, body = run_turn(
        "5-1. 텍스트+사진1장으로 완성",
        text="301호 화장실 바닥에서 누수가 발생했어요",
        images=[image_attachment(1, URLS[0])],
    )
    run_turn("5-2. 사진 1장 추가", text=None, images=[image_attachment(2, URLS[1])], prev_req=req, prev_body=body)


if __name__ == "__main__":
    main()
