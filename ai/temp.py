"""knowledge 노드의 미커밋 변경점(사진 맥락 검색 반영) 수동 점검.

실제 LLM·Qdrant 호출 없이 generate_text/search_chunks 등을 고정 응답으로 바꿔서
handle_knowledge() 로직만 돈다. 비용·네트워크 없음.

실행: ai/.venv/bin/python temp.py (ai/ 안에서 실행)
"""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent / "src"))

import zipsai.knowledge.node as node
import zipsai.orchestration.graph as graph
from zipsai.contracts.converse import (
    ConverseRequest,
    HistoryTurn,
    ImageAnalysis,
    ImageObservation,
    Route,
)
from zipsai.knowledge.prompts import IMAGE_QUESTION_PROMPT

CHUNK = SimpleNamespace(
    payload={"doc_id": "d1", "title": "공용시설 안내", "text": "세탁실은 1층이며 22시까지 이용 가능합니다.", "page": 1}
)


def _request(text, images=None, history=None) -> ConverseRequest:
    return ConverseRequest.model_validate(
        {
            "building_id": 1,
            "room_no": "301",
            "resident_id": "demo",
            "conversation_id": "demo",
            "turn_id": "turn-1",
            "trace_id": "trace-1",
            "current_route": Route.KNOWLEDGE,
            "current_complaint_state": None,
            "message": {"message_id": "m1", "text": text, "images": images or []},
            "conversation_history": history or [],
            "complaint_draft": None,
        }
    )


def _fake_generate_text(system_prompt, user_prompt, **_kwargs):
    if system_prompt == IMAGE_QUESTION_PROMPT:
        return "세탁실 위치와 운영시간"
    return "세탁실은 1층에 있고 22시까지 이용하실 수 있습니다."


def check(label, condition):
    print(f"[{'PASS' if condition else 'FAIL'}] {label}")
    assert condition, label


with patch.object(node, "generate_text", side_effect=_fake_generate_text), patch.object(
    node, "search_chunks", return_value=[CHUNK]
), patch.object(node, "encode_question", return_value=[0.0]), patch.object(
    node, "query_encoder", return_value=None
), patch.object(node, "get_client", return_value=None):

    # A) 이미지 없음 -> 기존 동작. query_rewrite 안 타고 질문 그대로 검색.
    calls = []
    with patch.object(node, "encode_question", side_effect=lambda q, **kw: (calls.append(q), [0.0])[1]):
        out = node.handle_knowledge(_request("세탁실 몇 시까지 해요?"))
    check("no image: searches with the original question", calls == ["세탁실 몇 시까지 해요?"])
    check("no image: answers normally", out["reply"].startswith("세탁실은"))

    # B) 이미지 있음 -> query_rewrite 거쳐 재작성된 질문으로 검색하고, 프롬프트에 image_context 반영.
    analysis = ImageAnalysis(images=[ImageObservation(attachmentId=1, summary="세탁실 요금표 사진", ocrText="22:00 마감")])
    calls = []
    with patch.object(node, "encode_question", side_effect=lambda q, **kw: (calls.append(q), [0.0])[1]):
        out = node.handle_knowledge(_request("이거 언제까지 써요?"), image_analysis=analysis)
    check("with image: rewrites the search question", calls == ["세탁실 위치와 운영시간"])
    check("with image: route_result carries no image_analysis by itself", out["result"].image_analysis is None)

    # C) 이번 턴 "사진만" 보냄(clarify 후속) + 직전 user 턴에 질문 있음 -> 그 질문을 재사용(신규 로직).
    # 주의: 이 재사용은 request.message.images가 있을 때만 걸린다 — 사진 없이 텍스트만 비면 안 걸림(D).
    history = [
        HistoryTurn(message_id="h1", role="user", text="세탁실 요금 얼마예요?"),
        HistoryTurn(message_id="h2", role="assistant", text="1층에 있습니다"),
    ]
    new_image = {"attachmentId": 9, "url": "https://example.com/receipt.jpg"}
    calls = []
    with patch.object(node, "encode_question", side_effect=lambda q, **kw: (calls.append(q), [0.0])[1]):
        out = node.handle_knowledge(_request(None, images=[new_image], history=history))
    check("blank text + new photo falls back to last user question", calls == ["세탁실 요금 얼마예요?"])

    # D) 텍스트도 이미지도 없음(history에 질문이 있어도) -> 재사용 안 하고 empty_question 폴백.
    out = node.handle_knowledge(_request(None, history=history))
    check(
        "blank text w/o photo does NOT reuse history question",
        out["reply"] == node.EMPTY_QUESTION_REPLY,
    )
    check("empty_question fallback has no qa_card", out["result"].qa_card_draft is None)

# E) graph._include_image_analysis가 RouteResult에 분석 결과를 붙이는지.
with patch.object(node, "generate_text", side_effect=_fake_generate_text), patch.object(
    node, "search_chunks", return_value=[CHUNK]
), patch.object(node, "encode_question", return_value=[0.0]), patch.object(
    node, "query_encoder", return_value=None
), patch.object(node, "get_client", return_value=None):
    state = {"request": _request("세탁실 몇 시까지 해요?"), "image_analysis": analysis}
    result = graph._run_knowledge(state)
    check("graph attaches image_analysis onto the knowledge result", result["result"].image_analysis is analysis)

    state_no_image = {"request": _request("세탁실 몇 시까지 해요?"), "image_analysis": None}
    result_no_image = graph._run_knowledge(state_no_image)
    check("graph leaves result untouched when there's no image", result_no_image["result"].image_analysis is None)

print("\nAll checks passed.")
