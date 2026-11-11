"""실제 모델로 Plan(요청→도구)과 민원 전환 판정을 반복 채점하는 회귀 게이트.

프롬프트 한 줄이 다른 케이스를 깨는지 손으로 돌려 보지 않도록, 통과율이 기준 아래면 1로 끝난다.

실행 (KTB4-19th-AI/ai 에서, LLM 설정이 있는 환경):

    uv run python scripts/check_turn_routing.py --repeat 3 --min-pass 0.9
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import logging
import sys

from zipsai.complaint import node
from zipsai.contracts.converse import ConverseRequest
from zipsai.integrations.runpod import get_locator
from zipsai.orchestration.intent import classify_intent

C, B, W = "process_complaint_turn", "search_building_docs", "search_public_web"

_LEAK_HISTORY = [
    {"message_id": "h1", "role": "user", "text": "천장에서 물이 새요"},
    {"message_id": "h2", "role": "assistant", "text": "어디에서 생긴 문제인가요?"},
]
_LEAK_DRAFT = {"issue_type": "plumbing", "symptom": "천장에서 물이 새요", "attachment_ids": []}

# (현재 발화, 기대 도구, 위치를 묻던 민원 대화 중인지)
PLAN_CASES = [
    ("그런데 헬스장은 몇 시에 열어요?", {B}, True),
    ("주차 등록은 어떻게 해요?", {B}, True),
    ("관리비 납부일이 언제예요?", {B}, True),
    ("분리수거 요일이 언제예요?", {B}, True),
    ("거실 천장이요", {C}, True),
    ("욕실이요", {C}, True),
    ("어제부터요", {C}, True),
    ("거실이에요. 그리고 헬스장은 몇 시에 열어요?", {C, B}, True),
    ("고마워요", set(), True),
    ("근데 민원 접수하면 어디서 확인해요?", set(), True),
    ("화장실 변기가 막혔어요", {C}, False),
    ("헬스장은 몇 시에 열어요?", {B}, False),
    ("음식물 쓰레기는 어디에 버려요?", {B}, False),
    ("이사 들어올 때 관리사무소에 뭐 신고해야 해요?", {B}, False),
    ("전입신고는 어떻게 해요?", {W}, False),
    ("오늘 서울 날씨 어때요?", {W}, False),
    ("페트병 라벨은 떼고 버려야 해요?", {W}, False),
    ("안녕하세요", set(), False),
    ("야 저녁 메뉴 추천해줘", set(), False),
    ("여기서 민원은 어떻게 넣어요?", set(), False),
    ("윗집 발소리 때문에 잠을 못 자요. 층간소음 규정 있어요?", {C, B}, False),
    ("보일러 온수가 안 나와요", {C}, False),
]

_WINDOW_HISTORY = [
    {"message_id": "h1", "role": "user", "text": "창문이 잘 안 닫혀요"},
    {"message_id": "h2", "role": "assistant", "text": "어디에서 생긴 문제인가요?"},
]
_WINDOW_DRAFT = {"issue_type": "interior_repair", "symptom": "창문이 잘 안 닫혀요", "attachment_ids": []}
_SWITCH_QUESTION = [
    *_WINDOW_HISTORY,
    {"message_id": "h3", "role": "user", "text": "현관 도어락도 고장났어요"},
    {
        "message_id": "h4",
        "role": "assistant",
        "text": "창문이 잘 안 닫혀요 건은 접어두고 「현관 도어락이 고장났어요」으로 전환할까요?",
    },
]

# (현재 발화, 기대 전환 판정, 대화)
SWITCH_CASES = [
    ("아 그리고 현관 도어락도 고장났어요", "ask", _WINDOW_HISTORY),
    ("현관 도어락이 고장났어요", "ask", _WINDOW_HISTORY),
    ("아니 근데 현관 도어락도 고장났어요", "ask", _WINDOW_HISTORY),
    ("아니 그리고 화장실 변기도 막혔어요", "ask", _WINDOW_HISTORY),
    ("아 창문 말고 방충망이 안 닫혀요", "same", _WINDOW_HISTORY),
    ("아니 그게 아니라 창틀에서 물이 새요", "same", _WINDOW_HISTORY),
    ("창문 손잡이가 헐거워서 그런 것 같아요", "same", _WINDOW_HISTORY),
    ("바람이 세게 불면 더 안 닫혀요", "same", _WINDOW_HISTORY),
    ("안방 창문이요", "same", _WINDOW_HISTORY),
    ("네 바꿔주세요", "accept", _SWITCH_QUESTION),
]


def _request(text: str, history: list[dict], draft: dict | None) -> ConverseRequest:
    return ConverseRequest.model_validate(
        {
            "building_id": 101,
            "room_no": "301",
            "resident_id": "probe",
            "conversation_id": "probe",
            "turn_id": "probe",
            "trace_id": "probe",
            "current_route": "complaint" if draft else None,
            "current_complaint_state": "collecting" if draft else None,
            "message": {"message_id": "m", "text": text},
            "conversation_history": history,
            "complaint_draft": draft,
        }
    )


def _plan(text: str, in_complaint: bool) -> set[str]:
    request = _request(
        text,
        _LEAK_HISTORY if in_complaint else [],
        _LEAK_DRAFT if in_complaint else None,
    )
    return set(classify_intent({"request": request})["planned_tools"])


def _switch(text: str, history: list[dict]) -> str:
    request = _request(text, history, _WINDOW_DRAFT)
    interpretation = node._interpret_turn(request, contextualized_message=text)
    return node._settle_switch(interpretation, request.complaint_draft).switch.value


def _score(name: str, runs: list[tuple[str, object, object]]) -> float:
    fails = collections.Counter(f"{text} -> {got}" for text, want, got in runs if got != want)
    rate = 1 - sum(fails.values()) / len(runs)
    print(f"{name}: {len(runs) - sum(fails.values())}/{len(runs)} ({rate:.0%})")
    for line, count in fails.most_common():
        print(f"  x{count} {line}")
    return rate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--min-pass", type=float, default=0.9)
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)
    if locator := get_locator():
        asyncio.run(locator.refresh())

    plan = [
        (text, want, _plan(text, in_complaint))
        for text, want, in_complaint in PLAN_CASES
        for _ in range(args.repeat)
    ]
    switch = [
        (text, want, _switch(text, history))
        for text, want, history in SWITCH_CASES
        for _ in range(args.repeat)
    ]
    rates = [_score("plan", plan), _score("switch", switch)]
    return 0 if min(rates) >= args.min_pass else 1


if __name__ == "__main__":
    sys.exit(main())
