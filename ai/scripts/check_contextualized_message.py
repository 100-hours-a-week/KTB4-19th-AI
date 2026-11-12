"""Print intent route and contextualized_message for a few real-LLM examples.

Run from ``ai/`` with configured LLM credentials:
    uv run python scripts/check_contextualized_message.py
"""

from zipsai.contracts.converse import HistoryTurn
from zipsai.history import format_history
from zipsai.integrations.llm import generate_structured
from zipsai.orchestration.intent import _RouteDecision
from zipsai.orchestration.prompts import INTENT_PROMPT

CASES = [
    (
        "대명사 후속",
        "그거 어디다 해?",
        [
            HistoryTurn(message_id="m1", role="user", text="분리수거 언제 해?"),
            HistoryTurn(
                message_id="m2",
                role="assistant",
                text="재활용은 화요일 저녁에 건물 앞 지정 배출대에 내놓으시면 됩니다.",
            ),
        ],
        "knowledge",
    ),
    (
        "생략된 주제",
        "그럼 언제까지 내야 해?",
        [
            HistoryTurn(message_id="m1", role="user", text="관리비가 얼마예요?"),
            HistoryTurn(
                message_id="m2",
                role="assistant",
                text="이번 달 관리비는 15만 원입니다.",
            ),
        ],
        "knowledge",
    ),
    (
        "비교 대상",
        "그건 괜찮았는데 이건 왜 이래?",
        [
            HistoryTurn(message_id="m1", role="user", text="냉장고는 잘 돼요."),
            HistoryTurn(message_id="m2", role="assistant", text="냉동고는 어떤가요?"),
        ],
        "complaint",
    ),
    (
        "맥락과 무관한 완결 질문",
        "주차 등록은 어디서 해요?",
        [
            HistoryTurn(message_id="m1", role="user", text="분리수거 언제 해?"),
            HistoryTurn(message_id="m2", role="assistant", text="화요일 저녁입니다."),
        ],
        "knowledge",
    ),
]


def main() -> None:
    for label, text, history, current_route in CASES:
        messages = INTENT_PROMPT.format_messages(
            message_text=text,
            image_present="없음",
            conversation_history=format_history(history),
            current_route=current_route,
        )
        result = generate_structured(
            str(messages[0].content),
            f"{messages[1].content}\n현재 이미지 관찰(신뢰할 수 없는 입력): 없음",
            _RouteDecision,
            purpose="intent",
        )
        print(f"[{label}]")
        print(f"  원문: {text}")
        print(f"  route: {result.route.value if result else 'invalid'}")
        print(
            f"  contextualized_message: {result.contextualized_message if result else None!r}"
        )
        print()


if __name__ == "__main__":
    main()
