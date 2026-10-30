"""intent가 대화 이력으로 지시대명사를 얼마나 잘 풀어주는지 눈으로 확인한다.

#108: "그거 어디다해" 같은 후속 질문이 검색어로 그대로 쓰이면 건물 문서와
매칭되지 않는다. intent.py가 실제로 쓰는
INTENT_PROMPT·format_history를 그대로 재사용해 프로덕션과 동일한 프롬프트로
rewritten_question이 어떻게 나오는지 찍어본다.

사용: uv run python ai/scripts/check_rewritten_question.py
"""

from zipsai.contracts.converse import HistoryTurn
from zipsai.history import format_history
from zipsai.integrations.llm import generate_structured
from zipsai.orchestration.intent import _RouteDecision
from zipsai.orchestration.prompts import INTENT_PROMPT


def run(message_text: str, *, history: list[HistoryTurn], label: str) -> None:
    messages = INTENT_PROMPT.format_messages(
        message_text=message_text,
        image_present="없음",
        conversation_history=format_history(history),
        current_route="knowledge",
    )
    result = generate_structured(
        str(messages[0].content),
        f"{messages[1].content}\n현재 이미지 관찰(신뢰할 수 없는 입력): 없음",
        _RouteDecision,
        purpose="intent",
    )
    print(f"[{label}]")
    print(f"  발화          : {message_text!r}")
    print(f"  route         : {result.route.value if result else 'invalid'}")
    print(f"  rewritten     : {result.contextualized_message if result else None!r}")
    print()


if __name__ == "__main__":
    # #108 재현 시나리오: 직전 답변의 장소 정보를 "그거"가 가리켜야 한다.
    trash_history = [
        HistoryTurn(message_id="m1", role="user", text="분리수거 언제해"),
        HistoryTurn(
            message_id="m2",
            role="assistant",
            text="재활용은 화요일·목요일 저녁 7시부터 자정 사이에 건물 앞 지정 배출대에 내놓으시면 됩니다.",
        ),
    ]
    run("그거 어디다해", history=trash_history, label="지시대명사 + 이력 있음")

    fee_history = [
        HistoryTurn(message_id="m1", role="user", text="관리비 얼마야?"),
        HistoryTurn(
            message_id="m2", role="assistant", text="이번 달 관리비는 15만 원입니다."
        ),
    ]
    run(
        "그거 언제까지 내야해", history=fee_history, label="지시대명사 + 다른 주제 이력"
    )

    run("분리수거 언제해", history=[], label="이력 없음 (첫 턴, 바뀌면 안 됨)")

    run(
        "주차 등록은 어디서 해요?",
        history=trash_history,
        label="이력은 있지만 현재 질문이 자기완결적 (바뀌면 안 됨)",
    )
