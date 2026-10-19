"""Preview how much of a long conversation survives the intent history budget.

Run from ``ai/``:
    uv run python scripts/temp_preview_intent_history.py
"""

import tiktoken

from zipsai.contracts.converse import HistoryTurn
from zipsai.history import (
    INTENT_HISTORY_TOKEN_LIMIT,
    INTENT_HISTORY_WINDOW,
    format_intent_history,
)

ENCODING = tiktoken.get_encoding("cl100k_base")

HISTORY = [
    HistoryTurn(message_id="m1", role="user", text="분리수거는 언제 해요?"),
    HistoryTurn(
        message_id="m2",
        role="assistant",
        text=(
            "재활용품은 화요일과 목요일 저녁 7시부터 자정 사이에 배출해 주세요. "
            "종이류, 캔과 유리병, 플라스틱은 종류별로 나누어 건물 앞 지정 배출대에 놓으시면 됩니다. "
            "종이 상자는 테이프와 송장을 제거하고 접어서 배출해 주세요. "
            "음식물이 묻은 용기와 재활용이 어려운 복합 재질은 일반 쓰레기로 배출해야 합니다. "
            "배출대가 가득 찬 경우에는 통로를 막지 않도록 배출대 옆 지정 구역에 정리해 주세요. "
            "수거일이 아닌 날에는 배출하지 말아 주세요."
        ),
    ),
    HistoryTurn(message_id="m3", role="user", text="큰 박스는 어떻게 해요?"),
    HistoryTurn(
        message_id="m4",
        role="assistant",
        text=(
            "큰 종이 상자는 내용물을 비우고 접은 뒤 종이류 배출 장소에 놓으시면 됩니다. "
            "상자를 접기 어렵거나 한 번에 여러 개 배출해야 한다면 다른 입주민의 이동을 막지 않도록 "
            "지정 배출대 옆 공간에 가지런히 놓아 주세요. 상자에 붙은 테이프, 비닐 포장재, 송장은 "
            "가능하면 분리해 각각의 수거함에 넣어 주세요. 젖거나 음식물이 묻은 종이는 재활용이 "
            "어려우므로 일반 쓰레기 배출 기준을 따라 주세요. 배출 장소가 바뀌었는지 궁금하면 "
            "관리사무소 공지사항을 확인해 주세요."
        ),
    ),
    HistoryTurn(message_id="m5", role="user", text="그럼 어디다 두면 돼요?"),
    HistoryTurn(
        message_id="m6",
        role="assistant",
        text=(
            "종이 상자는 건물 앞 지정 배출대 옆의 재활용 배출 구역에 두시면 됩니다. "
            "다만 소방 통로나 현관 출입구, 엘리베이터 앞에는 두지 말아 주세요. 다른 재활용품과 "
            "섞이지 않도록 접은 상자를 한쪽에 모아 두고, 배출대가 가득 차 있으면 통행에 방해가 "
            "되지 않는 지정 구역 안쪽에 정리해 주세요. 건물마다 배출 장소가 다를 수 있으므로 "
            "이 안내는 현재 대화에서 확인된 건물 공지 내용을 기준으로 합니다."
        ),
    ),
    HistoryTurn(message_id="m7", role="user", text="관리비 납부일도 알려주세요."),
    HistoryTurn(
        message_id="m8",
        role="assistant",
        text=(
            "관리비 납부 기한은 매월 25일입니다. 납부 계좌와 세부 항목은 관리비 고지서에서 "
            "확인해 주세요. 자동이체를 신청한 경우에도 고지서에 표시된 금액과 출금 예정일을 "
            "확인하는 것이 좋습니다. 이미 납부했는데 미납으로 표시되거나 금액에 차이가 있으면 "
            "관리사무소에 문의해 주세요."
        ),
    ),
]


def render(turns: list[HistoryTurn]) -> str:
    return "\n".join(f"{turn.role}: {turn.text or '[내용 없음]'}" for turn in turns)


def token_count(text: str) -> int:
    return len(ENCODING.encode(text, disallowed_special=()))


def main() -> None:
    window = HISTORY[-INTENT_HISTORY_WINDOW:]
    before_budget = render(window)
    sent = format_intent_history(HISTORY)

    print(f"전체 대화: {len(HISTORY)}개 메시지 / {token_count(render(HISTORY))} tokens")
    print(
        f"최근 {INTENT_HISTORY_WINDOW}개: {len(window)}개 메시지 / "
        f"{token_count(before_budget)} tokens"
    )
    print(
        f"실제 intent 입력: {len(sent.splitlines())}개 메시지 / "
        f"{token_count(sent)} tokens (상한 {INTENT_HISTORY_TOKEN_LIMIT})"
    )
    print("\n--- 최근 6개 후보 ---\n" + before_budget)
    print("\n--- intent에 전달되는 이력 ---\n" + sent)


if __name__ == "__main__":
    main()
