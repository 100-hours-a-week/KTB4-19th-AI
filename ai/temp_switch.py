"""민원 전환(same / ask / accept) 흐름을 실제 모델로 확인하는 스크립트.

실행:
    cd ai && uv run python temp_switch.py          시나리오 4개 자동 실행
    cd ai && uv run python temp_switch.py --talk   직접 입력해보기

백엔드가 하는 일을 이 스크립트가 대신한다 — 응답에서 받은 route·state·초안·이력을
다음 요청에 그대로 되돌려 보낸다. 유닛 테스트는 추출 결과를 주입하므로, 실제 모델이
same/ask/accept를 제대로 내는지는 여기서만 확인된다.

전환 판정은 응답에 없고 로그에만 있다. 그래서 complaint_turn 로그를 주워서 보여준다.

실제 모델을 호출하므로 결과가 매번 조금씩 다르다. ⚠ 가 뜨면 프롬프트를 고칠
대상이지 스크립트를 고칠 대상이 아니다.
"""

import logging
import sys

from fastapi.testclient import TestClient

from zipsai.main import app

CLIENT = TestClient(app, raise_server_exceptions=False)


class TurnLog(logging.Handler):
    """complaint_turn 로그에서 판정 결과만 주워 둔다."""

    def __init__(self) -> None:
        super().__init__()
        self.switch: str | None = None
        self.reply_source: str | None = None

    def emit(self, record: logging.LogRecord) -> None:
        if record.getMessage() == "complaint_turn":
            self.switch = getattr(record, "switch", None)
            self.reply_source = getattr(record, "reply_source", None)


TURN_LOG = TurnLog()
logging.getLogger("zipsai.complaint.node").addHandler(TURN_LOG)
logging.getLogger("zipsai.complaint.node").setLevel(logging.INFO)

# 운영용 JSON 로그가 화면을 덮어 대화가 안 보인다. 핸들러만 떼고 로거는 살려둔다
# (TURN_LOG는 node 로거에 직접 붙어 있어 그대로 동작한다).
logging.getLogger().handlers.clear()


class Conversation:
    """백엔드 역할 대행. 받은 상태를 다음 요청에 그대로 실어 보낸다."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.route: str | None = None
        self.state: str | None = None
        self.draft: dict | None = None
        self.history: list[dict] = []
        self.turn = 0
        self.ended = False

    def say(self, text: str) -> None:
        self.turn += 1
        TURN_LOG.switch = TURN_LOG.reply_source = None

        response = CLIENT.post(
            "/api/v3/ai/converse",
            json={
                "building_id": 1,
                "room_no": "301",
                "resident_id": "linda",
                "conversation_id": self.name,
                "turn_id": f"{self.name}-{self.turn}",
                "current_route": self.route,
                "current_complaint_state": self.state,
                "message": {
                    "message_id": f"msg-{self.turn}",
                    "text": text,
                    "image_urls": [],
                },
                "conversation_history": self.history,
                "complaint_draft": self.draft,
            },
        )
        print(f"\n  나   {text}")
        if response.status_code != 200:
            print(f"  ⚠ HTTP {response.status_code}  {response.text[:200]}")
            self.ended = True
            return

        data = response.json()["data"]
        print(f"  AI   {data['reply']}")
        self._show(data)
        self._carry(text, data)

    def _show(self, data: dict) -> None:
        result = data["result"]
        draft = result.get("complaint_draft") or {}
        missing = result["missing_fields"]
        card = data["route"] == "complaint" and missing == []

        print(f"       route={data['route']}  state={data['next_complaint_state']}")
        print(f"       판정={TURN_LOG.switch}  문구={TURN_LOG.reply_source}")
        print(
            "       용지  "
            f"종류={draft.get('issue_type')} "
            f"위치={draft.get('location')} "
            f"증상={draft.get('symptom')} "
            f"사진={len(draft.get('image_urls') or [])}"
        )
        print(f"       빈칸={missing}" + ("   ← 카드 생성, 대화 끝" if card else ""))

    def _carry(self, said: str, data: dict) -> None:
        """다음 턴에 실어 보낼 것들. 백엔드가 하는 일이다."""
        self.route = data["route"]
        self.state = data["next_complaint_state"]
        self.draft = data["result"].get("complaint_draft")
        self.history += [
            {
                "message_id": f"msg-{self.turn}",
                "role": "user",
                "text": said,
                "image_urls": [],
            },
            {
                "message_id": f"ai-{self.turn}",
                "role": "assistant",
                "text": data["reply"],
                "image_urls": [],
            },
        ]
        # 필수 항목이 다 차면 백엔드가 카드를 만들고 대화를 닫는다.
        self.ended = (
            data["route"] == "complaint" and data["result"]["missing_fields"] == []
        )


def scenario(title: str, lines: list[str], expect: str) -> None:
    print("\n" + "=" * 72)
    print(f"[{title}]")
    print(f"기대: {expect}")
    talk = Conversation(title)
    for line in lines:
        talk.say(line)
        if talk.ended:
            break


def main() -> None:
    if "--talk" in sys.argv:
        talk = Conversation("직접입력")
        print("빈 줄이나 Ctrl-D 로 종료.\n")
        while not talk.ended:
            try:
                text = input("나 > ").strip()
            except EOFError:
                break
            if not text:
                break
            talk.say(text)
        return

    only = next((a for a in sys.argv[1:] if a.isdigit()), None)
    if only:
        chosen = {
            "1": ("1. 같은 민원 이어가기", ["물이 계속 떨어져요", "화장실이요"]),
            "2": (
                "2. 새 민원으로 전환 — 수락",
                [
                    "물이 계속 떨어져요",
                    "아 그런데 세탁기도 안 돌아가요",
                    "네",
                    "주방이요",
                ],
            ),
            "3": (
                "3. 새 민원으로 전환 — 거절",
                [
                    "물이 계속 떨어져요",
                    "아 그런데 세탁기도 안 돌아가요",
                    "아니요 물 새는 거요",
                ],
            ),
            "4": ("4. 위치를 모른다고 답하기", ["이상한 소리가 나요", "몰라요"]),
        }[only]
        scenario(chosen[0], chosen[1], "위 제목 참고")
        return

    scenario(
        "1. 같은 민원 이어가기",
        ["물이 계속 떨어져요", "화장실이요"],
        "판정 same 두 번, 위치를 받는 턴에 카드",
    )
    scenario(
        "2. 새 민원으로 전환 — 수락",
        ["물이 계속 떨어져요", "아 그런데 세탁기도 안 돌아가요", "네", "주방이요"],
        "ask(전환 물어봄) → accept(용지 교체) → 위치 받고 카드",
    )
    scenario(
        "3. 새 민원으로 전환 — 거절",
        ["물이 계속 떨어져요", "아 그런데 세탁기도 안 돌아가요", "아니요 물 새는 거요"],
        "ask → same(옛 민원 유지). 증상이 천장 누수로 남아야 한다",
    )
    scenario(
        "4. 위치를 모른다고 답하기",
        ["이상한 소리가 나요", "몰라요"],
        "판정 same. 위치가 「모름」으로 채워지고 카드",
    )

    print("\n" + "=" * 72)
    print("확인 포인트")
    print("  · 2번에서 판정이 ask → accept 로 이어지는가")
    print(
        "  · 2번의 전환 문구에 「세탁기…」처럼 낫표가 들어가는가 (수락 턴의 회수 근거)"
    )
    print("  · 3번에서 거절 후 증상이 물 새는 것으로 남는가 (세탁기로 바뀌면 버그)")
    print("  · 빈칸=[] 는 카드가 나가는 턴에만 찍히는가")


if __name__ == "__main__":
    main()
