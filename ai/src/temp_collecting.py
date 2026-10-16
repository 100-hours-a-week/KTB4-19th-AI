"""
민원 수집 대화가 실제로 끝나는지 여러 턴으로 확인하는 스모크 테스트.

실행: cd ai && uv run python ../temp_collecting.py

백엔드가 하는 일(민원 초안·진행 상태·대화 이력을 다음 요청에 실어 보내기)을 이
스크립트가 대신한다. 유닛 테스트는 초안을 직접 주입하므로 "턴을 넘겨 누적되는" 경로가
검증되지 않고, e2e_happy.sh는 매 요청을 complaint_draft=null로 보낸다.

확인하는 것:
  1. 위치 질문에 "몰라"라고 답하면 location이 "모름"으로 채워지고 대화가 끝난다
  2. 한 턴에 한 항목만 묻는다 (위치와 증상을 한 문장으로 같이 묻지 않는다)
  3. 정상 대화는 여전히 접수까지 간다 (회귀 확인)
  4. 증상을 모른다고 해도 "모름"으로 채우지 않고 계속 묻는다

실제 모델을 호출하므로 결과가 매번 조금씩 다를 수 있다. 판정이 ⚠️ 로 뜨면 프롬프트를
고칠 대상이지, 스크립트를 고칠 대상이 아니다.
"""

from fastapi.testclient import TestClient

from zipsai.main import app

CLIENT = TestClient(app, raise_server_exceptions=False)


class Conversation:
    """백엔드 역할 대행 — 응답에서 받은 상태를 다음 요청에 그대로 되돌려 보낸다."""

    def __init__(self, conversation_id: str) -> None:
        self.conversation_id = conversation_id
        self.route: str | None = None
        self.complaint_state: str | None = None
        self.draft: dict | None = None
        self.history: list[dict] = []
        self.turn = 0

    def say(self, text: str) -> dict[str, object]:
        self.turn += 1
        message_id = f"msg-{self.turn}"
        response = CLIENT.post(
            "/api/v3/ai/converse",
            json={
                "building_id": 1,
                "room_no": "301",
                "resident_id": "linda",
                "conversation_id": self.conversation_id,
                "trace_id": f"{self.conversation_id}-{self.turn}",
                "current_route": self.route,
                "current_complaint_state": self.complaint_state,
                "message": {
                    "message_id": message_id,
                    "text": text,
                    "image_urls": [],
                },
                "conversation_history": self.history,
                "complaint_draft": self.draft,
            },
        )
        print(f"  나      {text}")
        if response.status_code != 200:
            print(f"  ⚠️  HTTP {response.status_code}: {response.text[:200]}")
            raise SystemExit(1)

        data = response.json()["data"]
        self.route = data["route"]
        self.complaint_state = data["next_complaint_state"]
        self.draft = data["result"]["complaint_draft"]
        self.history += [
            {"message_id": message_id, "role": "user", "text": text, "image_urls": []},
            {
                "message_id": f"{message_id}-a",
                "role": "assistant",
                "text": data["reply"],
                "image_urls": [],
            },
        ]

        location = (self.draft or {}).get("location")
        symptom = (self.draft or {}).get("symptom")
        print(f"  집사이  {data['reply']}")
        print(
            f"          route={self.route} state={self.complaint_state}"
            f" missing={data['result']['missing_fields']}"
            f" location={location!r} symptom={symptom!r}"
        )
        return data

    def ended(self) -> bool:
        return self.complaint_state is None

    def location(self) -> str | None:
        return (self.draft or {}).get("location")


def check(label: str, passed: bool) -> None:
    print(f"  {'✅' if passed else '⚠️ '} {label}")


def scenario(title: str) -> Conversation:
    print(f"\n{'=' * 72}\n[{title}]")
    return Conversation(f"smoke-{title.split('.')[0]}")


if __name__ == "__main__":
    # 1. 실제로 무한루프가 났던 대화. 증상만 있고 위치를 모른다고 답한다.
    talk = scenario("1. 위치를 모른다고 답하면 끝나는가")
    talk.say("보일러가 안 켜져요")
    talk.say("몰라")
    check("대화가 끝났다 (state=null)", talk.ended())
    check('location이 "모름"으로 채워졌다', talk.location() == "모름")

    # 2. 위치·증상이 둘 다 없는 첫 턴에서 한 항목만 묻는지.
    talk = scenario("2. 한 턴에 한 항목만 묻는가")
    first = talk.say("집에 문제가 생겼어요")
    reply = str(first["reply"])
    check("아직 수집 중이다", not talk.ended())
    check("증상을 같이 묻지 않는다", "증상" not in reply)

    # 3. 정상 접수 경로 회귀 확인.
    talk = scenario("3. 정상 대화는 접수까지 가는가")
    talk.say("화장실 천장에서 물이 떨어져요")
    check("한 턴에 접수까지 갔다", talk.ended())
    check('location이 "모름"이 아니다', talk.location() != "모름")

    # 4. 증상에는 "모름" 탈출구를 주지 않는다. 계속 물어야 한다.
    talk = scenario("4. 증상을 모른다고 하면 계속 묻는가")
    talk.say("주방이 좀 이상해요")
    talk.say("모르겠어요")
    check("아직 끝나지 않았다", not talk.ended())
    check(
        'symptom이 "모름"으로 채워지지 않았다',
        (talk.draft or {}).get("symptom") != "모름",
    )

    print(f"\n{'=' * 72}")
    print("⚠️ 가 하나라도 있으면 프롬프트를 고칠 대상이다. 배포 전에 다시 돌려볼 것.")
