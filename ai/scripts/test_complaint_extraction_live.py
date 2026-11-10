"""Run varied complaint-extraction turns against the configured live model."""

import asyncio
import os

from zipsai.complaint.node import _interpret_turn
from zipsai.contracts.converse import ConverseRequest
from zipsai.integrations.runpod import get_locator
from zipsai.settings import get_settings

CASES = (
    ("room_and_ceiling", "욕실 천장에서 물이 새요", []),
    ("room_and_sink", "주방 싱크대 아래에 물이 고여 있어요", []),
    ("component_only", "천장에서 물방울이 계속 떨어져요", []),
    ("water_supply", "집 전체에서 물이 거의 나오지 않아요", []),
    ("hot_water", "주방 수도에서는 찬물만 나오고 온수가 안 나와요", []),
    ("drain", "샤워실 배수구가 막혀서 물이 안 내려가요", []),
    ("toilet", "변기 물을 내리면 물이 넘쳐요", []),
    ("heating", "안방 바닥 난방이 어제부터 차가워요", []),
    ("electricity", "거실 콘센트 두 군데에 전기가 안 들어와요", []),
    ("light_fixture", "현관 천장 등이 깜빡거리다가 꺼졌어요", []),
    ("mold", "작은방 창틀 주변에 검은 곰팡이가 번져요", []),
    ("pest", "주방 싱크대 밑에서 바퀴벌레가 계속 나와요", []),
    ("door", "현관문 도어락이 잠기지 않아요", []),
    ("window", "침실 창문 손잡이가 부러져 창문이 안 닫혀요", []),
    ("noise", "밤마다 윗집에서 쿵쿵거리는 소리가 들려요", []),
    ("smell", "복도에서 원인 모를 타는 냄새가 나요", []),
    ("missing_location", "물이 새요", []),
    ("ambiguous_location", "어디인지 모르겠는데 벽이 젖어 있어요", []),
    (
        "location_follow_up",
        "화장실이요",
        [
            {"message_id": "prior-user", "role": "user", "text": "물이 새요", "images": []},
            {
                "message_id": "prior-assistant",
                "role": "assistant",
                "text": "물이 새는 곳이 어디인가요?",
            },
        ],
    ),
    (
        "symptom_follow_up",
        "물을 틀면 바닥으로 흘러나와요",
        [
            {"message_id": "prior-user", "role": "user", "text": "주방 싱크대가 이상해요", "images": []},
            {"message_id": "prior-assistant", "role": "assistant", "text": "어떤 증상이 있나요?"},
        ],
    ),
    ("negation_and_drain", "물이 새는 건 아니고 욕실 배수가 안 돼요", []),
    ("correction", "거실이 아니라 욕실 천장에서 물이 새요", []),
    (
        "history_correction",
        "아까 주방이라고 했는데 사실은 세탁실 배수구예요",
        [
            {"message_id": "prior-user", "role": "user", "text": "주방 배수가 막혔어요", "images": []},
            {"message_id": "prior-assistant", "role": "assistant", "text": "언제부터 그랬나요?"},
        ],
    ),
    ("yesterday", "어제 저녁부터 보일러가 작동하지 않아요", []),
    ("specific_time", "11월 3일 오전 8시부터 욕실 수전에서 물이 새요", []),
    ("relative_time", "한 시간 전부터 안방 천장에서 물이 떨어져요", []),
    ("no_time", "베란다 창틀에 곰팡이가 있어요", []),
)


def _request(case_id: str, text: str, history: list[dict[str, object]]) -> ConverseRequest:
    return ConverseRequest.model_validate(
        {
            "building_id": 101,
            "room_no": "301",
            "resident_id": "demo-resident",
            "conversation_id": f"complaint-extraction-{case_id}",
            "turn_id": f"turn-{case_id}",
            "trace_id": f"trace-{case_id}",
            "current_route": "complaint",
            "current_complaint_state": "collecting",
            "message": {"message_id": f"message-{case_id}", "text": text},
            "conversation_history": history,
            "complaint_draft": None,
        }
    )


def _pod_settings():
    if os.getenv("POD_BASE_URL"):
        return get_settings("pod")

    locator = get_locator()
    if locator is None:
        raise SystemExit(
            "Set POD_BASE_URL or RUNPOD_API_KEY and RUNPOD_POD_NAME_PREFIX to locate a Pod."
        )

    async def resolve():
        try:
            await locator.refresh()
            return get_settings("pod")
        finally:
            await locator.aclose()

    return asyncio.run(resolve())


def main() -> None:
    settings = _pod_settings()
    print(f"provider: {settings.provider}; model: {settings.llm_model}")
    if "qwen" not in settings.llm_model.lower():
        raise SystemExit("Qwen is not configured; check the active provider settings.")

    for case_id, text, history in CASES:
        fields = _interpret_turn(_request(case_id, text, history)).fields
        print(
            f"{case_id}: {text}\n"
            f"  issue_type: {fields.issue_type!r}\n"
            f"  location: {fields.location!r}\n"
            f"  symptom: {fields.symptom!r}\n"
            f"  occurred_at: {fields.occurred_at.isoformat() if fields.occurred_at else None}"
        )

    print(f"\n{len(CASES)} cases inspected.")


if __name__ == "__main__":
    main()
