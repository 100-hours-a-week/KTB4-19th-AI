"""민원 위치 추출을 실제 모델로 채점한다.

장소를 말한 문장은 그 장소를 뽑아야 하고(recall), 부품만 말한 문장은 장소를 지어내지
않아야 한다(null 유지). 실행: cd ai && PYTHONPATH=src uv run python scripts/eval_complaint_location_live.py [반복 수]
"""

import sys

from test_complaint_extraction_live import CASES, _pod_settings, _request

from zipsai.complaint.node import _interpret_turn

# 정답: 장소 문자열(부분 일치) 또는 None. door·noise·symptom_follow_up은 정답이 갈려 뺀다.
GOLD = {
    "room_and_ceiling": "욕실",
    "room_and_sink": "주방",
    "hot_water": "주방",
    "drain": "샤워실",
    "heating": "안방",
    "electricity": "거실",
    "light_fixture": "현관",
    "mold": "작은방",
    "pest": "주방",
    "window": "침실",
    "smell": "복도",
    "location_follow_up": "화장실",
    "negation_and_drain": "욕실",
    "correction": "욕실",
    "history_correction": "세탁실",
    "specific_time": "욕실",
    "relative_time": "안방",
    "no_time": "베란다",
    "water_supply": "집",
    "ambiguous_location": "모름",
    "component_only": None,
    "toilet": None,
    "missing_location": None,
    "yesterday": None,
}

# 부품·설비만 말하고 장소는 말하지 않은 문장. 위치를 추론해 채우면 실패다.
COMPONENT_ONLY = (
    "천장에서 물이 새요",
    "수도꼭지에서 물이 계속 떨어져요",
    "변기 물이 안 내려가요",
    "보일러가 안 켜져요",
    "창문이 안 닫혀요",
    "콘센트에 전기가 안 들어와요",
    "벽지에 곰팡이가 폈어요",
    "싱크대 배수구가 막혔어요",
    "에어컨에서 물이 떨어져요",
    "형광등이 깜빡거려요",
    "인터폰이 안 돼요",
    "샤워기 수압이 너무 약해요",
    "배수구에서 하수구 냄새가 올라와요",
    "환풍기가 안 돌아가요",
    "세면대 아래 배관에서 물이 새요",
    "온수가 안 나와요",
)


def _cases():
    for case_id, text, history in CASES:
        if case_id in GOLD:
            yield case_id, text, history, GOLD[case_id]
    for index, text in enumerate(COMPONENT_ONLY):
        yield f"component_{index:02d}", text, [], None


def _correct(location: str | None, gold: str | None) -> bool:
    if gold is None:
        return location is None
    return location is not None and gold in location


def main() -> None:
    repeats = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    settings = _pod_settings()
    print(
        f"provider: {settings.provider}; model: {settings.llm_model}; repeats: {repeats}"
    )

    place: list[bool] = []
    empty: list[bool] = []
    misses: dict[tuple[str, str], list[str | None]] = {}
    for _ in range(repeats):
        for case_id, text, history, gold in _cases():
            location = _interpret_turn(_request(case_id, text, history)).fields.location
            ok = _correct(location, gold)
            (place if gold is not None else empty).append(ok)
            if not ok:
                misses.setdefault((case_id, text), []).append(location)

    print(
        f"장소 있는 문장 recall: {sum(place)}/{len(place)} ({sum(place) / len(place):.0%})"
    )
    print(
        f"장소 없는 문장 null 유지: {sum(empty)}/{len(empty)} ({sum(empty) / len(empty):.0%})"
    )
    for (case_id, text), got in sorted(misses.items()):
        print(f"  miss {case_id}: {text} -> {got}")


if __name__ == "__main__":
    main()
