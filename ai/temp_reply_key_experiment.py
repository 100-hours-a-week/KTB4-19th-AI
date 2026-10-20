"""추출 프롬프트의 reply/missing 키가 지연에 얼마나 기여하는지 재는 실험.

가설: reply 키가 "한국어 문장 하나를 작문하라"고 시키므로 출력 토큰과
reasoning이 늘어난다. missing은 작문이 아니지만 같이 뺀다 — 코드가
_missing_fields()로 이미 계산하므로 LLM 출력에 있을 필요가 없다.

실행: cd ai && uv run python temp_reply_key_experiment.py

같은 입력 N개를 두 프롬프트(A=현재 7키, B=5키)로 각각 돌려 text_extraction
소요시간만 비교한다. 실제 모델을 호출하므로 매번 값이 다르다 — 평균과 분산을
같이 본다.

결과가 유의미하게 줄면: reply/missing을 프롬프트에서 빼고 _follow_up의
llm 경로를 없앤다 (고정 문구만 남긴다).
결과가 안 줄면: 원인은 reasoning이지 이 두 키가 아니다. #58 Structured
Output으로 넘긴다.
"""

import json
import statistics
import time

from langchain_core.prompts import ChatPromptTemplate

from zipsai.complaint.prompts import COMPLAINT_USER_TEMPLATE
from zipsai.integrations.llm import generate_text, strip_json_code_fence

RUNS_PER_CASE = 3

CASES = [
    ("빈 초안 + 텍스트", None, "화장실 천장에서 물이 계속 떨어져요"),
    ("위치 물은 뒤 답변", {"issue_type": "leak", "symptom": "천장에서 물이 떨어져요"}, "화장실이요"),
    ("전환 상황", {"issue_type": "leak", "location": "화장실", "symptom": "천장에서 물이 떨어져요"}, "아 그런데 세탁기도 안 돌아가요"),
]

# 변형 A: 현재 프롬프트 (issue_type 규칙까지는 그대로 두고 reply/missing만 자른다)
_SHARED_HEAD = (
    "You extract facility-complaint details from a resident's CURRENT turn in a residential-building "
    "assistant. Use the existing draft and conversation history only to interpret a correction in the "
    "current message. Extract ONLY values explicitly stated or corrected in the current message; do not "
    "repeat unchanged values. Those are merged separately.\n\n"
    "issue_type taxonomy (pick exactly one, or null if the current turn doesn't clearly indicate one):\n"
    "- water_supply: water supply is missing or insufficient (no water, low pressure, no hot water).\n"
    "- drain: drainage is blocked or malfunctioning (clogged drain, backflow, sewage smell).\n"
    "- leak: water is escaping where it shouldn't (leaking from ceiling, wall, pipe, faucet).\n"
    "- heating: heating/hot-water-for-heating is not working or not controllable.\n"
    "- electricity: power issue (outage, broken outlet/light, tripped breaker, exposed wiring).\n"
    "- mold: mold or mildew growth.\n"
    "- pest: insects or pests (cockroaches, ants, rodents, etc.).\n"
    "- facility: a non-water/electrical fixture is broken (elevator, door, window, lock, intercom).\n"
    "- noise: noise complaint (from another unit, construction, equipment).\n"
    "- other: a real facility complaint that doesn't fit any category above.\n\n"
    "Other fields:\n"
    "- complaint_switch: whether this turn is still about the SAME complaint as the existing draft "
    'below. Output exactly one of "same", "ask", "accept". Default to "same" whenever you are not '
    "confident.\n"
    "- location: where the problem is. null if not stated in this turn.\n"
    "- symptom: what's wrong, in the resident's own words. null if not stated in this turn.\n"
    "- occurred_at: when the problem started, only if stated in this turn. ISO 8601 date. null "
    "if not stated.\n"
)

PROMPT_A_SYSTEM = (
    _SHARED_HEAD
    + '- missing: an array listing which of "location"/"symptom" are still unknown overall.\n'
    "- reply: a short, natural Korean follow-up question asking about EXACTLY ONE field — the "
    'first still-unknown field. Empty string "" if `missing` is empty.\n\n'
    "Output contract: output ONLY a JSON object with exactly these seven keys, nothing else.\n"
    '{{"complaint_switch": "same", "issue_type": null, "location": null, "symptom": null, '
    '"occurred_at": null, "missing": [], "reply": ""}}'
)

PROMPT_B_SYSTEM = (
    _SHARED_HEAD
    + "\nOutput contract: output ONLY a JSON object with exactly these five keys, nothing else. "
    "Do not add missing or reply — the service computes those itself.\n"
    '{{"complaint_switch": "same", "issue_type": null, "location": null, "symptom": null, '
    '"occurred_at": null}}'
)

PROMPT_A = ChatPromptTemplate.from_messages(
    [("system", PROMPT_A_SYSTEM), ("user", COMPLAINT_USER_TEMPLATE)]
)
PROMPT_B = ChatPromptTemplate.from_messages(
    [("system", PROMPT_B_SYSTEM), ("user", COMPLAINT_USER_TEMPLATE)]
)


def timed_call(prompt: ChatPromptTemplate, draft: dict | None, text: str) -> float:
    messages = prompt.format_messages(
        today="2026-09-30",
        conversation_history="없음",
        complaint_draft=draft,
        message_text=text,
    )
    started = time.perf_counter()
    raw = generate_text(str(messages[0].content), str(messages[1].content))
    elapsed = time.perf_counter() - started
    try:
        json.loads(strip_json_code_fence(raw))
        ok = True
    except json.JSONDecodeError:
        ok = False
    return elapsed, ok


def main() -> None:
    results: dict[str, list[float]] = {"A(7키, reply·missing 포함)": [], "B(5키)": []}
    failures: dict[str, int] = {"A(7키, reply·missing 포함)": 0, "B(5키)": 0}

    for label, draft, text in CASES:
        print(f"\n[{label}]  \"{text}\"")
        for prompt, key in [(PROMPT_A, "A(7키, reply·missing 포함)"), (PROMPT_B, "B(5키)")]:
            times = []
            for run in range(1, RUNS_PER_CASE + 1):
                elapsed, ok = timed_call(prompt, draft, text)
                times.append(elapsed)
                if not ok:
                    failures[key] += 1
                print(f"  {key}  run{run}  {elapsed:.2f}s  {'OK' if ok else '⚠ JSON 파싱 실패'}")
            results[key] += times

    print("\n" + "=" * 60)
    print("요약 (초, 낮을수록 좋음)")
    for key, times in results.items():
        print(
            f"  {key:28} 평균={statistics.mean(times):.2f}  "
            f"중앙값={statistics.median(times):.2f}  "
            f"최대={max(times):.2f}  실패={failures[key]}"
        )

    diff = statistics.mean(results["A(7키, reply·missing 포함)"]) - statistics.mean(
        results["B(5키)"]
    )
    print(f"\n평균 차이(A-B) = {diff:+.2f}s")
    if diff > 1.0:
        print("→ 유의미하게 줄었다. reply/missing을 빼고 고정 문구만 쓰는 걸 검토.")
    else:
        print("→ 차이가 작다. 원인은 reply/missing이 아니라 reasoning 자체 — #58로.")


if __name__ == "__main__":
    main()
