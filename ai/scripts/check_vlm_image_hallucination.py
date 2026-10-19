"""VLM이 이번 턴에 이미지가 없는데도 images 필드를 채워 돌려주는지 실제로 재현해본다.

가설: conversation_history에 이전 턴 사진 요약이 "[사진: ...]" 텍스트로 남아있으면,
이번 턴은 이미지 0장(image_present="없음")인데도 모델이 그 과거 사진을 다시
분석해서 images에 채워 넣는다. classify_and_analyze/intent.py가 실제로 쓰는
INTENT_PROMPT·format_history를 그대로 재사용해 프로덕션과 동일한 프롬프트로 찔러본다.

사용: uv run python ai/scripts/check_vlm_image_hallucination.py
"""

from zipsai.contracts.converse import HistoryTurn, ImageObservation, Route
from zipsai.history import format_history
from zipsai.integrations.vlm import classify_and_analyze
from zipsai.orchestration.prompts import INTENT_PROMPT

history = [
    HistoryTurn(
        message_id="m1",
        role="user",
        text="세탁기 사진 보낼게요",
        images=[
            ImageObservation(
                attachmentId=1,
                summary="세탁기 표시창에 오류 코드가 보임",
                ocrText="E1",
            )
        ],
    ),
    HistoryTurn(message_id="m2", role="assistant", text="확인했습니다, E1 오류네요."),
]


def run(message_text: str, *, with_history: bool) -> bool:
    messages = INTENT_PROMPT.format_messages(
        message_text=message_text,
        image_present="없음",
        conversation_history=format_history(history) if with_history else "없음",
        current_route="complaint",
    )
    route, analysis, _rewritten_question = classify_and_analyze(
        [],  # 이번 턴 이미지 0장
        system_prompt=str(messages[0].content),
        user_prompt=str(messages[1].content),
    )
    hit = bool(analysis.images)
    print(
        f"history={with_history!s:5} 발화={message_text!r:20} route={route.value:9} "
        f"images={[obs.model_dump() for obs in analysis.images]} "
        f"{'재현됨' if hit else ''}"
    )
    return hit


if __name__ == "__main__":
    texts = ["그거 언제 고쳐지나요", "감사합니다", "네 알겠습니다", "아직도 안 고쳐졌어요", "다른 문제가 있어요"]
    hits = 0
    total = 0
    for text in texts:
        for with_history in (True, False):
            total += 1
            hits += run(text, with_history=with_history)
    print(f"\n{hits}/{total} 회 재현됨")
