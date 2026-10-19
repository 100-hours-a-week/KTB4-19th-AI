"""실제 LLM 긴급도 평가 예시. ai/에서 `uv run python temp_urgency_demo.py`로 실행."""

import json

from zipsai.complaint.urgency_llm import assess_urgency
from zipsai.contracts.converse import ComplaintDraft, ImageAnalysis, ImageObservation

CASES = [
    (
        "즉시 위험",
        ComplaintDraft(symptom="주방에서 가스 누출 냄새가 나고 지금도 계속 나요."),
        None,
    ),
    (
        "해결된 문제",
        ComplaintDraft(symptom="어제 천장에서 물이 샜지만 지금은 멈췄어요."),
        None,
    ),
    (
        "다수 세대 영향",
        ComplaintDraft(symptom="우리 동 여러 세대에서 지금 단수가 계속되고 있어요."),
        None,
    ),
    (
        "사진 근거",
        ComplaintDraft(
            symptom="분전반에서 검은 연기가 계속 나고 있어요.", attachment_ids=[17]
        ),
        ImageAnalysis(
            images=[
                ImageObservation(
                    attachmentId=17,
                    summary="분전반 틈에서 검은 연기가 올라오는 모습이 보임.",
                )
            ]
        ),
    ),
    ("정보 부족", ComplaintDraft(symptom="불편해요."), None),
]


def main() -> None:
    for label, draft, image_analysis in CASES:
        result = assess_urgency(draft, image_analysis)
        print(f"\n[{label}] {draft.symptom}")
        print(f"점수={result.score}, 긴급={result.is_urgent}, 버전={result.urgency_version}")
        print(json.dumps(result.dimensions, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
