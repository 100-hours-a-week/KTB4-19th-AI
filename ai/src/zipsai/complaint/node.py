import json
from datetime import datetime
from zoneinfo import ZoneInfo

from pydantic import ValidationError

from zipsai.complaint.prompts import COMPLAINT_PROMPT, VLM_ANALYSIS_PROMPT
from zipsai.contracts.converse import (
    ComplaintDraft,
    ComplaintState,
    ConverseRequest,
    ImageAnalysis,
    RouteResult,
)
from zipsai.errors import (
    ComplaintExtractionError,
    ImageAnalysisError,
    LlmRateLimitedError,
    LlmTimeoutError,
    LlmUnavailableError,
    LlmUpstreamError,
)
from zipsai.history import format_history
from zipsai.integrations.llm import generate_text, strip_json_code_fence
from zipsai.integrations.vlm import analyze_images

_KST = ZoneInfo("Asia/Seoul")
# 일시적인 JSON/스키마 오류만 한 번 더 시도한다. rate limit·timeout 등 LLM 레벨 오류는
# generate_text가 별도 예외로 던지므로 여기서 재시도하지 않고 API 계층까지 그대로 올려보낸다.
_EXTRACTION_ATTEMPTS = 2

# 민원 카드를 만들기 위한 최소 항목. 묻는 순서도 이 순서다.
# 위치를 모른다는 답은 추출 프롬프트가 "모름"으로 채우므로 여기서 빠진다. 증상은
# "모름"을 허용하지 않아 계속 null로 남고, 그래서 계속 질문 대상이 된다.
_REQUIRED_FIELDS = ("location", "symptom")


def extract_complaint_fields(request: ConverseRequest) -> ComplaintDraft:
    draft, _reply, _missing = _extract_complaint_fields_and_reply(request)
    return draft


def _extract_complaint_fields_and_reply(
    request: ConverseRequest,
) -> tuple[ComplaintDraft, str, set[str]]:
    system_message, user_message = COMPLAINT_PROMPT.format_messages(
        today=datetime.now(_KST).date().isoformat(),
        conversation_history=format_history(request.conversation_history),
        complaint_draft=request.complaint_draft,
        message_text=request.message.text,
    )

    last_error: Exception | None = None
    for _attempt in range(_EXTRACTION_ATTEMPTS):
        raw = generate_text(system_message.content, user_message.content)
        try:
            data = json.loads(strip_json_code_fence(raw))
        except json.JSONDecodeError as error:
            last_error = error
            continue

        reply = data.pop("reply", "")
        if not isinstance(reply, str):
            reply = ""

        llm_missing = data.pop("missing", [])
        if not isinstance(llm_missing, list):
            llm_missing = []
        llm_missing = {field for field in llm_missing if field in _REQUIRED_FIELDS}

        try:
            draft = ComplaintDraft(**data)
        except ValidationError as error:
            last_error = error
            continue

        return draft, reply.strip(), llm_missing

    raise ComplaintExtractionError(
        "LLM returned an invalid complaint draft after retry"
    ) from last_error


def _merge_complaint_draft(
    current: ComplaintDraft | None, extracted: ComplaintDraft
) -> ComplaintDraft:
    updates = {
        field: getattr(extracted, field)
        for field in ("issue_type", "location", "symptom", "occurred_at")
        if getattr(extracted, field) is not None
    }
    return (current or ComplaintDraft()).model_copy(update=updates)


def _append_image_urls(draft: ComplaintDraft, image_urls: list[str]) -> ComplaintDraft:
    return draft.model_copy(
        update={"image_urls": list(dict.fromkeys([*draft.image_urls, *image_urls]))}
    )


def _missing_fields(draft: ComplaintDraft | None) -> list[str]:
    return [
        field
        for field in _REQUIRED_FIELDS
        if draft is None or not getattr(draft, field)
    ]


# 한 턴에 한 필드만 묻는다. 둘을 한 문장으로 같이 물으면 "몰라" 같은 답이 어느 필드에
# 대한 것인지 추출 프롬프트도 판단할 수 없다.
# LLM의 reply를 못 받았거나, LLM이 물은 필드가 우리가 물을 필드와 다를 때 쓰는 fallback.
_MISSING_FIELD_REPLY = {
    "location": "어디에서 생긴 문제인가요?",
    "symptom": "어떤 불편 증상인지 알려주세요.",
}
_PHOTO_ANALYZED_PREFIX = "사진은 확인했습니다. "
_PHOTO_FAILED_PREFIX = "사진을 받았지만 분석에 실패했어요. "


def _photo_prefix(image_analysis: ImageAnalysis | None) -> str:
    """사진에서 읽은 내용을 응답에 그대로 노출한다.

    VLM 결과를 초안에 직접 넣으면 신뢰도 임계값이 필요해진다(확인 단계 없이 카드가
    만들어지므로). 대신 읽은 내용을 보여주고 입주민이 맞다고 하면 그 발화로 symptom이
    채워지게 한다 — 확인 주체가 사람이라 임계값이 필요 없다.
    """
    if image_analysis is None:
        return _PHOTO_FAILED_PREFIX
    summary = next(
        (observed.summary for observed in image_analysis.images if observed.summary),
        None,
    )
    if not summary:
        return _PHOTO_ANALYZED_PREFIX
    return f"사진은 확인했습니다 — {summary.rstrip('. ')}. "


def handle_complaint(request: ConverseRequest) -> dict[str, object]:
    extracted, llm_reply, llm_missing = _extract_complaint_fields_and_reply(request)
    draft = _append_image_urls(
        _merge_complaint_draft(request.complaint_draft, extracted),
        request.message.image_urls,
    )
    photo_sent = bool(request.message.image_urls)
    try:
        image_analysis = (
            analyze_images(request.message.image_urls, VLM_ANALYSIS_PROMPT)
            if photo_sent
            else None
        )
    except (
        ImageAnalysisError,
        LlmRateLimitedError,
        LlmTimeoutError,
        LlmUnavailableError,
        LlmUpstreamError,
    ):
        image_analysis = None
    missing_fields = _missing_fields(draft)

    if missing_fields:
        complaint_state = ComplaintState.COLLECTING
        asked_now = missing_fields[0]
        if llm_reply and llm_missing == {asked_now}:
            follow_up = llm_reply
        else:
            follow_up = _MISSING_FIELD_REPLY[asked_now]
        reply = (
            follow_up if not photo_sent else _photo_prefix(image_analysis) + follow_up
        )
    else:
        if draft.issue_type is None:
            draft = draft.model_copy(update={"issue_type": "other"})
        # 필수 필드가 찼으면 상태를 비운다. 백엔드가 missing_fields가 빈 것을 보고
        # 민원 카드를 만들고 대화를 끝내므로, 확인 대기 상태를 따로 둘 필요가 없다.
        complaint_state = None
        reply = "민원 정보를 확인했습니다. 접수할 내용을 확인해 주세요."

    return {
        "complaint_state": complaint_state,
        "reply": reply,
        "result": RouteResult(
            complaint_draft=draft,
            missing_fields=missing_fields,
            image_analysis=image_analysis,
        ),
    }
