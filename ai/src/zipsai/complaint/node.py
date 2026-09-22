from zipsai.contracts.converse import ComplaintState, ConverseRequest, RouteResult


def handle_complaint(request: ConverseRequest) -> dict[str, object]:
    draft = request.complaint_draft
    missing_fields = [
        field
        for field in ("location", "symptom")
        if draft is None or not getattr(draft, field)
    ]
    reply = "민원 접수를 위해 발생 위치와 불편 증상을 알려주세요."
    if not missing_fields:
        reply = "민원 정보를 확인했습니다. 접수할 내용을 확인해 주세요."

    return {
        "complaint_state": ComplaintState.COLLECTING,
        "reply": reply,
        "result": RouteResult(missing_fields=missing_fields),
    }
