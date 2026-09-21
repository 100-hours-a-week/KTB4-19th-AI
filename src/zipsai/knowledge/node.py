from zipsai.contracts.converse import ConverseRequest, RouteResult


def handle_knowledge(_request: ConverseRequest) -> dict[str, object]:
    return {
        "complaint_state": None,
        "reply": "현재 건물 문서 검색 기능을 준비 중입니다.",
        "result": RouteResult(),
    }
