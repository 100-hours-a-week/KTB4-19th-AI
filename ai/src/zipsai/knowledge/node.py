import logging

from zipsai.contracts.converse import ConverseRequest, RouteResult
from zipsai.integrations.llm import generate_text
from zipsai.integrations.qdrant import get_client
from zipsai.knowledge.prompts import KNOWLEDGE_PROMPT, NO_EVIDENCE, format_context
from zipsai.knowledge.retrieve import encode_question, query_encoder, search_chunks

logger = logging.getLogger(__name__)

# 위키 [AI] 모델 API 설계의 질의 응답 예시에 고정된 문구다. 백엔드가 이 응답을
# QA 카드로 저장하므로 문구를 바꾸면 위키부터 고쳐야 한다.
NO_EVIDENCE_REPLY = "건물 문서에서 근거를 찾지 못해 답변드리기 어렵습니다. 질문을 관리자에게 전달해 두었습니다."

# 관리자에게 전달할 질문 자체가 없으므로 위 문구를 쓰면 빈 QA 카드가 만들어진다.
EMPTY_QUESTION_REPLY = (
    "질문 내용을 입력해 주시면 건물 문서에서 찾아 안내해 드리겠습니다."
)


def handle_knowledge(request: ConverseRequest) -> dict[str, object]:
    question = (request.message.text or "").strip()
    if not question:
        # 이미지만 온 요청은 intent가 complaint로 보내지만, current_route=knowledge로
        # 유지되는 경로가 있어 검색까지 가기 전에 막는다.
        return _fallback(request, reason="empty_question", reply=EMPTY_QUESTION_REPLY)

    chunks = search_chunks(
        encode_question(question, encoder=query_encoder()),
        request.building_id,
        client=get_client(),
    )
    if not chunks:
        return _fallback(request, reason="no_hit", reply=NO_EVIDENCE_REPLY)

    messages = KNOWLEDGE_PROMPT.format_messages(
        question=question,
        context=format_context(chunks),
    )
    answer = generate_text(
        system_prompt=str(messages[0].content),
        user_prompt=str(messages[1].content),
    )
    # 전체 일치로만 판정한다. 부분 문자열로 보면 건물 문서 본문에 이 단어가 들어 있는
    # 정상 답변이 회피로 뒤집힌다.
    if answer.strip() == NO_EVIDENCE:
        return _fallback(request, reason="model_declined", reply=NO_EVIDENCE_REPLY)

    return {
        "complaint_state": None,
        "reply": answer,
        "result": RouteResult(),
    }


def _fallback(
    request: ConverseRequest, *, reason: str, reply: str
) -> dict[str, object]:
    logger.info(
        "knowledge_fallback building_id=%s reason=%s trace_id=%s",
        request.building_id,
        reason,
        request.trace_id,
    )
    return {
        "complaint_state": None,
        "reply": reply,
        "result": RouteResult(),
    }
