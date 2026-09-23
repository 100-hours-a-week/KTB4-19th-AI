import logging

from qdrant_client import models

from zipsai.contracts.converse import (
    Citation,
    ConverseRequest,
    QaCardDraft,
    RouteResult,
)
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

# 청크가 400자까지 나오는데 그대로 실어 보내면 응답이 화면에 담기지 않는다.
SNIPPET_LENGTH = 200


def handle_knowledge(request: ConverseRequest) -> dict[str, object]:
    question = (request.message.text or "").strip()
    if not question:
        # 이미지만 온 요청은 intent가 complaint로 보내지만, current_route=knowledge로
        # 유지되는 경로가 있어 검색까지 가기 전에 막는다.
        # 질문이 없으니 QA 카드도 만들지 않는다. 관리자에게 빈 질문이 전달된다.
        return _fallback(request, reason="empty_question", reply=EMPTY_QUESTION_REPLY)

    chunks = search_chunks(
        encode_question(question, encoder=query_encoder()),
        request.building_id,
        client=get_client(),
    )
    if not chunks:
        return _fallback(
            request,
            reason="no_hit",
            reply=NO_EVIDENCE_REPLY,
            qa_question=question,
        )

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
        return _fallback(
            request,
            reason="model_declined",
            reply=NO_EVIDENCE_REPLY,
            qa_question=question,
        )

    return {
        "complaint_state": None,
        "reply": answer,
        "result": RouteResult(
            citations=_citations(chunks),
            has_sufficient_evidence=True,
        ),
    }


def _fallback(
    request: ConverseRequest,
    *,
    reason: str,
    reply: str,
    qa_question: str | None = None,
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
        "result": RouteResult(
            qa_card_draft=QaCardDraft(question=qa_question) if qa_question else None,
            has_sufficient_evidence=False,
        ),
    }


def _citations(chunks: list[models.ScoredPoint]) -> list[Citation]:
    """검색에 쓴 문서를 근거 목록으로 만든다.

    한 문서에서 여러 청크가 걸리므로 doc_id로 접는다. 제목으로 접으면 "공지사항"처럼
    제목이 겹치는 다른 문서가 한 건으로 뭉개진다. LLM이 실제로 어느 청크를 인용했는지는
    자유 텍스트 답변에서 알 수 없어, 검색된 문서를 모두 싣는다.
    """
    citations: list[Citation] = []
    seen: set[str] = set()
    for chunk in chunks:
        payload = chunk.payload or {}
        title = payload.get("title") or "제목 없음"
        doc_id = payload.get("doc_id")
        key = doc_id or title
        if key in seen:
            continue
        seen.add(key)
        snippet = (payload.get("text") or "")[:SNIPPET_LENGTH]
        citations.append(
            Citation(
                source_type="building_document",
                source_id=doc_id,
                title=title,
                snippet=snippet or None,
                url=None,
            )
        )
    return citations
