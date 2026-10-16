import logging
from collections.abc import Callable
from dataclasses import dataclass

from qdrant_client import models

from zipsai.contracts.converse import (
    Citation,
    ConverseRequest,
    ImageAnalysis,
    QaCardDraft,
    Route,
    RouteResult,
)
from zipsai.integrations.llm import generate_text, generate_text_stream
from zipsai.integrations.qdrant import get_client
from zipsai.knowledge.prompts import (
    IMAGE_QUESTION_PROMPT,
    KNOWLEDGE_PROMPT,
    NO_EVIDENCE,
    format_context,
)
from zipsai.knowledge.retrieve import encode_question, query_encoder, search_chunks
from zipsai.observability import skipped, stage

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


@dataclass(frozen=True)
class _KnowledgePreparation:
    question: str
    chunks: list[models.ScoredPoint]
    messages: list[object]


def handle_knowledge(
    request: ConverseRequest,
    image_analysis: ImageAnalysis | None = None,
    contextualized_message: str | None = None,
) -> dict[str, object]:
    prepared = _prepare_knowledge(request, image_analysis, contextualized_message)
    if isinstance(prepared, dict):
        return prepared
    with stage("generate", logger) as step:
        answer = generate_text(
            system_prompt=str(prepared.messages[0].content),
            user_prompt=str(prepared.messages[1].content),
            usage_sink=step,
        )
        declined = answer.strip() == NO_EVIDENCE
        step["answer_chars"] = len(answer)
        step["declined"] = declined

    if declined:
        return _fallback(
            request,
            reason="model_declined",
            reply=NO_EVIDENCE_REPLY,
            qa_question=prepared.question,
        )

    return _answer_result(request, answer, prepared.chunks)


async def handle_knowledge_stream(
    request: ConverseRequest,
    emit: Callable[[str], None],
    image_analysis: ImageAnalysis | None = None,
    contextualized_message: str | None = None,
) -> dict[str, object]:
    """Knowledge 답변만 토큰 단위로 생성한다.

    검색·근거 구성은 기존과 동일하게 수행하고, LLM 생성 구간만 비동기로
    소비한다. ``emit``은 LangGraph custom stream writer를 전달받는다.
    """
    prepared = _prepare_knowledge(request, image_analysis, contextualized_message)
    if isinstance(prepared, dict):
        return prepared

    answer_parts: list[str] = []
    pending = ""
    with stage("generate", logger) as step:
        async for text in generate_text_stream(
            system_prompt=str(prepared.messages[0].content),
            user_prompt=str(prepared.messages[1].content),
            usage_sink=step,
        ):
            answer_parts.append(text)
            pending += text
            if NO_EVIDENCE.startswith(pending):
                continue
            emit(pending)
            pending = ""

    answer = "".join(answer_parts)
    declined = answer.strip() == NO_EVIDENCE
    if not declined and pending:
        emit(pending)

    if declined:
        return _fallback(
            request,
            reason="model_declined",
            reply=NO_EVIDENCE_REPLY,
            qa_question=prepared.question,
        )

    return _answer_result(request, answer, prepared.chunks)


def _prepare_knowledge(
    request: ConverseRequest,
    image_analysis: ImageAnalysis | None = None,
    contextualized_message: str | None = None,
) -> _KnowledgePreparation | dict[str, object]:
    question = (request.message.text or "").strip()
    if not question and request.message.images:
        question = next(
            (
                (turn.text or "").strip()
                for turn in reversed(request.conversation_history)
                if turn.role == "user" and (turn.text or "").strip()
            ),
            "",
        )
    if not question:
        for name in ("encode", "gate", "hybrid", "context", "generate"):
            skipped(name, logger, skip_reason="empty_question")
        return _fallback(request, reason="empty_question", reply=EMPTY_QUESTION_REPLY)

    previous_image_turn = None
    image_observations = image_analysis.images if image_analysis else []
    if (
        not request.message.images
        and request.message.text
        and request.current_route is Route.CLARIFY
    ):
        previous_image_turn = next(
            (
                turn
                for turn in reversed(request.conversation_history)
                if turn.role == "user"
            ),
            None,
        )
        image_observations = (
            previous_image_turn.images or [] if previous_image_turn else []
        )

    observations = [
        value.strip()
        for image in image_observations
        for value in (image.summary, image.ocr_text)
        if value and value.strip()
    ]
    image_context = "; ".join(observations)
    search_question = question
    if image_context:
        with stage("query_rewrite", logger) as step:
            prompt_parts = [f"질문: {question}"]
            contextualized_question = (contextualized_message or "").strip()
            if contextualized_question and contextualized_question != question:
                prompt_parts.append(f"맥락 보강 참고문: {contextualized_question}")
            if previous_image_turn and previous_image_turn.text:
                prompt_parts.append(
                    f"사진 첨부 때 발화: {previous_image_turn.text.strip()}"
                )
            prompt_parts.append(f"사진 분석 맥락: {image_context}")
            rewritten = generate_text(
                system_prompt=IMAGE_QUESTION_PROMPT,
                user_prompt="\n".join(prompt_parts),
                usage_sink=step,
            ).strip()
            search_question = rewritten or question
            step["question_chars"] = len(search_question)

    def search(candidate: str) -> list[models.ScoredPoint]:
        return search_chunks(
            encode_question(
                candidate, encoder=query_encoder(), trace_id=request.trace_id
            ),
            request.building_id,
            client=get_client(),
        )

    chunks = search(search_question)
    contextualized_question = (contextualized_message or "").strip()
    if (
        not chunks
        and not image_context
        and contextualized_question
        and contextualized_question != search_question
    ):
        search_question = contextualized_question
        chunks = search(search_question)

    if not chunks:
        skipped("context", logger, skip_reason="no_hit")
        skipped("generate", logger, skip_reason="no_hit")
        return _fallback(
            request,
            reason="no_hit",
            reply=NO_EVIDENCE_REPLY,
            qa_question=search_question,
        )

    with stage("context", logger) as step:
        context = format_context(chunks)
        step["chunks"] = [
            {
                "doc_id": (chunk.payload or {}).get("doc_id"),
                "page": (chunk.payload or {}).get("page"),
                "rank": rank,
                "chars": len((chunk.payload or {}).get("text") or ""),
            }
            for rank, chunk in enumerate(chunks, start=1)
        ]
        step["context_chars"] = len(context)

    return _KnowledgePreparation(
        question=question,
        chunks=chunks,
        messages=KNOWLEDGE_PROMPT.format_messages(
            question=search_question, context=context
        ),
    )


def _answer_result(
    request: ConverseRequest,
    answer: str,
    chunks: list[models.ScoredPoint],
) -> dict[str, object]:
    return {
        "complaint_state": None,
        "reply": answer,
        "result": RouteResult(
            complaint_draft=request.complaint_draft,
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
        "fallback",
        extra={"reason": reason, "has_question": bool(qa_question)},
    )
    return {
        "complaint_state": None,
        "reply": reply,
        "result": RouteResult(
            complaint_draft=request.complaint_draft,
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
