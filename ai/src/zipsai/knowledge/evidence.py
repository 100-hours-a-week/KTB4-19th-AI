from qdrant_client import models

from zipsai.contracts.converse import Citation
from zipsai.integrations.qdrant import COMMENT_SOURCE_TYPE
from zipsai.orchestration.tool_contracts import SearchEvidence

SNIPPET_LENGTH = 200
EVIDENCE_LENGTH = 400


def _text(payload: dict) -> str:
    is_comment = payload.get("source_type") == COMMENT_SOURCE_TYPE
    text = payload.get("answer") if is_comment else payload.get("text")
    return text if isinstance(text, str) else ""


def _citation(payload: dict) -> Citation:
    title = payload.get("title") or "제목 없음"
    is_comment = payload.get("source_type") == COMMENT_SOURCE_TYPE
    source_id = (
        str(payload["complaint_detail_id"])
        if is_comment and "complaint_detail_id" in payload
        else payload.get("doc_id")
    )
    return Citation(
        source_type="qa_history" if is_comment else "building_document",
        source_id=source_id,
        title=title,
        snippet=_text(payload)[:SNIPPET_LENGTH] or None,
        url=None,
    )


def citations(chunks: list[models.ScoredPoint]) -> list[Citation]:
    """검색 청크를 입주민에게 공개 가능한 문서 근거로 바꾼다. 문서당 하나만 남긴다."""
    result: list[Citation] = []
    seen: set[str] = set()
    for chunk in chunks:
        citation = _citation(chunk.payload or {})
        key = citation.source_id or citation.title
        if key in seen:
            continue
        seen.add(key)
        result.append(citation)
    return result


def evidence(chunks: list[models.ScoredPoint]) -> list[SearchEvidence]:
    """답변 모델에 넘길 근거. 같은 청크만 중복으로 본다.

    정답이 한 문서의 둘째 청크에 있을 수 있어 문서 단위로 자르지 않는다.
    """
    result: list[SearchEvidence] = []
    seen: set[object] = set()
    for chunk in chunks:
        payload = chunk.payload or {}
        text = _text(payload)[:EVIDENCE_LENGTH]
        # 저장된 payload에는 청크 번호가 없어 점 id로 같은 청크를 가린다.
        if chunk.id in seen or not text:
            continue
        seen.add(chunk.id)
        result.append(SearchEvidence(text=text, citation=_citation(payload)))
    return result
