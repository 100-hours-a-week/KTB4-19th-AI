from qdrant_client import models

from zipsai.contracts.converse import Citation
from zipsai.integrations.qdrant import COMMENT_SOURCE_TYPE
from zipsai.orchestration.tool_contracts import SearchEvidence

SNIPPET_LENGTH = 200


def citations(chunks: list[models.ScoredPoint]) -> list[Citation]:
    """검색 청크를 입주민에게 공개 가능한 문서 근거로 바꾼다."""
    result: list[Citation] = []
    seen: set[str] = set()
    for chunk in chunks:
        payload = chunk.payload or {}
        title = payload.get("title") or "제목 없음"
        is_comment = payload.get("source_type") == COMMENT_SOURCE_TYPE
        source_id = (
            str(payload["complaint_detail_id"])
            if is_comment and "complaint_detail_id" in payload
            else payload.get("doc_id")
        )
        key = source_id or title
        if key in seen:
            continue
        seen.add(key)
        text = payload.get("answer") if is_comment else payload.get("text")
        snippet = text[:SNIPPET_LENGTH] if isinstance(text, str) else None
        result.append(
            Citation(
                source_type="qa_history" if is_comment else "building_document",
                source_id=source_id,
                title=title,
                snippet=snippet or None,
                url=None,
            )
        )
    return result


def evidence(chunks: list[models.ScoredPoint]) -> list[SearchEvidence]:
    return [
        SearchEvidence(text=citation.snippet, citation=citation)
        for citation in citations(chunks)
        if citation.snippet
    ]
