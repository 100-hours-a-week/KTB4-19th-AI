from zipsai.contracts.converse import ConverseRequest
from zipsai.errors import EmbeddingError, VectorStoreError
from zipsai.integrations.qdrant import get_client
from zipsai.knowledge.evidence import citations, evidence
from zipsai.knowledge.retrieve import encode_question, query_encoder, search_chunks
from zipsai.orchestration.tool_contracts import (
    BuildingDocsArgs,
    BuildingDocsResult,
    ToolFailure,
)

_UNAVAILABLE = "건물 문서를 검색할 수 없습니다."


def search_building_docs(
    args: BuildingDocsArgs, request: ConverseRequest
) -> BuildingDocsResult | ToolFailure:
    """현재 요청의 건물 범위 안에서만 근거를 검색한다."""
    def search(query: str):
        return search_chunks(
            encode_question(query, encoder=query_encoder(), trace_id=request.trace_id),
            request.building_id,
            client=get_client(),
        )

    original = (request.message.text or "").strip()
    try:
        chunks = search(args.query)
        # v2 knowledge 노드처럼 0건이면 입주민 원문으로 한 번 더 찾는다. QA 카드 질문은 query 그대로다.
        if not chunks and original and original != args.query:
            chunks = search(original)
    except (EmbeddingError, VectorStoreError):
        return ToolFailure(code="unavailable", message=_UNAVAILABLE)
    results = evidence(chunks)
    if not results:
        return BuildingDocsResult(status="no_results")
    return BuildingDocsResult(status="ok", evidence=results, citations=citations(chunks))
