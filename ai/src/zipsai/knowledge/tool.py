from zipsai.contracts.converse import ConverseRequest
from zipsai.errors import EmbeddingError, VectorStoreError
from zipsai.integrations.qdrant import get_client
from zipsai.knowledge.evidence import evidence
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
    try:
        chunks = search_chunks(
            encode_question(
                args.query, encoder=query_encoder(), trace_id=request.trace_id
            ),
            request.building_id,
            client=get_client(),
        )
    except (EmbeddingError, VectorStoreError):
        return ToolFailure(code="unavailable", message=_UNAVAILABLE)
    results = evidence(chunks)
    if not results:
        return BuildingDocsResult(status="no_results")
    return BuildingDocsResult(status="ok", evidence=results)
