from zipsai.complaint.node import handle_complaint
from zipsai.contracts.converse import ConverseRequest, ImageAnalysis
from zipsai.errors import (
    ComplaintExtractionError,
    LlmRateLimitedError,
    LlmTimeoutError,
    LlmUnavailableError,
    LlmUpstreamError,
)
from zipsai.orchestration.tool_contracts import (
    ComplaintTurnArgs,
    ComplaintTurnResult,
    ToolFailure,
)

_FAILURE_MESSAGE = "민원 내용을 처리할 수 없습니다."


def process_complaint_turn(
    _args: ComplaintTurnArgs,
    request: ConverseRequest,
    *,
    image_analysis: ImageAnalysis | None = None,
    contextualized_message: str | None = None,
) -> ComplaintTurnResult | ToolFailure:
    """현재 서버 문맥으로 민원 한 턴을 처리한다. 실제 민원 접수는 수행하지 않는다."""
    try:
        result = handle_complaint(
            request,
            image_analysis=image_analysis,
            contextualized_message=contextualized_message,
        )
    except LlmRateLimitedError:
        return ToolFailure(code="rate_limited", message=_FAILURE_MESSAGE)
    except LlmTimeoutError:
        return ToolFailure(code="timeout", message=_FAILURE_MESSAGE)
    except LlmUnavailableError:
        return ToolFailure(code="unavailable", message=_FAILURE_MESSAGE)
    except (LlmUpstreamError, ComplaintExtractionError):
        return ToolFailure(code="upstream_error", message=_FAILURE_MESSAGE)
    return ComplaintTurnResult.model_validate(result)
