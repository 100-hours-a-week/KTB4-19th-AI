class PdfParseError(ValueError):
    pass


class EmbeddingError(ValueError):
    pass


class ChunkTooLongError(EmbeddingError):
    """청크가 임베딩 서버 글자 수 한도를 넘는다. 문서 문제라 재시도하지 않는다."""

    def __init__(self, index: int, length: int, limit: int) -> None:
        self.index = index
        self.length = length
        self.limit = limit
        super().__init__(f"chunk {index} has {length} characters, limit is {limit}")


class EmbeddingRequestInvalidError(EmbeddingError):
    """임베딩 서버가 요청 자체를 거절했다. 계약이 어긋난 것이라 재시도하지 않는다."""

    def __init__(self, status_code: int) -> None:
        super().__init__(f"status={status_code}")


class DocumentFetchError(ValueError):
    pass


class EmptyDocumentError(ValueError):
    pass


class JobCancelled(Exception):
    """색인 작업이 교체·삭제로 취소됐다. 실패 기록이 아니다."""


class VectorStoreError(RuntimeError):
    """Raised when the vector store cannot serve a request."""


class IntentClassificationError(ValueError):
    """Raised when the LLM returns an unsupported intent route."""


class ComplaintExtractionError(ValueError):
    """Raised when the LLM returns an unparseable or invalid complaint draft."""


class ImageAnalysisError(ValueError):
    """Raised when the VLM returns an unparseable or invalid image analysis."""


class LlmUnavailableError(RuntimeError):
    """Raised when an LLM request cannot be completed."""


class LlmRateLimitedError(RuntimeError):
    """Raised when the LLM provider rate-limits the request."""

    def __init__(self, message: str, retry_after_seconds: int | None = None) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class LlmTimeoutError(RuntimeError):
    """Raised when the LLM request exceeds the time budget."""


class LlmUpstreamError(RuntimeError):
    """Raised when the LLM provider returns an upstream error."""


# 로그의 error_code와 응답 본문의 code가 같은 값이어야 한 필드로 두 곳을 이어
# 조회할 수 있다. 목록이 두 곳에 생기면 갈라지므로 여기만 둔다.
INTERNAL_ERROR_CODE = "INTERNAL_ERROR"

ERROR_CODES: dict[type[Exception], str] = {
    LlmRateLimitedError: "MODEL_RATE_LIMITED",
    LlmTimeoutError: "MODEL_TIMEOUT",
    LlmUpstreamError: "MODEL_UPSTREAM_ERROR",
    LlmUnavailableError: "DEPENDENCY_NOT_READY",
    IntentClassificationError: "MODEL_UPSTREAM_ERROR",
    ComplaintExtractionError: "MODEL_UPSTREAM_ERROR",
    ChunkTooLongError: "CHUNK_TOO_LONG",
    EmbeddingRequestInvalidError: "EMBEDDING_REQUEST_INVALID",
    EmbeddingError: "DEPENDENCY_NOT_READY",
    VectorStoreError: "DEPENDENCY_NOT_READY",
}


def error_code(error: Exception) -> str:
    return ERROR_CODES.get(type(error), INTERNAL_ERROR_CODE)
