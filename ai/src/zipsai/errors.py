class PdfParseError(ValueError):
    pass


class EmbeddingError(ValueError):
    pass


class DocumentFetchError(ValueError):
    pass


class EmptyDocumentError(ValueError):
    pass


class VectorStoreError(RuntimeError):
    """Raised when the vector store cannot serve a request."""


class IntentClassificationError(ValueError):
    """Raised when the LLM returns an unsupported intent route."""


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
