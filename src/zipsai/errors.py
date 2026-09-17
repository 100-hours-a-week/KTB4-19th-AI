class IntentClassificationError(ValueError):
    """Raised when the LLM returns an unsupported intent route."""


class LlmUnavailableError(RuntimeError):
    """Raised when an LLM request cannot be completed."""
