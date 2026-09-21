import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

from zipsai.errors import LlmUnavailableError

load_dotenv()


@dataclass(frozen=True)
class Settings:
    llm_api_key: str
    llm_base_url: str | None
    llm_model: str
    llm_timeout_seconds: float


@lru_cache
def get_settings() -> Settings:
    # OPENROUTER_API_KEY는 LLM_API_KEY로 넘어오기 전 이름 — .env 마이그레이션 끝나면 제거
    api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise LlmUnavailableError("LLM_API_KEY is not configured")

    model = os.getenv("LLM_MODEL")
    if not model:
        raise LlmUnavailableError("LLM_MODEL is not configured")

    return Settings(
        llm_api_key=api_key,
        llm_base_url=os.getenv("LLM_BASE_URL"),
        llm_model=model,
        llm_timeout_seconds=float(os.getenv("LLM_TIMEOUT_SECONDS", "30")),
    )
