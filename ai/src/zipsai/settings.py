import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Final

from dotenv import load_dotenv

from zipsai.errors import LlmUnavailableError

load_dotenv()

API_PREFIX: Final = "/api/v3/ai/indexing"
EMBEDDING_DIM: Final = 1024

# 인덱싱 필수 설정: 없으면 main.py의 lifespan에서 기동을 거부한다.
QDRANT_URL: Final = os.getenv("QDRANT_URL")
S3_BUCKET: Final = os.getenv("S3_BUCKET")
QDRANT_COLLECTION: Final = os.getenv("QDRANT_COLLECTION", "documents")
EMBEDDING_API_URL: Final = os.getenv("EMBEDDING_API_URL", "http://embedding:8000")
AWS_REGION: Final = os.getenv("AWS_REGION", "ap-northeast-2")
# 인증을 켜지 않은 Qdrant에는 키가 없다. 필수로 두면 로컬과 테스트가 기동하지 못한다.
QDRANT_API_KEY: Final = os.getenv("QDRANT_API_KEY")

REQUIRED_SETTINGS: Final = ("QDRANT_URL", "S3_BUCKET")


def missing_required_settings() -> list[str]:
    values = {
        "QDRANT_URL": QDRANT_URL,
        "S3_BUCKET": S3_BUCKET,
    }
    return [name for name in REQUIRED_SETTINGS if not values[name]]


@dataclass(frozen=True)
class Settings:
    llm_api_key: str
    llm_base_url: str | None
    llm_model: str
    llm_timeout_seconds: float


@lru_cache
def get_settings() -> Settings:
    # OPENROUTER_API_KEY는 LLM_API_KEY로 넘어오기 전 이름
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
