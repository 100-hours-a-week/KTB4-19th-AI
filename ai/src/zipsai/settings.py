import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Final

from dotenv import load_dotenv

from zipsai.errors import LlmUnavailableError

logger = logging.getLogger(__name__)

load_dotenv()

# 댓글 API는 v2기능이기 때문에 v2로 생성한다. 기존 문서 색인(v3)와 달라 따로 명시한다.
API_PREFIX: Final = "/api/v3/ai/indexing"
COMMENT_API_PREFIX: Final = "/api/v2/ai/indexing"

EMBEDDING_DIM: Final = 1024
# 로컬 바닐라 JS 테스트 페이지가 API를 호출할 수 있도록 기본값을 둔다.
# 운영에서는 CORS_ORIGINS 환경 변수에 실제 프론트 origin을 쉼표로 지정한다.
CORS_ORIGINS: Final = tuple(
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS",
        "http://127.0.0.1:5500,http://localhost:5500,http://[::1]:5500",
    ).split(",")
    if origin.strip()
)

# 인덱싱 필수 설정: 없으면 main.py의 lifespan에서 기동을 거부한다.
QDRANT_URL: Final = os.getenv("QDRANT_URL")
S3_BUCKET: Final = os.getenv("S3_BUCKET")
QDRANT_COLLECTION: Final = os.getenv("QDRANT_COLLECTION", "documents")
EMBEDDING_API_URL: Final = os.getenv("EMBEDDING_API_URL", "http://embedding:8000")
AWS_REGION: Final = os.getenv("AWS_REGION", "ap-northeast-2")
# 인증을 켜지 않은 Qdrant에는 키가 없다. 필수로 두면 로컬과 테스트가 기동하지 못한다.
QDRANT_API_KEY: Final = os.getenv("QDRANT_API_KEY")
# 마스킹 뒤 답변이 10자보다 짧으면 "처리 완료"같은 의미 없는 답으로 보고 색인하지 않는다.
COMMENT_MIN_CHARS: Final = int(os.getenv("COMMENT_MIN_CHARS", "10"))
# 문서 색인과 답변 반영·삭제가 함께 쓰는 동시 처리 수. 넘치면 기다리지 않고 503이다.
INDEXING_MAX_CONCURRENCY: Final = int(os.getenv("INDEXING_MAX_CONCURRENCY", "2"))
# 접수카드에서 위치를 모를 때 채우는 문구는 "미상"으로 되어있다. "미상"은 그대로 들어가지 않고 위치 없음(null)과 같게 다룬다.
LOCATION_PLACEHOLDER: Final = "미상"

# LLM_MODEL과 달리 없어도 기동은 된다. 민원 이미지 분석만 이 값을 쓴다.
# compose가 빈 문자열로 넘길 수 있다. getenv의 기본값은 "키가 아예 없을 때"만
# 쓰이므로 빈 문자열이 그대로 모델명이 된다. or로 막는다.
VLM_MODEL: Final = os.getenv("VLM_MODEL") or "z-ai/glm-5.3-flash"
LLM_JUDGE_MODEL: Final = os.getenv("LLM_JUDGE_MODEL") or None

# 평가 전용. 채점 사진은 공개 URL이 없어 data URL로 넣는다. 운영 compose는
# 이 값을 절대 설정하지 않는다 — 켜지면 /converse가 임의 크기의 base64 사진을
# images[].url로 받는다.
ALLOW_DATA_URL_IMAGES: Final = (
    os.getenv("ALLOW_DATA_URL_IMAGES", "false").lower() == "true"
)

REQUIRED_SETTINGS: Final = ("QDRANT_URL", "S3_BUCKET")


def _optional_price(name: str) -> float | None:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return None
    try:
        return float(raw)
    except ValueError:
        # LogRecord.name은 로거 이름이라 설정 이름은 setting에 둔다. 값은 남기지 않는다.
        logger.warning("invalid_price_setting", extra={"setting": name})
        return None


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
    llm_judge_model: str | None = None
    vlm_model: str = VLM_MODEL
    llm_price_input_per_1m: float | None = None
    llm_price_output_per_1m: float | None = None
    vlm_price_input_per_1m: float | None = None
    vlm_price_output_per_1m: float | None = None


def call_cost_usd(
    settings: Settings, purpose: str, tokens: dict[str, int | None]
) -> float | None:
    """토큰 단가가 한 쌍으로 있을 때만 USD를 계산한다. 캐시 토큰은 빼지 않는다."""
    if purpose == "image_analysis":
        in_price = settings.vlm_price_input_per_1m
        out_price = settings.vlm_price_output_per_1m
    else:
        in_price = settings.llm_price_input_per_1m
        out_price = settings.llm_price_output_per_1m
    if in_price is None or out_price is None:
        return None
    incoming = tokens.get("input")
    outgoing = tokens.get("output")
    if incoming is None or outgoing is None:
        return None
    return incoming / 1_000_000 * in_price + outgoing / 1_000_000 * out_price


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
        llm_judge_model=LLM_JUDGE_MODEL,
        llm_price_input_per_1m=_optional_price("LLM_PRICE_INPUT_PER_1M"),
        llm_price_output_per_1m=_optional_price("LLM_PRICE_OUTPUT_PER_1M"),
        vlm_price_input_per_1m=_optional_price("VLM_PRICE_INPUT_PER_1M"),
        vlm_price_output_per_1m=_optional_price("VLM_PRICE_OUTPUT_PER_1M"),
    )
