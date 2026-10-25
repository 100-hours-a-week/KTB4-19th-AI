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
# /converse 한 턴 전체 상한. 개별 LLM 호출(LLM_TIMEOUT_SECONDS)이 여러 번
# 연쇄될 수 있어 따로 둔다. ponytail: 상한 초과 시 응답은 끊지만, 스레드/비동기
# 작업 자체를 강제 종료하진 않는다(Python은 실행 중 스레드를 취소할 수 없음).
CONVERSE_TIMEOUT_SECONDS: Final = float(os.getenv("CONVERSE_TIMEOUT_SECONDS", "115"))

# 운영 환경변수는 모든 provider 값을 보관하고, provider 전환은 이 값을 바꿔 배포한다.
ACTIVE_MODEL_PROVIDER: Final = "serverless"
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
    vlm_model: str = ""
    llm_price_input_per_1m: float | None = None
    llm_price_output_per_1m: float | None = None
    vlm_price_input_per_1m: float | None = None
    vlm_price_output_per_1m: float | None = None
    provider: str = ACTIVE_MODEL_PROVIDER


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
def get_settings(provider_override: str | None = None) -> Settings:
    profiles = {
        "openrouter": (
            os.getenv("OPENROUTER_API_KEY"),
            os.getenv("OPENROUTER_BASE_URL"),
            os.getenv("OPENROUTER_LLM_MODEL"),
            os.getenv("OPENROUTER_VLM_MODEL"),
            "OPENROUTER_LLM_MODEL",
            "OPENROUTER_VLM_MODEL",
        ),
        "serverless": (
            os.getenv("SERVERLESS_API_KEY"),
            os.getenv("SERVERLESS_BASE_URL"),
            os.getenv("SERVERLESS_MODEL"),
            os.getenv("SERVERLESS_MODEL"),
            "SERVERLESS_MODEL",
            "SERVERLESS_MODEL",
        ),
        "pod": (
            os.getenv("POD_API_KEY"),
            os.getenv("POD_BASE_URL"),
            os.getenv("POD_MODEL"),
            os.getenv("POD_MODEL"),
            "POD_MODEL",
            "POD_MODEL",
        ),
    }
    provider = provider_override or ACTIVE_MODEL_PROVIDER
    if provider not in profiles:
        raise LlmUnavailableError(
            "ACTIVE_MODEL_PROVIDER must be openrouter, serverless, or pod"
        )
    api_key, base_url, llm_model, vlm_model, llm_model_env, vlm_model_env = profiles[
        provider
    ]
    for name, value in (
        (f"{provider.upper()}_API_KEY", api_key),
        (f"{provider.upper()}_BASE_URL", base_url),
        (llm_model_env, llm_model),
        (vlm_model_env, vlm_model),
    ):
        if not value:
            raise LlmUnavailableError(f"{name} is not configured")
    if not base_url.startswith("https://"):
        raise LlmUnavailableError("Provider base URLs must use HTTPS")

    return Settings(
        llm_api_key=api_key,
        llm_base_url=base_url,
        llm_model=llm_model,
        llm_timeout_seconds=float(os.getenv("LLM_TIMEOUT_SECONDS", "30")),
        vlm_model=vlm_model,
        provider=provider,
        llm_judge_model=LLM_JUDGE_MODEL,
        llm_price_input_per_1m=_optional_price("LLM_PRICE_INPUT_PER_1M"),
        llm_price_output_per_1m=_optional_price("LLM_PRICE_OUTPUT_PER_1M"),
        vlm_price_input_per_1m=_optional_price("VLM_PRICE_INPUT_PER_1M"),
        vlm_price_output_per_1m=_optional_price("VLM_PRICE_OUTPUT_PER_1M"),
    )
