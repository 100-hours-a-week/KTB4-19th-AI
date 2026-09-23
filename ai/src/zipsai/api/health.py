import logging
from functools import lru_cache

import httpx
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from zipsai.errors import LlmUnavailableError
from zipsai.integrations.qdrant import create_client
from zipsai.settings import EMBEDDING_API_URL, get_settings

logger = logging.getLogger(__name__)
router = APIRouter()

PROMPT_VERSION = "v1.2"
HEALTH_CHECK_TIMEOUT_SECONDS = 2.0


@lru_cache(maxsize=1)
def _qdrant_client():
    return create_client()


def _vector_store_ready() -> bool:
    try:
        _qdrant_client().get_collections()
        return True
    except Exception:
        logger.exception("health_check_vector_store_failed")
        return False


def _models_ready() -> bool:
    try:
        get_settings()
    except LlmUnavailableError:
        return False
    try:
        response = httpx.get(
            f"{EMBEDDING_API_URL}/health", timeout=HEALTH_CHECK_TIMEOUT_SECONDS
        )
        return response.status_code == 200
    except httpx.HTTPError:
        return False


@router.get("/health", response_model=None)
def health() -> dict[str, object] | JSONResponse:
    vector_store_ready = _vector_store_ready()
    models_ready = _models_ready()

    data = {
        "ai_service": "ready",
        "vector_store": "ready" if vector_store_ready else "not_ready",
        "models": "ready" if models_ready else "not_ready",
        "prompt_version": PROMPT_VERSION,
    }
    if vector_store_ready and models_ready:
        return {"message": "ok", "data": data}
    return JSONResponse(status_code=503, content={"message": "not_ready", "data": data})
