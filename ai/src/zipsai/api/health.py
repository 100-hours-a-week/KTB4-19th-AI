from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
def health() -> dict[str, object]:
    # ponytail: static readiness until model and vector-store integrations expose probes.
    return {
        "message": "ok",
        "data": {
            "ai_service": "ready",
            "vector_store": "ready",
            "models": "ready",
            "prompt_version": "v1.2",
        },
    }
