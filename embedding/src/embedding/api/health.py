from fastapi import APIRouter, HTTPException

from embedding.encoder import encoder

router = APIRouter()


@router.get("/liveness")
def liveness() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/health")
def health() -> dict[str, str]:
    # 모델 적재 전에는 요청을 처리할 수 없으므로 준비 완료로 보고하지 않는다.
    if not encoder.ready:
        raise HTTPException(status_code=503, detail="Model is still loading")
    return {"status": "ok"}


@router.get("/readiness")
def readiness() -> dict[str, str]:
    return health()
