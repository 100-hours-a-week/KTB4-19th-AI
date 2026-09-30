import logging
import os

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from embedding.encoder import encoder

logger = logging.getLogger(__name__)
router = APIRouter()

# CPU 추론이라 한 요청이 쥐는 메모리와 시간을 여기서 묶는다.
# 호출 측(ai-api)이 이 크기로 잘라 보내며, 값을 올리면 양쪽을 같이 올려야 한다.
MAX_BATCH_SIZE = 32
# 청크는 400자 기준이라 정상 요청은 여기 닿지 않는다. 오용을 막는 상한이다.
MAX_TEXT_CHARS = 8000


class EmbedRequest(BaseModel):
    texts: list[str]
    trace_id: str


class EmbedResponse(BaseModel):
    dense: list[list[float]]
    sparse: list[dict[str, float]]


@router.post("/embed", response_model=EmbedResponse)
def embed(payload: EmbedRequest) -> EmbedResponse:
    if not encoder.ready:
        raise HTTPException(status_code=503, detail="Model is still loading")
    if len(payload.texts) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=422,
            detail=f"texts must hold at most {MAX_BATCH_SIZE} items",
        )
    if any(len(text) > MAX_TEXT_CHARS for text in payload.texts):
        raise HTTPException(
            status_code=422,
            detail=f"each text must be at most {MAX_TEXT_CHARS} characters",
        )
    if not payload.texts:
        return EmbedResponse(dense=[], sparse=[])

    # 구조화 로거가 없어 trace_id를 메시지 문자열에 싣는다 — CloudWatch
    # filter-log-events로 값 검색은 되지만 필드 파싱은 안 된다.
    # load는 CPU 경합 의심 시 그 순간 값을 사후에 지표 권한 없이도 로그로 확인하기 위함.
    logger.info(
        "embed request trace_id=%s texts=%d load=%s",
        payload.trace_id,
        len(payload.texts),
        os.getloadavg(),
    )
    dense, sparse = encoder.encode(payload.texts)
    # uvicorn 접속 로그("POST /embed ... 200 OK")엔 trace_id가 안 실린다.
    # 요청 줄과 시간순으로만 대조하던 걸 trace_id로 직접 맞출 수 있게 완료 줄을 따로 남긴다.
    logger.info("embed done trace_id=%s texts=%d", payload.trace_id, len(payload.texts))
    return EmbedResponse(dense=dense, sparse=sparse)
