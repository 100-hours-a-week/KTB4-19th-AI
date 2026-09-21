from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from embedding.encoder import encoder

router = APIRouter()

# CPU 추론이라 한 요청이 쥐는 메모리와 시간을 여기서 묶는다.
# 호출 측(ai-api)이 이 크기로 잘라 보내며, 값을 올리면 양쪽을 같이 올려야 한다.
MAX_BATCH_SIZE = 32
# 청크는 400자 기준이라 정상 요청은 여기 닿지 않는다. 오용을 막는 상한이다.
MAX_TEXT_CHARS = 8000


class EmbedRequest(BaseModel):
    texts: list[str]


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

    dense, sparse = encoder.encode(payload.texts)
    return EmbedResponse(dense=dense, sparse=sparse)
