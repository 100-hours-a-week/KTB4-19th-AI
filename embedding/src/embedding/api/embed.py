from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from embedding.encoder import encoder

router = APIRouter()


class EmbedRequest(BaseModel):
    texts: list[str]


class EmbedResponse(BaseModel):
    dense: list[list[float]]
    sparse: list[dict[str, float]]


@router.post("/embed", response_model=EmbedResponse)
def embed(payload: EmbedRequest) -> EmbedResponse:
    if not encoder.ready:
        raise HTTPException(status_code=503, detail="Model is still loading")
    if not payload.texts:
        return EmbedResponse(dense=[], sparse=[])

    dense, sparse = encoder.encode(payload.texts)
    return EmbedResponse(dense=dense, sparse=sparse)
