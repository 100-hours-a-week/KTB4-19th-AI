import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from embedding.api.embed import _tracing_client
from embedding.api.embed import router as embed_router
from embedding.api.health import router as health_router
from embedding.encoder import encoder

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    # 기동 시 미리 적재한다. 첫 요청이 수십 초를 기다리면 호출 측이 먼저 끊는다.
    encoder.load()
    try:
        yield
    finally:
        client = _tracing_client()
        if client:
            client.flush()


app = FastAPI(title="embedding", lifespan=lifespan)
app.include_router(health_router)
app.include_router(embed_router)
