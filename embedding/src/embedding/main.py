from fastapi import FastAPI

from embedding.api.health import router as health_router

app = FastAPI(title="embedding")
app.include_router(health_router)
