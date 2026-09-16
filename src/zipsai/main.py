from fastapi import FastAPI

from zipsai.api.health import router as health_router

app = FastAPI(title="zipsai")
app.include_router(health_router)
