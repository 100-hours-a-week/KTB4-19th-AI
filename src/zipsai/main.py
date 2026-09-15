from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from zipsai.api.indexing import router
from zipsai.errors import JobNotFoundError
from zipsai.settings import API_PREFIX

app = FastAPI(title="zipsai")
app.include_router(router, prefix=API_PREFIX)


@app.exception_handler(JobNotFoundError)
async def job_not_found(_: Request, exc: JobNotFoundError) -> JSONResponse:
    return JSONResponse(
        status_code=404,
        content={"detail": f"Job '{exc.job_id}' not found"},
    )
