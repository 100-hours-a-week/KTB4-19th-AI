import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from zipsai.api.indexing import router
from zipsai.errors import JobNotFoundError
from zipsai.settings import API_PREFIX

# 설정이 없으면 파이썬이 WARNING 이상만 내보내서 단계 로그가 전부 묻힌다.
# 컨테이너 표준 출력이 곧 로그 수집 입력이라 파일로 쓰지 않는다.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

app = FastAPI(title="zipsai")
app.include_router(router, prefix=API_PREFIX)


@app.exception_handler(JobNotFoundError)
async def job_not_found(_: Request, exc: JobNotFoundError) -> JSONResponse:
    return JSONResponse(
        status_code=404,
        content={"detail": f"Job '{exc.job_id}' not found"},
    )
