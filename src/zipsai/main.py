import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from zipsai.api.indexing import router
from zipsai.settings import API_PREFIX, missing_required_settings

# 설정이 없으면 파이썬이 WARNING 이상만 내보내서 단계 로그가 전부 묻힌다.
# 컨테이너 표준 출력이 곧 로그 수집 입력이라 파일로 쓰지 않는다.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    missing = missing_required_settings()
    if missing:
        # 뜬 다음 첫 색인에서 실패하면 원인을 찾느라 로그를 뒤져야 한다.
        raise RuntimeError(f"Missing required settings: {', '.join(missing)}")
    yield


app = FastAPI(title="zipsai", lifespan=lifespan)
app.include_router(router, prefix=API_PREFIX)
