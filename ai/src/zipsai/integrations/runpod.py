from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Mapping
from functools import lru_cache
from typing import Any

import httpx

logger = logging.getLogger(__name__)

PODS_URL = "https://api.runpod.io/v2/pods"

_NOTHING_LOGGED = object()


class RunPodPodLocator:
    def __init__(
        self,
        *,
        client: httpx.AsyncClient,
        api_key: str,
        name_prefix: str,
        port: int = 8000,
        interval_seconds: float = 30.0,
        # 주입된 client가 timeout=None으로 만들어졌을 수 있어 요청마다 명시한다.
        request_timeout_seconds: float = 5.0,
    ) -> None:
        self._client = client
        self._api_key = api_key
        self._name_prefix = name_prefix
        self._port = port
        self._interval_seconds = interval_seconds
        self._timeout = httpx.Timeout(request_timeout_seconds)
        self._base_url: str | None = None
        self._logged: object = _NOTHING_LOGGED
        self._lookup_failing = False

    @property
    def base_url(self) -> str | None:
        """준비된 Pod의 OpenAI 호환 주소(…/v1). 없으면 None이다."""
        return self._base_url

    def is_ready(self) -> bool:
        return self._base_url is not None

    def _pick(self, pods: list[Mapping[str, Any]]) -> Mapping[str, Any] | None:
        matched = [
            pod
            for pod in pods
            if str(pod.get("name") or "").startswith(self._name_prefix)
            and pod.get("status") == "RUNNING"
        ]
        # startedAt은 stop→start마다 갱신돼 옛 Pod가 최신으로 뒤집힌다. createdAt은 불변이다.
        # RFC3339 문자열이라 사전식 비교가 시간순과 같다.
        return max(matched, key=lambda pod: str(pod.get("createdAt") or ""), default=None)

    async def _list_pods(self) -> list[Mapping[str, Any]]:
        response = await self._client.get(
            PODS_URL,
            headers={"Authorization": f"Bearer {self._api_key}"},
            timeout=self._timeout,
        )
        response.raise_for_status()
        body = response.json()
        pods = body.get("pods") if isinstance(body, Mapping) else None
        if not isinstance(pods, list):
            # ValueError로 둔다: response.json()이 던지는 JSONDecodeError도 ValueError라
            # "본문이 깨짐"과 "스키마가 다름"을 refresh()에서 한 번에 받을 수 있다.
            # TypeError로 바꾸면 우리 코드의 진짜 TypeError까지 삼키게 된다.
            raise ValueError("RunPod pods response has no pods list")  # noqa: TRY004
        return pods

    async def _healthy(self, root: str) -> bool:
        try:
            response = await self._client.get(f"{root}/health", timeout=self._timeout)
        except httpx.HTTPError:
            # 소켓은 bind됐지만 vLLM이 아직 안 받으면 연결이 행(hang)된다 → 타임아웃.
            return False
        return response.status_code == 200

    async def refresh(self) -> None:
        try:
            pods = await self._list_pods()
        except (httpx.HTTPError, ValueError):
            if not self._lookup_failing:
                logger.warning("runpod_lookup_failed", exc_info=True)
            self._lookup_failing = True
            return
        self._lookup_failing = False

        pod = self._pick(pods)
        root = f"https://{pod['id']}-{self._port}.proxy.runpod.net" if pod else None
        self._base_url = f"{root}/v1" if root and await self._healthy(root) else None

        if self._base_url != self._logged:
            logger.info(
                "runpod_pod_resolved",
                extra={
                    "pod_id": pod["id"] if pod else None,
                    "ready": self._base_url is not None,
                },
            )
            self._logged = self._base_url

    async def run(self) -> None:
        """기동 시 refresh()를 한 번 부른 뒤 이 루프를 백그라운드 태스크로 띄운다."""
        while True:
            await asyncio.sleep(self._interval_seconds)
            try:
                await self.refresh()
            except Exception:
                # 루프가 죽으면 주소가 다시는 갱신되지 않는다. 로그만 남기고 계속 돈다.
                logger.exception("runpod_lookup_crashed")


@lru_cache(maxsize=1)
def get_locator() -> RunPodPodLocator | None:
    """키와 접두사가 둘 다 있을 때만 locator를 만든다. 없으면 None(기능 꺼짐).

    settings.py가 요청마다 부르므로 커넥션 풀을 공유하도록 싱글톤으로 둔다.
    설정 상수는 보통 settings.py에 모으지만, settings가 이 모듈을 임포트하므로
    순환을 피하려고 여기서 직접 읽는다.
    """
    api_key = os.getenv("RUNPOD_API_KEY")
    # 접두사에 기본값을 두면 남의 Pod를 운영 트래픽으로 잡을 수 있다. 명시해야 켜진다.
    name_prefix = os.getenv("RUNPOD_POD_NAME_PREFIX")
    if not api_key or not name_prefix:
        return None
    return RunPodPodLocator(
        client=httpx.AsyncClient(),
        api_key=api_key,
        name_prefix=name_prefix,
        port=int(os.getenv("RUNPOD_POD_PORT", "8000")),
        interval_seconds=float(os.getenv("RUNPOD_LOOKUP_INTERVAL_SECONDS", "30")),
    )
