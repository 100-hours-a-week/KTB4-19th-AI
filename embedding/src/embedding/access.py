"""HTTP 접속 한 줄을 JSON으로 남긴다. 쿼리와 본문은 싣지 않는다."""

import logging
import time
from typing import Any

logger = logging.getLogger("embedding.access")

_QUIET_PATHS = frozenset({"/health", "/metrics"})


class AccessLogMiddleware:
    def __init__(self, app: Any, record_http: Any | None = None) -> None:
        self.app = app
        self._record_http = record_http

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path = scope.get("path") or ""
        if path in _QUIET_PATHS:
            await self.app(scope, receive, send)
            return

        started = time.perf_counter()
        status = 500

        async def send_wrapper(message: dict) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            route = scope.get("route")
            template = getattr(route, "path", None) or "unmatched"
            duration_ms = int((time.perf_counter() - started) * 1000)
            logger.info(
                "http_access",
                extra={
                    "method": scope.get("method"),
                    "path": template,
                    "status": status,
                    "duration_ms": duration_ms,
                },
            )
            if self._record_http is not None:
                self._record_http(
                    scope.get("method") or "UNKNOWN",
                    template,
                    status,
                    duration_ms / 1000,
                )
