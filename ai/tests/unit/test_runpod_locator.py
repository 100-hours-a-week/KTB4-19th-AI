import asyncio
import logging

import httpx

from zipsai.integrations.runpod import RunPodPodLocator, get_locator

PODS_URL = "api.runpod.io"


def locator_for(handler, **kwargs) -> RunPodPodLocator:
    return RunPodPodLocator(
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        api_key="rpa_test",
        name_prefix="zipsai-vllm",
        **kwargs,
    )


def pod(pod_id: str, *, name="zipsai-vllm", status="RUNNING", created="2026-10-01T00:00:00Z"):
    return {"id": pod_id, "name": name, "status": status, "createdAt": created}


def responder(pods: list[dict], *, health_status: int = 200):
    """RunPod 목록 조회와 pod 프록시 /health를 호스트로 갈라 응답한다."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == PODS_URL:
            return httpx.Response(200, json={"pods": pods})
        return httpx.Response(health_status)

    return handler


def test_running_pod_becomes_base_url() -> None:
    locator = locator_for(responder([pod("abc123")]))

    asyncio.run(locator.refresh())

    assert locator.base_url == "https://abc123-8000.proxy.runpod.net/v1"
    assert locator.is_ready() is True


def test_pod_with_other_name_is_ignored() -> None:
    locator = locator_for(responder([pod("abc123", name="someone-else-pod")]))

    asyncio.run(locator.refresh())

    assert locator.base_url is None
    assert locator.is_ready() is False


def test_stopped_pod_is_ignored() -> None:
    locator = locator_for(responder([pod("abc123", status="EXITED")]))

    asyncio.run(locator.refresh())

    assert locator.base_url is None


def test_newest_created_pod_wins() -> None:
    # 마이그레이션 직후 옛 Pod가 남아 있는 상황. 이름에 접미어가 붙어도 접두사로 잡힌다.
    locator = locator_for(
        responder(
            [
                pod("old111", created="2026-10-01T00:00:00Z"),
                pod("new222", name="zipsai-vllm-migrated", created="2026-10-02T00:00:00Z"),
            ]
        )
    )

    asyncio.run(locator.refresh())

    assert locator.base_url == "https://new222-8000.proxy.runpod.net/v1"


def test_running_pod_without_vllm_health_is_not_ready() -> None:
    # RUNNING은 컨테이너가 떴다는 뜻일 뿐이다. 모델 로딩 중에는 주소를 내주지 않는다.
    locator = locator_for(responder([pod("abc123")], health_status=503))

    asyncio.run(locator.refresh())

    assert locator.base_url is None


def test_health_check_goes_to_the_pod_proxy() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        if request.url.host == PODS_URL:
            return httpx.Response(200, json={"pods": [pod("abc123")]})
        return httpx.Response(200)

    asyncio.run(locator_for(handler).refresh())

    assert "https://abc123-8000.proxy.runpod.net/health" in seen


def test_lookup_failure_keeps_the_last_known_base_url() -> None:
    # RunPod API가 죽어도 Pod는 멀쩡할 수 있다. 조회 실패로 서비스를 끊지 않는다.
    failing = False

    def handler(request: httpx.Request) -> httpx.Response:
        if failing and request.url.host == PODS_URL:
            raise httpx.ConnectError("runpod down", request=request)
        if request.url.host == PODS_URL:
            return httpx.Response(200, json={"pods": [pod("abc123")]})
        return httpx.Response(200)

    locator = locator_for(handler)
    asyncio.run(locator.refresh())
    assert locator.base_url == "https://abc123-8000.proxy.runpod.net/v1"

    failing = True
    asyncio.run(locator.refresh())

    assert locator.base_url == "https://abc123-8000.proxy.runpod.net/v1"


def test_lookup_failure_before_any_success_stays_none() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("runpod down", request=request)

    locator = locator_for(handler)
    asyncio.run(locator.refresh())

    assert locator.base_url is None


def test_unreachable_pod_health_is_not_ready() -> None:
    # 소켓은 bind됐지만 vLLM이 아직 안 받으면 연결이 행(hang)된다 → 타임아웃으로 떨어진다.
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == PODS_URL:
            return httpx.Response(200, json={"pods": [pod("abc123")]})
        raise httpx.ConnectTimeout("no answer", request=request)

    locator = locator_for(handler)
    asyncio.run(locator.refresh())

    assert locator.base_url is None


def test_unchanged_state_is_logged_once(caplog) -> None:
    locator = locator_for(responder([pod("abc123")]))

    with caplog.at_level(logging.INFO, logger="zipsai.integrations.runpod"):
        asyncio.run(locator.refresh())
        asyncio.run(locator.refresh())
        asyncio.run(locator.refresh())

    assert [r.message for r in caplog.records].count("runpod_pod_resolved") == 1


def test_first_result_is_logged_even_when_no_pod_is_up(caplog) -> None:
    # 기동했는데 Pod가 꺼져 있는 경우도 한 번은 알려줘야 배선이 됐는지 확인할 수 있다.
    locator = locator_for(responder([]))

    with caplog.at_level(logging.INFO, logger="zipsai.integrations.runpod"):
        asyncio.run(locator.refresh())

    assert [r.message for r in caplog.records].count("runpod_pod_resolved") == 1


def test_repeated_lookup_failure_warns_once(caplog) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("runpod down", request=request)

    locator = locator_for(handler)

    with caplog.at_level(logging.WARNING, logger="zipsai.integrations.runpod"):
        asyncio.run(locator.refresh())
        asyncio.run(locator.refresh())
        asyncio.run(locator.refresh())

    assert [r.message for r in caplog.records].count("runpod_lookup_failed") == 1


def test_pod_id_is_logged_but_not_the_proxy_url(caplog) -> None:
    # 프록시 URL은 아는 사람 누구나 때릴 수 있는 공개 주소다. 로그에 흘리지 않는다.
    locator = locator_for(responder([pod("abc123")]))

    with caplog.at_level(logging.INFO, logger="zipsai.integrations.runpod"):
        asyncio.run(locator.refresh())

    logged = [(r.message, getattr(r, "pod_id", None), getattr(r, "ready", None)) for r in caplog.records]
    assert ("runpod_pod_resolved", "abc123", True) in logged
    assert all("proxy.runpod.net" not in r.getMessage() for r in caplog.records)


async def drive_run(locator: RunPodPodLocator, turns: int = 10) -> None:
    """run()을 몇 바퀴 돌리고 취소한다. interval_seconds=0으로 시간에 묶이지 않게 한다."""
    task = asyncio.create_task(locator.run())
    for _ in range(turns):
        await asyncio.sleep(0)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


def test_run_refreshes_repeatedly() -> None:
    calls: list[int] = []

    class Counting(RunPodPodLocator):
        async def refresh(self) -> None:
            calls.append(1)

    locator = Counting(
        client=httpx.AsyncClient(transport=httpx.MockTransport(responder([]))),
        api_key="rpa_test",
        name_prefix="zipsai-vllm",
        interval_seconds=0,
    )
    asyncio.run(drive_run(locator))

    assert len(calls) >= 2


def test_run_survives_a_crashing_refresh() -> None:
    # 루프가 죽으면 주소가 다시는 갱신되지 않는다. 예외를 먹고 계속 돌아야 한다.
    calls: list[int] = []

    class Crashing(RunPodPodLocator):
        async def refresh(self) -> None:
            calls.append(1)
            raise RuntimeError("boom")

    locator = Crashing(
        client=httpx.AsyncClient(transport=httpx.MockTransport(responder([]))),
        api_key="rpa_test",
        name_prefix="zipsai-vllm",
        interval_seconds=0,
    )
    asyncio.run(drive_run(locator))

    assert len(calls) >= 2


def test_malformed_pods_response_keeps_the_last_known_base_url() -> None:
    broken = False

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host != PODS_URL:
            return httpx.Response(200)
        if broken:
            return httpx.Response(200, json={"unexpected": "shape"})
        return httpx.Response(200, json={"pods": [pod("abc123")]})

    locator = locator_for(handler)
    asyncio.run(locator.refresh())

    broken = True
    asyncio.run(locator.refresh())

    assert locator.base_url == "https://abc123-8000.proxy.runpod.net/v1"


def set_env(monkeypatch, **values) -> None:
    for name in ("RUNPOD_API_KEY", "RUNPOD_POD_NAME_PREFIX", "RUNPOD_POD_PORT"):
        monkeypatch.delenv(name, raising=False)
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    get_locator.cache_clear()


def test_factory_returns_none_without_an_api_key(monkeypatch) -> None:
    # 로컬·CI에는 키가 없다. 기동을 막지 않고 조용히 꺼져야 한다.
    set_env(monkeypatch, RUNPOD_POD_NAME_PREFIX="zipsai-vllm")

    assert get_locator() is None


def test_factory_returns_none_without_a_name_prefix(monkeypatch) -> None:
    # 접두사 기본값을 두면 남의 Pod를 잡을 수 있다. 명시하지 않으면 끈다.
    set_env(monkeypatch, RUNPOD_API_KEY="rpa_test")

    assert get_locator() is None


def test_factory_builds_a_locator_from_env(monkeypatch) -> None:
    set_env(monkeypatch, RUNPOD_API_KEY="rpa_test", RUNPOD_POD_NAME_PREFIX="zipsai-vllm")

    locator = get_locator()

    assert isinstance(locator, RunPodPodLocator)
    assert locator.base_url is None


def test_factory_is_a_singleton(monkeypatch) -> None:
    # settings.py가 요청마다 부른다. 매번 새 커넥션 풀을 만들면 안 된다.
    set_env(monkeypatch, RUNPOD_API_KEY="rpa_test", RUNPOD_POD_NAME_PREFIX="zipsai-vllm")

    assert get_locator() is get_locator()
