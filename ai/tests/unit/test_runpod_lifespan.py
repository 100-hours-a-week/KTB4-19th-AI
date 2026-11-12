import asyncio
from importlib import import_module

import pytest

main_module = import_module("zipsai.main")


def test_lifespan_refreshes_before_serving_and_stops_locator(monkeypatch):
    class Locator:
        def __init__(self):
            self.refreshed = False
            self.started = asyncio.Event()
            self.cancelled = False
            self.closed = False

        async def refresh(self):
            self.refreshed = True

        async def run(self):
            self.started.set()
            try:
                await asyncio.Event().wait()
            finally:
                self.cancelled = True

        async def aclose(self):
            self.closed = True

    locator = Locator()

    def locator_factory():
        return locator

    cleared = []
    locator_factory.cache_clear = lambda: cleared.append(True)
    monkeypatch.setattr(main_module, "ACTIVE_MODEL_PROVIDER", "pod")
    monkeypatch.setattr(main_module, "get_locator", locator_factory)
    monkeypatch.setattr(main_module, "missing_required_settings", list)
    monkeypatch.setattr(main_module, "get_tracing_client", lambda: None)

    async def check():
        async with main_module.lifespan(main_module.app):
            assert locator.refreshed
            await asyncio.wait_for(locator.started.wait(), timeout=1)
        assert locator.cancelled
        assert locator.closed
        assert cleared == [True]

    asyncio.run(check())


def test_lifespan_skips_locator_for_serverless(monkeypatch):
    monkeypatch.setattr(main_module, "ACTIVE_MODEL_PROVIDER", "serverless")
    monkeypatch.setattr(
        main_module, "get_locator", lambda: pytest.fail("locator accessed")
    )
    monkeypatch.setattr(main_module, "missing_required_settings", list)
    monkeypatch.setattr(main_module, "get_tracing_client", lambda: None)

    async def check():
        async with main_module.lifespan(main_module.app):
            pass

    asyncio.run(check())


def test_lifespan_starts_without_runpod_credentials(monkeypatch):
    monkeypatch.setattr(main_module, "ACTIVE_MODEL_PROVIDER", "pod")
    monkeypatch.setattr(main_module, "get_locator", lambda: None)
    monkeypatch.setattr(main_module, "missing_required_settings", list)
    monkeypatch.setattr(main_module, "get_tracing_client", lambda: None)

    async def check():
        async with main_module.lifespan(main_module.app):
            pass

    asyncio.run(check())
