from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from zipsai.api import health
from zipsai.errors import LlmUnavailableError
from zipsai.main import app


def test_health_returns_ok_when_dependencies_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(health, "_vector_store_ready", lambda: True)
    monkeypatch.setattr(health, "_models_ready", lambda: True)
    monkeypatch.setattr(health, "_llm_ready", lambda: True)

    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "message": "ok",
        "data": {
            "ai_service": "ready",
            "vector_store": "ready",
            "models": "ready",
            "llm": "ready",
            "prompt_version": "v1.2",
        },
    }


def test_health_returns_503_when_a_dependency_is_not_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(health, "_vector_store_ready", lambda: False)
    monkeypatch.setattr(health, "_models_ready", lambda: True)
    monkeypatch.setattr(health, "_llm_ready", lambda: True)

    response = TestClient(app).get("/health")

    assert response.status_code == 503
    assert response.json() == {
        "message": "not_ready",
        "data": {
            "ai_service": "ready",
            "vector_store": "not_ready",
            "models": "ready",
            "llm": "ready",
            "prompt_version": "v1.2",
        },
    }


@pytest.mark.parametrize(("ready", "status"), [(True, 200), (False, 503)])
def test_health_uses_cached_pod_readiness_without_network(monkeypatch, ready, status):
    monkeypatch.setattr(health, "ACTIVE_MODEL_PROVIDER", "pod")
    monkeypatch.delenv("POD_BASE_URL", raising=False)
    monkeypatch.setattr(health, "_vector_store_ready", lambda: True)
    monkeypatch.setattr(health, "_models_ready", lambda: True)
    monkeypatch.setattr(
        health, "get_locator", lambda: SimpleNamespace(is_ready=lambda: ready)
    )
    monkeypatch.setattr(
        health, "get_settings", lambda: pytest.fail("settings accessed")
    )

    response = TestClient(app).get("/health")

    assert response.status_code == status
    assert response.json()["data"]["llm"] == ("ready" if ready else "not_ready")
    assert "proxy.runpod.net" not in response.text


def test_health_uses_settings_for_manual_pod_url(monkeypatch):
    monkeypatch.setattr(health, "ACTIVE_MODEL_PROVIDER", "pod")
    monkeypatch.setenv("POD_BASE_URL", "https://manual.example/v1")
    monkeypatch.setattr(health, "get_locator", lambda: pytest.fail("locator accessed"))
    monkeypatch.setattr(health, "get_settings", lambda: object())

    assert health._llm_ready()


def test_health_falls_back_to_settings_without_locator(monkeypatch):
    monkeypatch.setattr(health, "ACTIVE_MODEL_PROVIDER", "pod")
    monkeypatch.delenv("POD_BASE_URL", raising=False)
    monkeypatch.setattr(health, "get_locator", lambda: None)
    monkeypatch.setattr(health, "get_settings", lambda: object())
    assert health._llm_ready()

    def unavailable():
        raise LlmUnavailableError("POD_BASE_URL is not configured")

    monkeypatch.setattr(health, "get_settings", unavailable)
    assert not health._llm_ready()
