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
def test_health_uses_available_provider_without_network(monkeypatch, ready, status):
    monkeypatch.setattr(health, "_vector_store_ready", lambda: True)
    monkeypatch.setattr(health, "_models_ready", lambda: True)

    def available_settings():
        if not ready:
            raise LlmUnavailableError("no provider available")
        return object()

    monkeypatch.setattr(health, "get_settings", available_settings)

    response = TestClient(app).get("/health")

    assert response.status_code == status
    assert response.json()["data"]["llm"] == ("ready" if ready else "not_ready")
    assert "proxy.runpod.net" not in response.text
