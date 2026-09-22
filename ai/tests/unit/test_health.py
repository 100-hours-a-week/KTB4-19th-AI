import pytest
from fastapi.testclient import TestClient

from zipsai.api import health
from zipsai.main import app


def test_health_returns_ok_when_dependencies_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(health, "_vector_store_ready", lambda: True)
    monkeypatch.setattr(health, "_models_ready", lambda: True)

    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "message": "ok",
        "data": {
            "ai_service": "ready",
            "vector_store": "ready",
            "models": "ready",
            "prompt_version": "v1.2",
        },
    }


def test_health_returns_503_when_a_dependency_is_not_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(health, "_vector_store_ready", lambda: False)
    monkeypatch.setattr(health, "_models_ready", lambda: True)

    response = TestClient(app).get("/health")

    assert response.status_code == 503
    assert response.json() == {
        "message": "not_ready",
        "data": {
            "ai_service": "ready",
            "vector_store": "not_ready",
            "models": "ready",
            "prompt_version": "v1.2",
        },
    }
