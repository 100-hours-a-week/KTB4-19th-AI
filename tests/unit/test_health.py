from fastapi.testclient import TestClient

from zipsai.main import app


def test_health_returns_ok() -> None:
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
