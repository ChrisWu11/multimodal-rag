from fastapi.testclient import TestClient

from app.main import app


def test_health() -> None:
    client = TestClient(app)
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_model_config() -> None:
    client = TestClient(app)
    response = client.get("/api/model-config")
    assert response.status_code == 200
    body = response.json()
    assert "llm_provider" in body
    assert "model_options" in body
