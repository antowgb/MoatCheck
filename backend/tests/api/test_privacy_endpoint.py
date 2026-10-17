from uuid import uuid4

from tests.asgi_client import TestClient

from app.api.dependencies import require_current_user
from app.api.v1.endpoints import privacy as privacy_endpoint
from app.main import app


def test_privacy_delete_requires_authenticated_user():
    with TestClient(app) as client:
        response = client.delete(
            "/api/v1/privacy/data",
            headers={"X-MoatCheck-Request": "1"},
        )
    assert response.status_code == 401


def test_privacy_delete_uses_verified_owner_id(monkeypatch):
    owner_id = uuid4()
    observed: dict[str, object] = {}

    class FakeRepository:
        def __init__(self, _client):
            pass

        async def delete_owner_data(self, actual_owner_id):
            observed["owner_id"] = actual_owner_id
            return {
                "portfolios_deleted": 1,
                "backtest_runs_deleted": 2,
                "stress_tests_deleted": 3,
            }

    async def fake_client():
        return object()

    async def fake_current_user():
        return type("User", (), {"id": owner_id})()

    monkeypatch.setattr(privacy_endpoint, "get_async_supabase_client", fake_client)
    monkeypatch.setattr(privacy_endpoint, "PrivacyRepository", FakeRepository)
    app.dependency_overrides[require_current_user] = fake_current_user
    try:
        with TestClient(app) as client:
            response = client.delete(
                "/api/v1/privacy/data",
                headers={"X-MoatCheck-Request": "1"},
            )
    finally:
        app.dependency_overrides.pop(require_current_user, None)

    assert response.status_code == 200, response.text
    assert observed["owner_id"] == owner_id
    assert response.json()["data"]["deleted"] == {
        "portfolios_deleted": 1,
        "backtest_runs_deleted": 2,
        "stress_tests_deleted": 3,
    }
