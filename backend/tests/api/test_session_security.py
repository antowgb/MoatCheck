from uuid import uuid4

from tests.asgi_client import TestClient

from app.api.v1.endpoints import session as session_endpoint
from app.main import app


def test_session_exchange_sets_short_lived_httponly_cookie(monkeypatch):
    user_id = uuid4()

    async def verify(token):
        assert token == "valid-user-token"
        return type("User", (), {"id": user_id})()

    monkeypatch.setattr(session_endpoint, "verify_supabase_access_token", verify)
    monkeypatch.setenv("COOKIE_SECURE", "true")

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/session",
            headers={"Authorization": "Bearer valid-user-token", "X-MoatCheck-Request": "1"},
        )

    assert response.status_code == 200, response.text
    cookie = response.cookies.get("moatcheck_session")
    assert cookie == "valid-user-token"
    set_cookie = response.headers["set-cookie"].lower()
    assert "httponly" in set_cookie
    assert "secure" in set_cookie
    assert "samesite=lax" in set_cookie
    assert "path=/api" in set_cookie
    assert "max-age=3600" in set_cookie


def test_session_exchange_rejects_invalid_token(monkeypatch):
    async def verify(_token):
        raise ValueError("invalid")

    monkeypatch.setattr(session_endpoint, "verify_supabase_access_token", verify)
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/session",
            headers={"Authorization": "Bearer invalid-token", "X-MoatCheck-Request": "1"},
        )
    assert response.status_code == 401
    assert "moatcheck_session" not in response.headers.get("set-cookie", "")


def test_sign_out_expires_httponly_cookie_with_matching_api_scope(monkeypatch):
    monkeypatch.setenv("COOKIE_SECURE", "true")
    with TestClient(app) as client:
        response = client.delete("/api/v1/session", headers={"X-MoatCheck-Request": "1"})
    assert response.status_code == 200
    cookie = response.headers["set-cookie"].lower()
    assert "max-age=0" in cookie
    assert "httponly" in cookie and "secure" in cookie and "path=/api" in cookie
