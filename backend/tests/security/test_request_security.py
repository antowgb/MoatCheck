from fastapi import FastAPI

from app.security.middleware import CookieCSRFMiddleware, SecurityHeadersMiddleware
from tests.asgi_client import TestClient


def test_cookie_write_rejects_untrusted_origin_and_missing_origin():
    app = FastAPI()
    app.add_middleware(CookieCSRFMiddleware, allowed_origins=("https://app.example.test",))

    @app.post("/api/write")
    async def write():
        return {"ok": True}

    with TestClient(app) as client:
        client.cookies.set("moatcheck_session", "opaque")
        missing = client.post("/api/write", headers={"X-MoatCheck-Request": "1"})
        foreign = client.post(
            "/api/write",
            headers={"X-MoatCheck-Request": "1", "Origin": "https://evil.example"},
        )
        allowed = client.post(
            "/api/write",
            headers={"X-MoatCheck-Request": "1", "Origin": "https://app.example.test"},
        )
    assert missing.status_code == 403
    assert foreign.status_code == 403
    assert allowed.status_code == 200


def test_production_hsts_includes_subdomains(tmp_path):
    app = FastAPI()
    app.add_middleware(SecurityHeadersMiddleware, static_dir=tmp_path, production=True)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    with TestClient(app) as client:
        response = client.get("/health")
    assert response.headers["strict-transport-security"] == "max-age=31536000; includeSubDomains"
