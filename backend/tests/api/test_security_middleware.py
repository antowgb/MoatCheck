from fastapi import FastAPI, Request
from tests.asgi_client import TestClient

from app.security.middleware import (
    APIRequestProtectionMiddleware,
    CookieCSRFMiddleware,
    SecurityHeadersMiddleware,
    build_content_security_policy,
)


def test_api_request_body_limit_rejects_oversized_payload():
    app = FastAPI()
    app.add_middleware(APIRequestProtectionMiddleware, max_body_bytes=10, requests_per_minute=10)

    @app.post("/api/calculate")
    async def calculate(request: Request):
        return {"length": len(await request.body())}

    with TestClient(app) as client:
        response = client.post("/api/calculate", content=b"01234567890")
    assert response.status_code == 413


def test_api_request_rate_limit_is_bounded_per_client():
    app = FastAPI()
    app.add_middleware(
        APIRequestProtectionMiddleware,
        max_body_bytes=100,
        requests_per_minute=2,
        window_seconds=60,
    )

    @app.get("/api/calculate")
    async def calculate():
        return {"ok": True}

    with TestClient(app) as client:
        assert client.get("/api/calculate").status_code == 200
        assert client.get("/api/calculate").status_code == 200
        response = client.get("/api/calculate")
    assert response.status_code == 429
    assert int(response.headers["retry-after"]) > 0


def test_api_request_global_rate_limit_is_shared_across_clients():
    app = FastAPI()
    app.add_middleware(
        APIRequestProtectionMiddleware,
        max_body_bytes=100,
        requests_per_minute=10,
        global_requests_per_minute=2,
        window_seconds=60,
        trust_cf_connecting_ip=True,
    )

    @app.get("/api/calculate")
    async def calculate():
        return {"ok": True}

    with TestClient(app) as client:
        assert client.get("/api/calculate", headers={"CF-Connecting-IP": "203.0.113.10"}).status_code == 200
        assert client.get("/api/calculate", headers={"CF-Connecting-IP": "203.0.113.11"}).status_code == 200
        response = client.get("/api/calculate", headers={"CF-Connecting-IP": "203.0.113.12"})
    assert response.status_code == 429


def test_cookie_authenticated_write_requires_request_proof_header():
    app = FastAPI()
    app.add_middleware(CookieCSRFMiddleware)

    @app.post("/api/write")
    async def write():
        return {"ok": True}

    with TestClient(app) as client:
        client.cookies.set("moatcheck_session", "opaque")
        blocked = client.post("/api/write")
        allowed = client.post(
            "/api/write",
            headers={"X-MoatCheck-Request": "1", "Origin": "http://testserver"},
        )
    assert blocked.status_code == 403
    assert allowed.status_code == 200


def test_security_headers_and_csp_hash_inline_static_scripts(tmp_path):
    (tmp_path / "index.html").write_text(
        '<script>window.bootstrap = true;</script><script src="/_next/static/app.js"></script>',
        encoding="utf-8",
    )
    policy = build_content_security_policy(tmp_path, production=True)
    assert "'unsafe-inline'" not in policy.partition("script-src ")[2].partition(";")[0]
    assert "'sha256-" in policy

    app = FastAPI()
    app.add_middleware(SecurityHeadersMiddleware, static_dir=tmp_path, production=True)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    with TestClient(app) as client:
        response = client.get("/health")
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["strict-transport-security"] == "max-age=31536000; includeSubDomains"
    assert response.headers["cache-control"] == "no-store"


def test_production_csp_allows_only_the_configured_supabase_auth_origin(tmp_path, monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://project-ref.supabase.co")

    policy = build_content_security_policy(tmp_path, production=True)

    assert "connect-src 'self' https://project-ref.supabase.co" in policy
    assert "https://other-project.supabase.co" not in policy

    monkeypatch.setenv("SUPABASE_URL", "http://project-ref.supabase.co")
    policy_without_https = build_content_security_policy(tmp_path, production=True)
    assert "connect-src 'self';" in policy_without_https
