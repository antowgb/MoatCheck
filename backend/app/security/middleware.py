"""Small, dependency-free HTTP protections for the single-instance API.

The in-process rate limiter is a practical abuse guard for the current single
Render instance. It is not a distributed DDoS control; use a shared limiter
such as Redis when running multiple application instances.
"""

from __future__ import annotations

import base64
import hashlib
import ipaddress
import json
import os
import time
from collections import OrderedDict
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


class _InlineScriptHashes(HTMLParser):
    """Collect CSP hashes for static inline scripts emitted by Next export."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.hashes: set[str] = set()
        self._inside_inline_script = False
        self._script_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "script":
            self._inside_inline_script = not any(name.lower() == "src" for name, _ in attrs)
            self._script_parts = []

    def handle_data(self, data: str) -> None:
        if self._inside_inline_script:
            self._script_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "script" and self._inside_inline_script:
            script = "".join(self._script_parts).encode("utf-8")
            digest = base64.b64encode(hashlib.sha256(script).digest()).decode("ascii")
            self.hashes.add(f"'sha256-{digest}'")
            self._inside_inline_script = False
            self._script_parts = []


def static_inline_script_hashes(static_dir: Path) -> tuple[str, ...]:
    """Return CSP SHA-256 sources for inline scripts in exported HTML files."""
    hashes: set[str] = set()
    if not static_dir.is_dir():
        return ()
    for html_path in static_dir.rglob("*.html"):
        try:
            parser = _InlineScriptHashes()
            parser.feed(html_path.read_text(encoding="utf-8"))
            hashes.update(parser.hashes)
        except (OSError, UnicodeError):
            # A broken static page should not prevent the API from starting.
            continue
    return tuple(sorted(hashes))


def build_content_security_policy(static_dir: Path, *, production: bool) -> str:
    """Build a restrictive CSP compatible with the static Next.js export.

    Static App Router pages include framework bootstrap scripts inline. Their
    content hashes are allowed explicitly, avoiding ``unsafe-inline`` for
    scripts while keeping the export hydratable.
    """
    script_sources = " ".join(("'self'", *static_inline_script_hashes(static_dir)))
    connect_sources = ["'self'"]
    if production:
        # Supabase Auth runs in the browser. Allow only the configured project
        # origin, rather than opening connect-src to arbitrary HTTPS hosts.
        supabase_url = os.getenv("SUPABASE_URL") or os.getenv("NEXT_PUBLIC_SUPABASE_URL")
        if supabase_url:
            parsed = urlsplit(supabase_url.strip())
            if parsed.scheme == "https" and parsed.hostname and not parsed.username and not parsed.password:
                origin = f"https://{parsed.hostname}"
                try:
                    port = parsed.port
                except ValueError:
                    port = None
                    valid_port = False
                else:
                    valid_port = True
                if port and port != 443:
                    origin += f":{port}"
                if valid_port:
                    connect_sources.append(origin)
    else:
        connect_sources.extend(("http://localhost:8000", "http://127.0.0.1:8000"))
    directives = [
        "default-src 'self'",
        "base-uri 'self'",
        "object-src 'none'",
        "frame-src 'none'",
        "frame-ancestors 'none'",
        "form-action 'self'",
        "img-src 'self' data: blob: https:",
        "font-src 'self' data:",
        "style-src 'self' 'unsafe-inline'",
        f"script-src {script_sources}",
        "script-src-attr 'none'",
        f"connect-src {' '.join(connect_sources)}",
    ]
    if production:
        directives.append("upgrade-insecure-requests")
    return "; ".join(directives)


class SecurityHeadersMiddleware:
    """Attach browser hardening headers and prevent API caching."""

    def __init__(self, app: Any, *, static_dir: Path, production: bool) -> None:
        self.app = app
        self.production = production
        self.csp = build_content_security_policy(static_dir, production=production)

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_headers(message: dict[str, Any]) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                existing = {key.lower() for key, _ in headers}
                defaults = {
                    b"content-security-policy": self.csp.encode("latin-1"),
                    b"x-content-type-options": b"nosniff",
                    b"x-frame-options": b"DENY",
                    b"referrer-policy": b"strict-origin-when-cross-origin",
                    b"permissions-policy": b"camera=(), microphone=(), geolocation=()",
                    b"cross-origin-resource-policy": b"same-origin",
                }
                for key, value in defaults.items():
                    if key not in existing:
                        headers.append((key, value))
                path = scope.get("path", "")
                if path.startswith("/api/") or path == "/health":
                    if b"cache-control" not in existing:
                        headers.append((b"cache-control", b"no-store"))
                if self.production and b"strict-transport-security" not in existing:
                    headers.append((b"strict-transport-security", b"max-age=31536000; includeSubDomains"))
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_with_headers)


class APIRequestProtectionMiddleware:
    """Limit API request size with per-client and instance-wide budgets.

    The instance-wide budget matters on Render: per-client limits alone allow
    many visitors to create an unbounded aggregate burst against one web
    process. This remains process-local; use a shared store if the service is
    later scaled to multiple instances.
    """

    def __init__(
        self,
        app: Any,
        *,
        max_body_bytes: int = 2_000_000,
        requests_per_minute: int = 60,
        global_requests_per_minute: int | None = None,
        window_seconds: int = 60,
        max_tracked_clients: int = 10_000,
        trust_cf_connecting_ip: bool = False,
    ) -> None:
        if max_body_bytes <= 0 or requests_per_minute <= 0 or (global_requests_per_minute is not None and global_requests_per_minute <= 0) or window_seconds <= 0:
            raise ValueError("request protection limits must be positive")
        self.app = app
        self.max_body_bytes = max_body_bytes
        self.requests_per_minute = requests_per_minute
        self.global_requests_per_minute = global_requests_per_minute
        self.window_seconds = window_seconds
        self.max_tracked_clients = max_tracked_clients
        self.trust_cf_connecting_ip = trust_cf_connecting_ip
        self._buckets: OrderedDict[str, tuple[int, int]] = OrderedDict()

    def _client_key(self, scope: dict[str, Any]) -> str:
        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        candidate: str | None = None
        if self.trust_cf_connecting_ip:
            forwarded = headers.get(b"cf-connecting-ip")
            if forwarded:
                candidate = forwarded.decode("ascii", errors="ignore").strip()
            if not candidate:
                # Render's edge proxy exposes the original address through
                # X-Forwarded-For. Only trust it when explicitly enabled.
                forwarded = headers.get(b"x-forwarded-for") or headers.get(b"x-real-ip")
                if forwarded:
                    candidate = forwarded.decode("ascii", errors="ignore").split(",", 1)[0].strip()
        client = scope.get("client")
        candidate = candidate or (str(client[0]) if client else "unknown")
        try:
            return str(ipaddress.ip_address(candidate))
        except ValueError:
            # Never use arbitrary forwarded strings as unbounded map keys.
            return "unknown"

    @staticmethod
    async def _respond(send: Any, status: int, detail: str, headers: list[tuple[bytes, bytes]] | None = None) -> None:
        body = json.dumps({"detail": detail}).encode("utf-8")
        response_headers = [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode("ascii")),
        ]
        response_headers.extend(headers or [])
        await send({"type": "http.response.start", "status": status, "headers": response_headers})
        await send({"type": "http.response.body", "body": body})

    def _allow(self, key: str, now: float, limit: int | None = None) -> tuple[bool, int]:
        window = int(now // self.window_seconds)
        old_window, count = self._buckets.get(key, (window, 0))
        count = count + 1 if old_window == window else 1
        self._buckets[key] = (window, count)
        self._buckets.move_to_end(key)
        if len(self._buckets) > self.max_tracked_clients:
            self._buckets.popitem(last=False)
        return count <= (limit if limit is not None else self.requests_per_minute), max(1, self.window_seconds - int(now % self.window_seconds))

    def _route_limit(self, path: str) -> tuple[int, str | None]:
        normalized = path.lower()
        if "/advisor/" in normalized or normalized.endswith("/advisor"):
            return min(self.requests_per_minute, int(os.getenv("ADVISOR_RATE_LIMIT_PER_MINUTE", "20"))), "advisor"
        if any(part in normalized for part in (
            "/integrations/", "/webhook/", "/newsletter/", "/universe/",
            "/qualitative/refresh", "/refresh",
        )):
            return min(self.requests_per_minute, int(os.getenv("INTEGRATION_RATE_LIMIT_PER_MINUTE", "30"))), "integration"
        return self.requests_per_minute, None

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http" or not scope.get("path", "").startswith("/api/"):
            await self.app(scope, receive, send)
            return

        now = time.monotonic()
        path = scope.get("path", "")
        route_limit, route_bucket = self._route_limit(path)
        client_key = self._client_key(scope)
        allowed, retry_after = self._allow(f"{client_key}:{route_bucket or 'api'}", now)
        if self.global_requests_per_minute is not None:
            global_allowed, global_retry_after = self._allow("__global__", now, self.global_requests_per_minute)
            allowed = allowed and global_allowed
            retry_after = max(retry_after, global_retry_after)
        if route_bucket:
            # Route-specific counters are bounded independently of the general API budget.
            window = int(now // self.window_seconds)
            _, route_count = self._buckets.get(f"{client_key}:{route_bucket}", (window, 0))
            allowed = allowed and route_count <= route_limit
        if not allowed:
            await self._respond(
                send,
                429,
                "Request rate limit exceeded",
                [(b"retry-after", str(retry_after).encode("ascii"))],
            )
            return

        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        content_length = headers.get(b"content-length")
        if content_length:
            try:
                if int(content_length) > self.max_body_bytes:
                    await self._respond(send, 413, "Request body exceeds the configured limit")
                    return
            except ValueError:
                await self._respond(send, 400, "Invalid Content-Length header")
                return

        received_bytes = 0

        async def limited_receive() -> dict[str, Any]:
            nonlocal received_bytes
            message = await receive()
            if message["type"] == "http.request":
                received_bytes += len(message.get("body", b""))
                if received_bytes > self.max_body_bytes:
                    raise _BodyLimitExceeded
            return message

        try:
            await self.app(scope, limited_receive, send)
        except _BodyLimitExceeded:
            await self._respond(send, 413, "Request body exceeds the configured limit")


class _BodyLimitExceeded(Exception):
    """Internal signal raised when a streamed body exceeds the configured cap."""


class CookieCSRFMiddleware:
    """Require a custom request header on unsafe cookie-authenticated API calls.

    Browsers cannot send this header cross-origin without a successful CORS
    preflight, and the CORS allow-list contains only explicitly configured UI
    origins. Bearer-only API clients are unaffected.
    """

    COOKIE_NAME = b"moatcheck_session="
    PROOF_HEADER = b"x-moatcheck-request"

    def __init__(self, app: Any, *, allowed_origins: tuple[str, ...] = ()) -> None:
        self.app = app
        self.allowed_origins = set(allowed_origins)

    def _trusted_origin(self, value: str, scope: dict[str, Any], *, referer: bool = False) -> bool:
        try:
            parsed = urlsplit(value)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
                return False
            if not referer and (parsed.path or parsed.query or parsed.fragment):
                return False
            origin = f"{parsed.scheme}://{parsed.hostname.lower()}"
            if parsed.port and parsed.port != (443 if parsed.scheme == "https" else 80):
                origin += f":{parsed.port}"
            headers = {key.lower(): val for key, val in scope.get("headers", [])}
            host = headers.get(b"host", b"").decode("latin-1").lower()
            scheme = scope.get("scheme", "http")
            same_origin = origin == f"{scheme}://{host}"
            return same_origin or origin in self.allowed_origins
        except (ValueError, UnicodeError):
            return False

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http" or not scope.get("path", "").startswith("/api/"):
            await self.app(scope, receive, send)
            return
        method = scope.get("method", "GET").upper()
        if method in {"GET", "HEAD", "OPTIONS"}:
            await self.app(scope, receive, send)
            return
        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        cookies = headers.get(b"cookie", b"")
        origin = headers.get(b"origin")
        referer = headers.get(b"referer")
        if origin is not None and not self._trusted_origin(origin.decode("latin-1"), scope):
            await APIRequestProtectionMiddleware._respond(send, 403, "Untrusted request origin")
            return
        if referer is not None and not self._trusted_origin(referer.decode("latin-1"), scope, referer=True):
            await APIRequestProtectionMiddleware._respond(send, 403, "Untrusted request referer")
            return
        if self.COOKIE_NAME in cookies and origin is None and referer is None:
            await APIRequestProtectionMiddleware._respond(send, 403, "Missing request origin")
            return
        proof = headers.get(self.PROOF_HEADER)
        if self.COOKIE_NAME in cookies and proof != b"1":
            await APIRequestProtectionMiddleware._respond(send, 403, "Missing request origin proof")
            return
        await self.app(scope, receive, send)


def middleware_settings() -> dict[str, Any]:
    """Read bounded HTTP protection settings from environment variables."""
    production = os.getenv("APP_ENV", "development").lower() == "production"
    return {
        "max_body_bytes": int(os.getenv("MAX_REQUEST_BODY_BYTES", "5000000")),
        "requests_per_minute": int(os.getenv("API_RATE_LIMIT_PER_MINUTE", "600" if not production else "60")),
        "global_requests_per_minute": int(os.getenv("GLOBAL_API_RATE_LIMIT_PER_MINUTE", "1200" if not production else "300")),
        "trust_cf_connecting_ip": os.getenv("RATE_LIMIT_TRUST_CF_CONNECTING_IP", "false").lower() == "true",
    }


def csrf_settings() -> dict[str, Any]:
    configured = os.getenv("FRONTEND_ORIGINS", "")
    return {"allowed_origins": tuple(origin.strip().rstrip("/") for origin in configured.split(",") if origin.strip())}
