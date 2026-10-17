"""Synchronous in-process ASGI test client for the Python 3.14 test runtime.

Starlette's ``TestClient`` uses AnyIO's cross-thread blocking portal. The
Python 3.14.4 runtime used by this repository currently stalls on
``loop.call_soon_threadsafe`` across threads, including for a bare FastAPI
application. This adapter preserves the synchronous request interface while
running HTTPX's ASGI transport on the calling thread. It never opens network
connections and intentionally skips lifespan hooks. The API app currently
defines no startup or shutdown hooks, and each route dependency is mocked.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx


class InProcessTestClient:
    """Synchronous request wrapper over HTTPX's in-process ASGI transport."""

    __test__ = False  # pytest must not interpret this helper as a test class.

    def __init__(self, app: Any, *, base_url: str = "http://testserver") -> None:
        self.app = app
        self.base_url = base_url
        self.cookies = httpx.Cookies()

    def __enter__(self) -> "InProcessTestClient":
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        """Issue an HTTP request directly to ASGI without DNS or sockets."""

        async def send_request() -> httpx.Response:
            transport = httpx.ASGITransport(app=self.app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url=self.base_url,
                cookies=self.cookies,
            ) as client:
                response = await client.request(method, url, **kwargs)
                self.cookies.update(client.cookies)
                self.cookies.update(response.cookies)
                return response

        return asyncio.run(send_request())

    def get(self, url: str, **kwargs: Any) -> httpx.Response:
        """Issue an in-process GET request."""
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs: Any) -> httpx.Response:
        """Issue an in-process POST request."""
        return self.request("POST", url, **kwargs)

    def delete(self, url: str, **kwargs: Any) -> httpx.Response:
        """Issue an in-process DELETE request."""
        return self.request("DELETE", url, **kwargs)


# Keep familiar test imports while exposing the transport's in-process nature.
TestClient = InProcessTestClient
