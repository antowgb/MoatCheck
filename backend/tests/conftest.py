"""Shared API test isolation for external services and Python 3.14 thread handoff."""

from __future__ import annotations

import socket
import urllib.request
from unittest.mock import AsyncMock, MagicMock

import anyio.to_thread
import pytest


@pytest.fixture(autouse=True)
def isolate_api_tests_from_external_io(monkeypatch: pytest.MonkeyPatch) -> None:
    """Forbid live database, mail, provider, or socket I/O during tests.

    Async repositories are guarded at the endpoint import sites with
    ``AsyncMock``. Email is mocked after persistence. Socket guards fail fast
    if a provider or database client accidentally escapes its fixture.

    The installed Python 3.14.4 build stalls when an asyncio loop is woken
    through ``call_soon_threadsafe`` from another thread. The API test adapter
    therefore invokes sync handlers inline by patching AnyIO only for the
    lifetime of each test. Production AnyIO behavior is unchanged.
    """

    async def inline_run_sync(func, *args, **kwargs):
        kwargs.pop("abandon_on_cancel", None)
        kwargs.pop("cancellable", None)
        kwargs.pop("limiter", None)
        return func(*args, **kwargs)

    monkeypatch.setattr(anyio.to_thread, "run_sync", inline_run_sync)

    async def forbidden_supabase_call(*_args, **_kwargs):
        raise AssertionError("Live Supabase access is forbidden in API tests")

    for module_path in (
        "app.api.dependencies",
        "app.api.v1.endpoints.advisor",
        "app.api.v1.endpoints.backtest",
        "app.api.v1.endpoints.portfolio",
        "app.api.v1.endpoints.privacy",
        "app.api.v1.endpoints.risk",
    ):
        monkeypatch.setattr(
            f"{module_path}.get_async_supabase_client",
            AsyncMock(side_effect=forbidden_supabase_call),
        )

    # Legacy sync data routes are out of scope for these integration tests.
    monkeypatch.setattr(
        "app.api.routes.get_supabase",
        MagicMock(side_effect=AssertionError("Live Supabase access is forbidden in API tests")),
    )
    monkeypatch.setattr(
        "app.advisor.email.send_recommendation_email",
        AsyncMock(return_value="mocked_email_delivery"),
    )
    monkeypatch.setattr(
        "app.api.v1.endpoints.advisor.send_recommendation_email",
        AsyncMock(return_value="mocked_email_delivery"),
    )

    def forbidden_network(*_args, **_kwargs):
        raise AssertionError("Live network access is forbidden in API tests")

    monkeypatch.setattr(socket.socket, "connect", forbidden_network)
    monkeypatch.setattr(socket, "create_connection", forbidden_network)
    monkeypatch.setattr(urllib.request, "urlopen", forbidden_network)
