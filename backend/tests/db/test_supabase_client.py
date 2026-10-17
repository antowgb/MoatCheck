"""Supabase client setup tests; no credentials or network calls are used."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from app.db import supabase


def test_client_factory_reads_service_role_environment(monkeypatch) -> None:
    client = object()
    factory = AsyncMock(return_value=client)
    monkeypatch.setattr(supabase, "_client", None)
    monkeypatch.setattr(supabase, "create_async_client", factory)
    monkeypatch.setenv("SUPABASE_URL", "https://quant.example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "eyJtest-service-secret")

    result = asyncio.run(supabase.get_async_supabase_client())

    assert result is client
    factory.assert_awaited_once_with(
        "https://quant.example.supabase.co", "eyJtest-service-secret"
    )


def test_client_factory_prefers_new_secret_key_without_bearer_default(monkeypatch) -> None:
    client = type(
        "Client",
        (),
        {
            "options": type("Options", (), {"headers": {"Authorization": "Bearer sb_secret"}})(),
            "auth": type("Auth", (), {"_headers": {"Authorization": "Bearer sb_secret"}})(),
        },
    )()
    factory = AsyncMock(return_value=client)
    monkeypatch.setattr(supabase, "_client", None)
    monkeypatch.setattr(supabase, "create_async_client", factory)
    monkeypatch.setenv("SUPABASE_URL", "https://quant.example.supabase.co")
    monkeypatch.setenv("SUPABASE_SECRET_KEY", "sb_secret_test-only")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "eyJlegacy-fallback")

    result = asyncio.run(supabase.get_async_supabase_client())

    assert result is client
    assert factory.await_args.args == (
        "https://quant.example.supabase.co",
        "sb_secret_test-only",
    )
    options = factory.await_args.kwargs["options"]
    assert options.auto_refresh_token is False
    assert options.persist_session is False
    assert "Authorization" not in client.options.headers
    assert "Authorization" not in client.auth._headers


def test_invalid_new_secret_falls_back_to_valid_legacy_key(monkeypatch) -> None:
    client = object()
    factory = AsyncMock(return_value=client)
    monkeypatch.setattr(supabase, "_client", None)
    monkeypatch.setattr(supabase, "create_async_client", factory)
    monkeypatch.setenv("SUPABASE_URL", "https://quant.example.supabase.co")
    monkeypatch.setenv("SUPABASE_SECRET_KEY", "misconfigured-publishable-key")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "eyJvalid-legacy-service-role")

    result = asyncio.run(supabase.get_async_supabase_client())

    assert result is client
    factory.assert_awaited_once_with(
        "https://quant.example.supabase.co", "eyJvalid-legacy-service-role"
    )


def test_missing_supabase_environment_fails_without_client_creation(monkeypatch) -> None:
    factory = AsyncMock()
    monkeypatch.setattr(supabase, "_client", None)
    monkeypatch.setattr(supabase, "create_async_client", factory)
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
    monkeypatch.delenv("SUPABASE_SECRET_KEY", raising=False)

    with pytest.raises(RuntimeError, match="SUPABASE_URL and a valid server key"):
        asyncio.run(supabase.get_async_supabase_client())
    factory.assert_not_awaited()
