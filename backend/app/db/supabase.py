"""Lazy asynchronous Supabase client configured from backend-only API keys.

The client is deliberately not created at import time, so pure core modules,
CLI tooling, and unit tests do not require Supabase credentials or network
access. The service-role key must only be used by trusted backend code.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Final

from dotenv import load_dotenv
from supabase import AsyncClient, create_async_client
from supabase.lib.client_options import AsyncClientOptions

load_dotenv()
logger = logging.getLogger(__name__)

_client: AsyncClient | None = None
_client_lock: Final[asyncio.Lock] = asyncio.Lock()


async def get_async_supabase_client() -> AsyncClient:
    """Return the process-wide async client, creating it on first use.

    Reads ``SUPABASE_SECRET_KEY`` (preferred) or the legacy
    ``SUPABASE_SERVICE_ROLE_KEY`` at call time. New-format secret keys must be
    sent as the ``apikey`` header, not as a Bearer JWT. Their default
    Authorization header is therefore removed after client creation; Auth's
    ``get_user(jwt)`` still supplies the verified user's JWT per request.
    Raises ``RuntimeError`` when configuration is missing and does not make
    any table request. Concurrent first calls share one client instance.
    """
    global _client
    if _client is not None:
        return _client
    async with _client_lock:
        if _client is not None:
            return _client
        url = os.getenv("SUPABASE_URL")
        configured_secret_key = os.getenv("SUPABASE_SECRET_KEY", "").strip()
        configured_legacy_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
        # Do not treat a publishable/anon key or placeholder value as a server
        # secret. A malformed preferred key must not shadow a valid legacy key.
        if configured_secret_key.startswith("sb_secret_"):
            api_key = configured_secret_key
            is_new_secret = True
        elif configured_secret_key.startswith("eyJ"):
            # Tolerate an old service_role JWT accidentally stored under the
            # newer variable name; it still has legacy Authorization semantics.
            api_key = configured_secret_key
            is_new_secret = False
            logger.warning("SUPABASE_SECRET_KEY contains a legacy JWT key; prefer an sb_secret_ key")
        elif configured_legacy_key.startswith("eyJ"):
            api_key = configured_legacy_key
            is_new_secret = False
            if configured_secret_key:
                logger.warning("Ignoring malformed SUPABASE_SECRET_KEY and using SUPABASE_SERVICE_ROLE_KEY")
        else:
            api_key = ""
            is_new_secret = False

        if not url or not api_key:
            raise RuntimeError(
                "SUPABASE_URL and a valid server key are required: set "
                "SUPABASE_SECRET_KEY (sb_secret_...) or "
                "SUPABASE_SERVICE_ROLE_KEY (legacy JWT). SUPABASE_KEY is public-only."
            )
        if is_new_secret:
            options = AsyncClientOptions(auto_refresh_token=False, persist_session=False)
            _client = await create_async_client(url, api_key, options=options)
            # supabase-py currently seeds Authorization with ``Bearer <key>``.
            # A new sb_secret key is API-key-only, so strip that default before
            # lazy PostgREST client construction. Auth.get_user(jwt) explicitly
            # adds the user's JWT for the Auth request itself.
            _client.options.headers.pop("Authorization", None)
            _client.auth._headers.pop("Authorization", None)
        else:
            _client = await create_async_client(url, api_key)
        return _client
