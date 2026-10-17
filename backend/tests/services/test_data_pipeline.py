"""Offline tests for provider routing, normalization, analytics, and upserts."""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from app.services.data_pipeline_orchestrator import (
    DataPipelineOrchestrator,
    FallbackHandler,
    ProviderError,
    ProviderRateLimitError,
    RateLimiter,
    derive_market_analytics,
    fetch_and_cache_eod_prices,
    normalize_prices,
    _json_safe,
)


class FakeQuery:
    def __init__(self, client, table):
        self.client = client
        self.table_name = table
        self.rows = None

    def select(self, *_args, **_kwargs):
        return self

    def eq(self, *_args, **_kwargs):
        return self

    def order(self, *_args, **_kwargs):
        return self

    def limit(self, *_args, **_kwargs):
        return self

    def in_(self, *_args, **_kwargs):
        return self

    def upsert(self, rows, **kwargs):
        self.rows = rows
        self.conflict = kwargs.get("on_conflict")
        return self

    async def execute(self):
        if self.rows is not None:
            self.client.writes.append((self.table_name, self.conflict, self.rows))
        return SimpleNamespace(data=self.client.stocks if self.table_name == "stocks" else [])


class FakeSupabase:
    def __init__(self):
        self.writes = []
        self.stocks = []

    def table(self, name):
        return FakeQuery(self, name)


def test_normalize_prices_adjusts_ohlc_and_deduplicates_dates():
    source = pd.DataFrame(
        {
            "Open": [100, 101], "High": [110, 111], "Low": [90, 91],
            "Close": [105, 106], "Adj Close": [52.5, 53], "Volume": [1000, 1200],
        },
        index=pd.to_datetime(["2025-01-02", "2025-01-02"]),
    )
    normalized = normalize_prices(source, "abc", "yfinance")
    assert len(normalized) == 1
    assert normalized.iloc[0]["ticker"] == "ABC"
    assert normalized.iloc[0]["open"] == pytest.approx(50.5)
    assert normalized.iloc[0]["close"] == pytest.approx(53.0)
    assert normalized.iloc[0]["volume"] == 1200


def test_derived_data_aligns_calendar_and_calculates_log_return_adv_and_covariance():
    prices = normalize_prices(
        pd.DataFrame(
            {"date": ["2025-01-02", "2025-01-03", "2025-01-06"],
             "close": [100, 101, 102], "volume": [10, 10, 10]},
        ),
        "A", "fixture",
    )
    prices_b = normalize_prices(
        pd.DataFrame({"date": ["2025-01-02", "2025-01-03", "2025-01-06"],
                      "close": [50, 51, 52], "volume": [20, 20, 20]}), "B", "fixture"
    )
    result = derive_market_analytics(pd.concat([prices, prices_b], ignore_index=True), adv_window=2)
    assert result["aligned_closes"].shape[1] == 2
    assert result["log_returns"].iloc[1]["A"] == pytest.approx(np.log(101 / 100))
    assert result["adv"]["A"] > 0
    assert result["rolling_covariance"].shape == (2, 2)


def test_jsonb_payload_sanitizer_converts_non_finite_values_to_null():
    assert _json_safe({"x": np.float64(np.nan), "y": [np.inf, 2.0]}) == {"x": None, "y": [None, 2.0]}


def test_rate_limiter_rejects_calls_after_daily_quota():
    limiter = RateLimiter(daily_limits={"fmp": 1}, now=lambda: datetime(2026, 1, 1, tzinfo=timezone.utc))
    async def run():
        await limiter.acquire("fmp")
        assert limiter.usage() == {"fmp": 1}
        with pytest.raises(ProviderRateLimitError):
            await limiter.acquire("FMP")
    asyncio.run(run())


def test_tiingo_hourly_limit_is_enforced_without_waiting_for_provider_429():
    now = datetime(2026, 1, 1, 15, tzinfo=timezone.utc)
    limiter = RateLimiter(
        daily_limits={"tiingo": 1000}, hourly_limits={"tiingo": 1},
        provider_timezones={"tiingo": "America/New_York"}, now=lambda: now,
    )
    async def run():
        await limiter.acquire("tiingo")
        with pytest.raises(ProviderRateLimitError, match="hourly"):
            await limiter.acquire("tiingo")
    asyncio.run(run())


def test_fallback_handler_uses_secondary_provider_when_primary_fails():
    calls = []

    async def broken():
        calls.append("primary")
        raise ProviderRateLimitError("quota")

    async def secondary():
        calls.append("secondary")
        return [1]

    data, name, attempts = asyncio.run(FallbackHandler().run([("primary", broken), ("secondary", secondary)]))
    assert data == [1]
    assert name == "secondary"
    assert calls == ["primary", "secondary"]
    assert [a.succeeded for a in attempts] == [False, True]


def test_http_429_is_mapped_to_rate_limit_error_without_real_network():
    class Response:
        status_code = 429
        def json(self):
            return {"message": "rate limit"}

    class HttpClient:
        async def get(self, *_args, **_kwargs):
            return Response()

    service = DataPipelineOrchestrator(http_client=HttpClient(), db_client=FakeSupabase())
    async def run():
        with pytest.raises(ProviderRateLimitError):
            await service._get_json("fred", "https://fixture.invalid/api")
    asyncio.run(run())


def test_fmp_request_adapter_uses_mocked_http_and_requires_license(monkeypatch):
    monkeypatch.setenv("FMP_API_KEY", "test-only")
    monkeypatch.setenv("FMP_DATA_LICENSE_CONFIRMED", "true")
    seen = {}

    class Response:
        status_code = 200
        def json(self):
            return [{"date": "2025-01-03", "close": 100}]

    class HttpClient:
        async def get(self, url, **kwargs):
            seen["url"] = url
            seen.update(kwargs)
            return Response()

    service = DataPipelineOrchestrator(http_client=HttpClient(), db_client=FakeSupabase())
    async def run():
        raw = await service._fmp_prices("ABC")
        assert raw[0]["close"] == 100
    asyncio.run(run())
    assert seen["url"].endswith("/stable/historical-price-eod/full")
    assert seen["params"]["apikey"] == "test-only"


def test_fmp_persistence_is_blocked_until_data_rights_are_confirmed(monkeypatch):
    monkeypatch.setenv("FMP_API_KEY", "test-only")
    monkeypatch.delenv("FMP_DATA_LICENSE_CONFIRMED", raising=False)
    service = DataPipelineOrchestrator(db_client=FakeSupabase())
    async def run():
        with pytest.raises(ProviderError, match="rights are confirmed"):
            await service._fmp_prices("ABC")
    asyncio.run(run())


def test_tiingo_and_fred_adapters_use_injected_http_client(monkeypatch):
    monkeypatch.setenv("TIINGO_API_KEY", "tiingo-test")
    monkeypatch.setenv("TIINGO_PERSISTENCE_ALLOWED", "true")
    monkeypatch.setenv("FRED_API_KEY", "fred-test")
    seen = []

    class Response:
        status_code = 200
        def __init__(self, payload):
            self.payload = payload
        def json(self):
            return self.payload

    class HttpClient:
        async def get(self, url, **kwargs):
            seen.append((url, kwargs))
            if "tiingo" in url:
                return Response([{"date": "2025-01-03", "close": 10, "adjClose": 9.5, "volume": 100}])
            return Response({"observations": [{"date": "2025-01-03", "value": "4.2"}]})

    service = DataPipelineOrchestrator(http_client=HttpClient(), db_client=FakeSupabase())
    async def run():
        bars = await service._tiingo_prices("ABC")
        observations = await service._fred_series("DGS10", "2025-01-01")
        return bars, observations
    bars, observations = asyncio.run(run())
    assert bars[0]["adjClose"] == 9.5
    assert observations[0]["value"] == "4.2"
    assert len(seen) == 2
    assert seen[0][1]["params"]["token"] == "tiingo-test"
    assert seen[1][1]["params"]["api_key"] == "fred-test"


def test_existing_sec_rss_runner_is_mockable_and_uses_legacy_service_key(monkeypatch):
    monkeypatch.delenv("SUPABASE_KEY", raising=False)
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "mock-server-key")
    called = []
    def runner():
        called.append(os.environ.get("SUPABASE_KEY"))
        return {"written": 2}
    service = DataPipelineOrchestrator(qualitative_runner=runner)
    result = asyncio.run(service.fetch_qualitative())
    assert result == {"written": 2}
    assert called == ["mock-server-key"]
    assert "SUPABASE_KEY" not in os.environ


def test_pipeline_uses_mocked_providers_and_idempotent_upsert_contracts():
    async def prices(_ticker):
        return pd.DataFrame(
            {"date": ["2025-01-02", "2025-01-03", "2025-01-06"],
             "open": [100, 101, 102], "high": [101, 102, 103],
             "low": [99, 100, 101], "close": [100, 101, 102],
             "volume": [1000, 1100, 1200]}
        )

    async def ratios(ticker):
        return [{"symbol": ticker, "date": "2025-01-03", "priceEarningsRatio": 20.0}]

    async def macro(series_id, _start_date):
        return [{"date": "2025-01-03", "value": "4.25"}]

    supabase = FakeSupabase()
    supabase.stocks = [{"ticker": "ABC"}]
    async def qualitative():
        return {"written": 0}

    service = DataPipelineOrchestrator(
        db_client=supabase,
        ticker_list=["ABC"],
        provider_overrides={
            "yfinance": prices,
            "fmp_fundamentals": ratios,
            "fred": macro,
        },
        qualitative_runner=qualitative,
    )
    report = asyncio.run(service.run(include_qualitative=True))
    assert report["ok"] is True
    assert report["prices_upserted"] == 3
    assert report["macro_indicators_upserted"] == 4
    assert report["company_fundamentals_upserted"] == 1
    assert report["analytics"]["log_returns_shape"] == [3, 1]
    assert report["qualitative"] == {"written": 0}

    writes = {table: (conflict, rows) for table, conflict, rows in supabase.writes}
    assert writes["prices"][0] == "ticker,date"
    assert writes["macro_indicators"][0] == "series_id,observation_date,provider"
    assert writes["company_fundamentals"][0] == "ticker,as_of_date,provider"
    assert "price_history" not in writes
    assert writes["prices"][1][0]["log_return"] is None
    assert writes["prices"][1][-1]["adv"] > 0


def test_no_tickers_produces_successful_empty_report_without_provider_calls():
    service = DataPipelineOrchestrator(db_client=FakeSupabase(), ticker_list=[])
    # Explicit empty list means use PIPELINE_TICKERS if configured, so isolate it.
    service.ticker_list = []
    service.fred_series = {}
    report = asyncio.run(service.run(include_qualitative=False))
    assert report["ok"] is True
    assert report["prices_upserted"] == 0
    assert report["analytics"]["input_shape"] == [0, 0]


def test_on_demand_eod_refresh_normalizes_and_upserts_price_cache(monkeypatch):
    frame = normalize_prices(
        pd.DataFrame({
            "date": ["2026-10-15", "2026-10-16"],
            "open": [99.0, 100.0], "high": [101.0, 102.0],
            "low": [98.0, 99.0], "close": [100.0, 101.0],
            "volume": [1000, 1200],
        }),
        "NEW", "fixture",
    )
    writes = []

    async def fetch_prices(self, ticker):
        assert ticker == "NEW"
        return frame.copy(), "fixture"

    async def upsert(self, table, rows, conflict):
        writes.append((table, rows, conflict))
        return len(rows)

    async def no_op(*_args, **_kwargs):
        return None

    monkeypatch.setattr(DataPipelineOrchestrator, "fetch_prices", fetch_prices)
    monkeypatch.setattr(DataPipelineOrchestrator, "_upsert", upsert)
    monkeypatch.setattr(DataPipelineOrchestrator, "_update_universe_liquidity", no_op)
    monkeypatch.setattr(DataPipelineOrchestrator, "_activate_refreshed_stocks", no_op)
    monkeypatch.setattr(DataPipelineOrchestrator, "_complete_refresh_queue", no_op)

    result = asyncio.run(fetch_and_cache_eod_prices(" new "))
    assert result["status"] == "cached"
    assert result["prices_upserted"] == 2
    assert writes[0][0] == "prices"
    assert writes[0][2] == "ticker,date"
    assert writes[0][1][-1]["log_return"] == pytest.approx(np.log(101 / 100))


def test_pipeline_fails_status_when_required_price_provider_chain_is_exhausted():
    async def unavailable(_ticker):
        raise ProviderError("fixture providers unavailable")
    service = DataPipelineOrchestrator(
        db_client=FakeSupabase(), ticker_list=["ABC"], fred_series={},
        provider_overrides={"yfinance": unavailable},
    )
    report = asyncio.run(service.run(include_qualitative=False))
    assert report["ok"] is False
    assert report["price_status"]["ABC"]["ok"] is False
