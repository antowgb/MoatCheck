"""Multi-provider market and macro data pipeline.

This is an application-layer service: it owns network access and Supabase
upserts. The numerical core under ``app.core`` remains pure. Provider calls
are injectable, allowing the pipeline to be tested entirely offline.

Prices prefer Yahoo Finance, then Tiingo, FMP and (for crypto pairs) Binance.
FRED supplies macro observations. FMP/Tiingo provide company ratios. SEC/RSS
news and filings use the existing qualitative collector when enabled.
"""

from __future__ import annotations

import asyncio
import json
import inspect
import logging
import math
import os
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

import httpx
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FRED_SERIES = {
    "DGS10": "us_treasury_10y_yield",
    "DGS2": "us_treasury_2y_yield",
    "FEDFUNDS": "us_federal_funds_rate",
    "CPIAUCSL": "us_consumer_price_index",
}
PRICE_CONFLICT = "ticker,date"
MACRO_CONFLICT = "series_id,observation_date,provider"
FUNDAMENTAL_CONFLICT = "ticker,as_of_date,provider"


class ProviderError(RuntimeError):
    """Base error for unavailable, malformed, or rejected provider responses."""


class ProviderRateLimitError(ProviderError):
    """Raised when the provider or our local quota limiter rejects a call."""


@dataclass(frozen=True)
class ProviderAttempt:
    """Small audit record for a provider call, without request secrets."""

    provider: str
    succeeded: bool
    error: str | None = None


class RateLimiter:
    """Async provider quota and concurrency guard.

    Provider limits reset in the provider's configured timezone and reset hour.
    Hourly caps and optional spacing prevent burst traffic. Counts are
    intentionally process-local; the scheduled GitHub job runs one process.
    """

    def __init__(
        self,
        daily_limits: dict[str, int] | None = None,
        hourly_limits: dict[str, int] | None = None,
        min_interval_seconds: dict[str, float] | None = None,
        provider_timezones: dict[str, str] | None = None,
        daily_reset_hours: dict[str, int] | None = None,
        max_concurrent: int = 4,
        *,
        now: Callable[[], datetime] | None = None,
        sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
    ) -> None:
        self.daily_limits = {k.lower(): v for k, v in (daily_limits or {}).items()}
        self.hourly_limits = {k.lower(): v for k, v in (hourly_limits or {}).items()}
        self.provider_timezones = {k.lower(): ZoneInfo(v) for k, v in (provider_timezones or {}).items()}
        self.daily_reset_hours = {k.lower(): int(v) for k, v in (daily_reset_hours or {}).items()}
        self.min_interval_seconds = {
            k.lower(): float(v) for k, v in (min_interval_seconds or {}).items()
        }
        self._counts: dict[str, int] = {}
        self._day_keys: dict[str, date] = {}
        self._hour_keys: dict[str, tuple[date, int]] = {}
        self._daily_counts: dict[str, int] = {}
        self._hourly_counts: dict[str, int] = {}
        self._last_call: dict[str, float] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._now = now or (lambda: datetime.now(timezone.utc))
        self._sleep = sleep

    async def acquire(self, provider: str) -> None:
        """Reserve one request slot, enforcing daily quota and spacing."""
        name = provider.lower()
        lock = self._locks.setdefault(name, asyncio.Lock())
        async with self._semaphore, lock:
            local_now = self._now().astimezone(self.provider_timezones.get(name, timezone.utc))
            reset_hour = self.daily_reset_hours.get(name, 0)
            day_key = local_now.date() if local_now.hour >= reset_hour else local_now.date() - timedelta(days=1)
            hour_key = (local_now.date(), local_now.hour)
            if self._day_keys.get(name) != day_key:
                self._daily_counts[name] = 0
                self._day_keys[name] = day_key
            if self._hour_keys.get(name) != hour_key:
                self._hourly_counts[name] = 0
                self._hour_keys[name] = hour_key
            limit = self.daily_limits.get(name)
            if limit is not None and self._daily_counts.get(name, 0) >= limit:
                raise ProviderRateLimitError(f"{name} local daily quota exhausted")
            hourly_limit = self.hourly_limits.get(name)
            if hourly_limit is not None and self._hourly_counts.get(name, 0) >= hourly_limit:
                raise ProviderRateLimitError(f"{name} local hourly quota exhausted")
            interval = self.min_interval_seconds.get(name, 0.0)
            elapsed = time.monotonic() - self._last_call.get(name, 0.0)
            if interval and elapsed < interval:
                await self._sleep(interval - elapsed)
            self._last_call[name] = time.monotonic()
            self._counts[name] = self._counts.get(name, 0) + 1
            self._daily_counts[name] = self._daily_counts.get(name, 0) + 1
            self._hourly_counts[name] = self._hourly_counts.get(name, 0) + 1

    def usage(self) -> dict[str, int]:
        """Return current per-provider request counts for telemetry."""
        return dict(self._counts)


class FallbackHandler:
    """Try providers in priority order and continue on failure or quota limits."""

    async def run(
        self,
        providers: Sequence[tuple[str, Callable[[], Awaitable[Any]]]],
    ) -> tuple[Any, str, list[ProviderAttempt]]:
        """Return ``(payload, provider, attempts)`` from the first valid source."""
        attempts: list[ProviderAttempt] = []
        for name, call in providers:
            try:
                result = await call()
                if result is None or (isinstance(result, (pd.DataFrame, list, dict)) and len(result) == 0):
                    raise ProviderError(f"{name} returned no usable data")
                attempts.append(ProviderAttempt(name, True))
                return result, name, attempts
            except Exception as exc:
                attempts.append(ProviderAttempt(name, False, type(exc).__name__))
                logger.warning("Provider %s failed; trying next source (%s)", name, type(exc).__name__)
        raise ProviderError("All configured data providers failed: " + ", ".join(a.provider for a in attempts))


def _as_float(value: Any) -> float | None:
    try:
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    except (TypeError, ValueError):
        return None


def _json_safe(value: Any) -> Any:
    """Recursively convert pandas/numpy objects and non-finite values for JSONB."""
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert pandas rows to PostgREST-safe Python scalars and JSON nulls."""
    records: list[dict[str, Any]] = []
    for row in frame.astype(object).to_dict("records"):
        records.append(_json_safe(row))
    return records


def normalize_prices(data: Any, ticker: str, provider: str) -> pd.DataFrame:
    """Normalize provider bars to adjusted OHLCV without fabricating rows.

    Yahoo's adjusted-close factor is applied to OHLC and volume is retained as
    reported. Missing observations are forward-filled only in the derived
    aligned analysis panel, never in rows persisted as market facts.
    """
    if isinstance(data, pd.DataFrame):
        frame = data.copy()
        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = frame.columns.get_level_values(0)
        has_date = any(str(c).strip().lower() in {"date", "datetime", "time"} for c in frame.columns)
        if not has_date:
            frame.index.name = "date"
            frame = frame.reset_index()
    else:
        frame = pd.DataFrame(data)
    if frame.empty:
        return pd.DataFrame(columns=["ticker", "date", "open", "high", "low", "close", "adjusted_close", "volume", "provider"])
    aliases = {str(c).strip().lower().replace(" ", "_"): c for c in frame.columns}
    date_col = next((aliases[k] for k in ("date", "datetime", "time") if k in aliases), None)
    if date_col is None:
        date_col = frame.columns[0]
    def col(*names: str) -> pd.Series:
        key = next((aliases[n] for n in names if n in aliases), None)
        return frame[key] if key is not None else pd.Series(np.nan, index=frame.index)
    raw_close = pd.to_numeric(col("close", "price"), errors="coerce")
    adj_close = pd.to_numeric(col("adj_close", "adjusted_close", "adjclose"), errors="coerce")
    adj_close = adj_close.where(adj_close.notna(), raw_close)
    factor = (adj_close / raw_close.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    result = pd.DataFrame({
        "ticker": ticker.upper().strip(),
        "date": pd.to_datetime(frame[date_col], utc=True, errors="coerce").dt.date.astype("string"),
        "open": pd.to_numeric(col("open"), errors="coerce") * factor,
        "high": pd.to_numeric(col("high"), errors="coerce") * factor,
        "low": pd.to_numeric(col("low"), errors="coerce") * factor,
        "close": adj_close,
        "adjusted_close": adj_close,
        "volume": pd.to_numeric(col("volume"), errors="coerce"),
        "provider": provider,
    })
    result = result.dropna(subset=["date", "close"])
    result = result[np.isfinite(result["close"]) & (result["close"] > 0)]
    result["volume"] = result["volume"].where(result["volume"] >= 0)
    result["volume"] = result["volume"].round().astype("Int64")
    return result.sort_values("date").drop_duplicates(["ticker", "date"], keep="last").reset_index(drop=True)


def derive_market_analytics(
    prices: pd.DataFrame, *, covariance_window: int = 60, adv_window: int = 20
) -> dict[str, Any]:
    """Build business-day aligned analysis, log returns, rolling covariance and ADV.

    Calendar alignment and forward-fill are limited to analysis calculations.
    No synthetic quote is returned for database persistence. ADV is the rolling
    mean of adjusted close times reported volume, in quote currency per day.
    """
    if prices.empty:
        return {"aligned_closes": pd.DataFrame(), "log_returns": pd.DataFrame(), "rolling_covariance": pd.DataFrame(), "adv": {}}
    work = prices.copy()
    work["date"] = pd.to_datetime(work["date"])
    close = work.pivot(index="date", columns="ticker", values="adjusted_close").sort_index()
    volume = work.pivot(index="date", columns="ticker", values="volume").sort_index()
    calendar = pd.bdate_range(close.index.min(), close.index.max())
    aligned = close.reindex(calendar).ffill()
    log_returns = np.log(aligned.where(aligned > 0)).diff().replace([np.inf, -np.inf], np.nan)
    dollar_volume = close.reindex(calendar).ffill() * volume.reindex(calendar).fillna(0)
    adv_frame = dollar_volume.rolling(adv_window, min_periods=1).mean()
    adv = {str(k): _as_float(v) for k, v in adv_frame.iloc[-1].items()}
    rolling_covariance = log_returns.tail(max(covariance_window, 2)).cov(min_periods=2)
    return {
        "aligned_closes": aligned,
        "log_returns": log_returns,
        "rolling_covariance": rolling_covariance,
        "adv": adv,
        "calendar_forward_filled_rows": int(aligned.shape[0] - close.shape[0]),
    }


class DataPipelineOrchestrator:
    """Coordinate provider retrieval, normalization, analytics, and idempotent writes.

    ``http_client``, ``db_client``, provider functions and the qualitative
    runner are injectable. The default ``run`` may access the network and
    Supabase; tests provide fakes for both and do not make live requests.
    """

    def __init__(
        self,
        *,
        http_client: Any | None = None,
        db_client: Any | None = None,
        rate_limiter: RateLimiter | None = None,
        fallback_handler: FallbackHandler | None = None,
        provider_overrides: dict[str, Callable[..., Awaitable[Any]]] | None = None,
        qualitative_runner: Callable[[], Any] | None = None,
        ticker_list: Sequence[str] | None = None,
        fred_series: dict[str, str] | None = None,
    ) -> None:
        self.http_client = http_client
        self.db_client = db_client
        self.rate_limiter = rate_limiter or RateLimiter(
            daily_limits={
                "fmp": int(os.getenv("FMP_DAILY_LIMIT", "250")),
                "tiingo": int(os.getenv("TIINGO_DAILY_LIMIT", "1000")),
                # Conservative local circuit breaker, not a statement of
                # Yahoo's unpublished contractual quota.
                "yfinance": int(os.getenv("YFINANCE_DAILY_LIMIT", "1800")),
            },
            hourly_limits={"tiingo": int(os.getenv("TIINGO_HOURLY_LIMIT", "50"))},
            # Etc/GMT+5 is fixed UTC-5 (EST), which matches the documented
            # reset clocks instead of shifting with daylight-saving time.
            provider_timezones={"fmp": "Etc/GMT+5", "tiingo": "Etc/GMT+5"},
            daily_reset_hours={"fmp": 15, "tiingo": 0},
            min_interval_seconds={"fred": 1.0, "tiingo": 0.1, "binance": 0.25, "sec": 0.12, "yfinance": 0.2},
        )
        self.fallback_handler = fallback_handler or FallbackHandler()
        self.provider_overrides = provider_overrides or {}
        self.qualitative_runner = qualitative_runner
        configured_tickers = ticker_list
        if configured_tickers is None:
            configured_tickers = os.getenv("PIPELINE_TICKERS", "").split(",")
        self.ticker_list = [str(t).upper().strip() for t in configured_tickers if str(t).strip()]
        self.fred_series = FRED_SERIES if fred_series is None else fred_series
        self.attempt_log: list[ProviderAttempt] = []
        self._price_start_cache: dict[str, str] = {}
        self.last_refresh_cohorts: dict[str, list[str]] = {"high": [], "normal": []}

    async def _price_start(self, ticker: str) -> str:
        """Use a short overlap for incremental ingestion, or a five-year first load."""
        ticker = ticker.upper()
        if ticker in self._price_start_cache:
            return self._price_start_cache[ticker]
        start = (date.today() - timedelta(days=365 * 5)).isoformat()
        try:
            client = await self._db()
            query = (client.table("prices").select("date").eq("ticker", ticker)
                     .order("date", desc=True).limit(1))
            result = query.execute()
            if hasattr(result, "__await__"):
                result = await result
            if result.data:
                latest = date.fromisoformat(str(result.data[0]["date"])[:10])
                # Enough history for a 60-session covariance window plus
                # trading holidays and intermittent provider gaps.
                start = (latest - timedelta(days=120)).isoformat()
        except Exception:
            # Missing table/migration or transient DB read: use a bounded
            # backfill. The eventual upsert remains idempotent.
            logger.warning("Could not read latest stored price date for %s; using five-year backfill", ticker)
        self._price_start_cache[ticker] = start
        return start

    async def _get_json(self, provider: str, url: str, *, params: dict[str, Any] | None = None,
                        headers: dict[str, str] | None = None) -> Any:
        await self.rate_limiter.acquire(provider)
        if self.http_client is not None:
            response = await self.http_client.get(url, params=params, headers=headers, timeout=30)
        else:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(url, params=params, headers=headers)
        status = getattr(response, "status_code", 200)
        if status == 429:
            raise ProviderRateLimitError(f"{provider} HTTP 429")
        if status >= 400:
            raise ProviderError(f"{provider} HTTP {status}")
        payload = response.json()
        if isinstance(payload, dict) and (payload.get("Error Message") or payload.get("error") or payload.get("Note")):
            message = str(payload.get("Error Message") or payload.get("error") or payload.get("Note"))
            if "limit" in message.lower() or "quota" in message.lower():
                raise ProviderRateLimitError(f"{provider} quota response")
            raise ProviderError(f"{provider} rejected request")
        return payload

    async def _yfinance_prices(self, ticker: str) -> Any:
        override = self.provider_overrides.get("yfinance")
        if override:
            return await override(ticker)
        await self.rate_limiter.acquire("yfinance")
        import yfinance as yf
        # This is a single-process scheduled CLI, not a request-serving event
        # loop. Keep the blocking SDK call inline rather than relying on a
        # worker-thread handoff that is unstable in some Python 3.14 runtimes.
        return yf.download(ticker, start=await self._price_start(ticker), interval="1d", auto_adjust=False, progress=False, threads=False)

    async def _tiingo_prices(self, ticker: str) -> Any:
        override = self.provider_overrides.get("tiingo_prices")
        if override:
            return await override(ticker)
        token = os.getenv("TIINGO_API_KEY")
        if not token:
            raise ProviderError("TIINGO_API_KEY not configured")
        if os.getenv("TIINGO_PERSISTENCE_ALLOWED", "false").lower() != "true":
            raise ProviderError("Tiingo persistence is disabled until data-retention rights are confirmed")
        return await self._get_json("tiingo", f"https://api.tiingo.com/tiingo/daily/{ticker}/prices",
                                    params={"token": token, "startDate": await self._price_start(ticker)})

    async def _fmp_prices(self, ticker: str) -> Any:
        override = self.provider_overrides.get("fmp_prices")
        if override:
            return await override(ticker)
        key = os.getenv("FMP_API_KEY")
        if not key:
            raise ProviderError("FMP_API_KEY not configured")
        if os.getenv("FMP_DATA_LICENSE_CONFIRMED", "false").lower() != "true":
            raise ProviderError("FMP persistence is disabled until display/redistribution rights are confirmed")
        return await self._get_json("fmp", "https://financialmodelingprep.com/stable/historical-price-eod/full",
                                    params={"symbol": ticker, "from": await self._price_start(ticker), "apikey": key})

    async def _binance_prices(self, ticker: str) -> Any:
        override = self.provider_overrides.get("binance_prices")
        if override:
            return await override(ticker)
        normalized_ticker = ticker.upper()
        symbol = (normalized_ticker[:-4] + "USDT" if normalized_ticker.endswith("-USD")
                  else normalized_ticker.replace("-", ""))
        if not symbol.endswith(("USD", "USDT", "BTC", "ETH")):
            raise ProviderError("not a supported Binance quote pair")
        return await self._get_json("binance", "https://api.binance.com/api/v3/klines",
                                    params={"symbol": symbol, "interval": "1d", "limit": 1000})

    async def fetch_prices(self, ticker: str) -> tuple[pd.DataFrame, str]:
        """Fetch daily bars with fallback and normalize all provider schemas."""
        candidates = [
            ("yfinance", lambda: self._yfinance_prices(ticker)),
            ("tiingo", lambda: self._tiingo_prices(ticker)),
            ("fmp", lambda: self._fmp_prices(ticker)),
        ]
        if any(ticker.upper().endswith(s) for s in ("-USD", "USDT", "-BTC", "-ETH")):
            candidates.append(("binance", lambda: self._binance_prices(ticker)))
        raw, provider, attempts = await self.fallback_handler.run(candidates)
        self.attempt_log.extend(attempts)
        if provider == "binance":
            raw = pd.DataFrame(raw, columns=["date", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "trades", "taker_base", "taker_quote", "ignore"])
            raw["date"] = pd.to_datetime(raw["date"], unit="ms", utc=True)
            raw["adjusted_close"] = raw["close"]
        return normalize_prices(raw, ticker, provider), provider

    async def _fred_series(self, series_id: str, start_date: str) -> list[dict[str, Any]]:
        override = self.provider_overrides.get("fred")
        if override:
            return await override(series_id, start_date)
        key = os.getenv("FRED_API_KEY")
        if not key:
            raise ProviderError("FRED_API_KEY not configured")
        payload = await self._get_json("fred", "https://api.stlouisfed.org/fred/series/observations",
                                       params={"series_id": series_id, "api_key": key, "file_type": "json", "sort_order": "asc", "observation_start": start_date})
        return payload.get("observations", [])

    async def _fmp_macro_series(self, series_id: str, start_date: str) -> list[dict[str, Any]]:
        """Secondary macro source, normalized from FMP Treasury/economic endpoints."""
        override = self.provider_overrides.get("fmp_macro")
        if override:
            return await override(series_id, start_date)
        key = os.getenv("FMP_API_KEY")
        if not key or os.getenv("FMP_DATA_LICENSE_CONFIRMED", "false").lower() != "true":
            raise ProviderError("FMP macro fallback disabled or not licensed for persistence")
        today = date.today().isoformat()
        from_date = max(start_date, (date.today() - timedelta(days=89)).isoformat())
        if series_id in {"DGS2", "DGS10"}:
            payload = await self._get_json("fmp", "https://financialmodelingprep.com/stable/treasury-rates",
                                           params={"from": from_date, "to": today, "apikey": key})
            field = "year2" if series_id == "DGS2" else "year10"
        else:
            macro_name = {"FEDFUNDS": "federalFunds", "CPIAUCSL": "CPI"}.get(series_id)
            if not macro_name:
                raise ProviderError(f"no FMP macro mapping for {series_id}")
            payload = await self._get_json("fmp", "https://financialmodelingprep.com/stable/economic-indicators",
                                           params={"name": macro_name, "from": from_date, "to": today, "apikey": key})
            field = "value"
        rows = []
        for item in (payload if isinstance(payload, list) else []):
            value = item.get(field)
            if value is not None and item.get("date"):
                rows.append({"date": item["date"], "value": value})
        return rows

    async def _macro_start(self, series_id: str) -> str:
        """Find the last persisted observation and overlap it for revision safety."""
        configured = os.getenv("PIPELINE_MACRO_START_DATE", "2000-01-01")
        try:
            client = await self._db()
            query = (client.table("macro_indicators").select("observation_date")
                     .eq("series_id", series_id).eq("provider", "fred")
                     .order("observation_date", desc=True).limit(1))
            result = query.execute()
            if hasattr(result, "__await__"):
                result = await result
            if result.data:
                latest = date.fromisoformat(str(result.data[0]["observation_date"])[:10])
                return (latest - timedelta(days=14)).isoformat()
        except Exception:
            logger.warning("Could not read latest macro date for %s; using configured backfill", series_id)
        return configured

    async def fetch_macro(self) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Fetch configured FRED yield, inflation and policy-rate observations."""
        rows: list[dict[str, Any]] = []
        statuses: dict[str, Any] = {}
        for series_id, name in self.fred_series.items():
            try:
                start_date = await self._macro_start(series_id)
                try:
                    observations = await self._fred_series(series_id, start_date)
                    provider = "fred"
                    self.attempt_log.append(ProviderAttempt("fred", True))
                except Exception as primary_error:
                    self.attempt_log.append(ProviderAttempt("fred", False, type(primary_error).__name__))
                    observations = await self._fmp_macro_series(series_id, start_date)
                    provider = "fmp"
                    self.attempt_log.append(ProviderAttempt("fmp", True))
                for item in observations:
                    value = _as_float(item.get("value"))
                    if value is None or not item.get("date"):
                        continue
                    rows.append({"series_id": series_id, "name": name, "observation_date": str(item["date"]),
                                 "value": value, "provider": provider})
                statuses[series_id] = {"ok": bool(observations), "rows": len(observations)}
            except Exception as exc:
                statuses[series_id] = {"ok": False, "error": type(exc).__name__}
                logger.warning("FRED series %s unavailable (%s)", series_id, type(exc).__name__)
        return rows, statuses

    async def _fmp_fundamentals(self, ticker: str) -> Any:
        override = self.provider_overrides.get("fmp_fundamentals")
        if override:
            return await override(ticker)
        key = os.getenv("FMP_API_KEY")
        if not key:
            raise ProviderError("FMP_API_KEY not configured")
        if os.getenv("FMP_DATA_LICENSE_CONFIRMED", "false").lower() != "true":
            raise ProviderError("FMP persistence is disabled until display/redistribution rights are confirmed")
        return await self._get_json("fmp", "https://financialmodelingprep.com/stable/ratios",
                                    params={"symbol": ticker, "limit": 1, "apikey": key})

    async def _tiingo_fundamentals(self, ticker: str) -> Any:
        override = self.provider_overrides.get("tiingo_fundamentals")
        if override:
            return await override(ticker)
        token = os.getenv("TIINGO_API_KEY")
        if not token:
            raise ProviderError("TIINGO_API_KEY not configured")
        if os.getenv("TIINGO_PERSISTENCE_ALLOWED", "false").lower() != "true":
            raise ProviderError("Tiingo persistence is disabled until data-retention rights are confirmed")
        return await self._get_json("tiingo", f"https://api.tiingo.com/tiingo/fundamentals/{ticker}/daily",
                                    params={"token": token, "limit": 1})

    async def fetch_fundamentals(self, ticker: str) -> list[dict[str, Any]]:
        """Fetch one latest company-ratios snapshot, preferring FMP then Tiingo."""
        try:
            payload, provider, attempts = await self.fallback_handler.run([
                ("fmp", lambda: self._fmp_fundamentals(ticker)),
                ("tiingo", lambda: self._tiingo_fundamentals(ticker)),
            ])
        except ProviderError:
            return []
        self.attempt_log.extend(attempts)
        snapshot = payload[0] if isinstance(payload, list) and payload else payload
        if not isinstance(snapshot, dict):
            return []
        as_of = str(snapshot.get("date") or snapshot.get("asOfDate") or datetime.now(timezone.utc).date().isoformat())[:10]
        return [{"ticker": ticker.upper(), "as_of_date": as_of, "provider": provider,
                 "metrics": _json_safe(snapshot)}]

    async def fetch_qualitative(self) -> dict[str, Any]:
        """Run existing SEC, IR and press RSS collectors when configured."""
        runner = self.qualitative_runner
        if runner is None:
            from app.qualitative.run import run_qualitative_refresh
            runner = run_qualitative_refresh
        if inspect.iscoroutinefunction(runner):
            return await runner()
        # Existing SEC/RSS collectors are synchronous, bounded CLI work. The
        # pipeline does not serve HTTP requests, so invoke them directly.
        # Their sync Supabase client expects the legacy service key alias.
        from app.data.supabase_client import get_supabase

        prior_public_key = os.environ.get("SUPABASE_KEY")
        if os.getenv("SUPABASE_SERVICE_ROLE_KEY"):
            os.environ["SUPABASE_KEY"] = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
        elif os.getenv("SUPABASE_SECRET_KEY", "").startswith("sb_secret_"):
            os.environ.pop("SUPABASE_KEY", None)
        # Ensure an earlier read-only client cannot be reused for service-role
        # writes; clear again so a caller cannot keep the elevated client.
        get_supabase.cache_clear()
        try:
            result = runner()
        finally:
            get_supabase.cache_clear()
            if prior_public_key is None:
                os.environ.pop("SUPABASE_KEY", None)
            else:
                os.environ["SUPABASE_KEY"] = prior_public_key
        if inspect.isawaitable(result):
            return await result
        return result

    async def _db(self) -> Any:
        if self.db_client is not None:
            return self.db_client
        from app.db.supabase import get_async_supabase_client
        return await get_async_supabase_client()

    async def _upsert(self, table: str, rows: list[dict[str, Any]], conflict: str) -> int:
        if not rows:
            return 0
        client = await self._db()
        count = 0
        for start in range(0, len(rows), 500):
            query = client.table(table).upsert(rows[start:start + 500], on_conflict=conflict)
            result = query.execute()
            if hasattr(result, "__await__"):
                result = await result
            count += len(rows[start:start + 500])
        return count

    async def _update_universe_liquidity(self, adv_by_ticker: dict[str, Any]) -> int:
        """Persist USD ADV checks without mislabeling foreign-currency volume."""
        if not adv_by_ticker:
            return 0
        client = await self._db()
        usd_tickers: set[str] = set()
        tickers = list(adv_by_ticker)
        for start in range(0, len(tickers), 500):
            result = client.table("stocks").select("ticker,currency").in_("ticker", tickers[start:start + 500]).execute()
            if hasattr(result, "__await__"):
                result = await result
            usd_tickers.update(
                str(row["ticker"]).upper() for row in result.data or []
                if str(row.get("currency") or "").upper() == "USD"
            )
        payload = {
            ticker: float(value) for ticker, value in adv_by_ticker.items()
            if ticker.upper() in usd_tickers and value is not None and math.isfinite(float(value)) and float(value) >= 0
        }
        if not payload:
            return 0
        result = client.rpc("update_market_universe_liquidity", {"p_adv_by_ticker": payload}).execute()
        if hasattr(result, "__await__"):
            result = await result
        return int(result.data or 0)

    async def _complete_refresh_queue(self, successful_tickers: Sequence[str]) -> None:
        """Remove successfully priced symbols from the pending ingestion queue."""
        if not successful_tickers:
            return
        client = await self._db()
        query = (client.table("refresh_queue")
                 .update({"status": "done", "processed_at": datetime.now(timezone.utc).isoformat()})
                 .eq("status", "pending")
                 .in_("ticker", list(successful_tickers)))
        result = query.execute()
        if hasattr(result, "__await__"):
            await result

    async def _activate_refreshed_stocks(self, successful_tickers: Sequence[str]) -> None:
        """Expose newly discovered symbols after their first price load succeeds."""
        if not successful_tickers:
            return
        client = await self._db()
        for start in range(0, len(successful_tickers), 500):
            result = (client.table("stocks").update({"status": "active"})
                      .in_("ticker", list(successful_tickers[start:start + 500])).execute())
            if hasattr(result, "__await__"):
                await result

    async def load_tickers(self) -> list[str]:
        """Select the scheduled daily cohort without exceeding provider budgets.

        An explicit ``PIPELINE_TICKERS`` list remains an operator override.
        Otherwise, core holdings and user portfolios run daily while the
        extended universe rotates through one weekday batch in three.
        """
        if self.ticker_list:
            self.last_refresh_cohorts = {"high": list(dict.fromkeys(self.ticker_list)), "normal": []}
            return list(dict.fromkeys(self.ticker_list))
        client = await self._db()
        memberships: list[dict[str, Any]] = []
        try:
            offset = 0
            while True:
                query = (client.table("market_universe_memberships")
                         .select("ticker,tier,rotation_batch,adv_usd,market_cap_usd,memberships,asset_type")
                         .order("ticker").range(offset, offset + 999))
                result = query.execute()
                if hasattr(result, "__await__"):
                    result = await result
                page = result.data or []
                memberships.extend(page)
                if len(page) < 1000:
                    break
                offset += 1000
        except Exception:
            # Keep the scheduler useful during deployment ordering. Never
            # fall back to an unbounded all-stock refresh.
            logger.exception("Universe membership table unavailable; using capped active stocks")
            query = (client.table("stocks").select("ticker").eq("status", "active")
                     .order("ticker").limit(int(os.getenv("UNIVERSE_CORE_LIMIT", "500"))))
            result = query.execute()
            if hasattr(result, "__await__"):
                result = await result
            tickers = sorted({str(row["ticker"]).upper() for row in (result.data or []) if row.get("ticker")})
            self.last_refresh_cohorts = {"high": tickers, "normal": []}
            return tickers

        from app.services.universe_manager import load_portfolio_tickers

        portfolio_tickers = await load_portfolio_tickers(client)

        queued_tickers: set[str] = set()
        try:
            result = (client.table("refresh_queue").select("ticker")
                      .eq("status", "pending").order("created_at").limit(1000).execute())
            if hasattr(result, "__await__"):
                result = await result
            queued_tickers.update(str(row["ticker"]).upper() for row in result.data or [])
        except Exception:
            logger.info("Pending refresh queue unavailable; continuing without queue supplement")

        from app.services.universe_manager import select_refresh_cohorts

        high_priority, normal_priority = select_refresh_cohorts(
            memberships,
            portfolio_tickers=sorted(portfolio_tickers),
            queued_tickers=sorted(queued_tickers),
            core_limit=int(os.getenv("UNIVERSE_CORE_LIMIT", "500")),
            rotation_days=int(os.getenv("UNIVERSE_ROTATION_DAYS", "3")),
            minimum_adv_usd=float(os.getenv("MIN_ADV_USD", "200000")),
        )
        self.last_refresh_cohorts = {"high": high_priority, "normal": normal_priority}
        return high_priority + normal_priority

    async def run(self, *, include_qualitative: bool = True) -> dict[str, Any]:
        """Execute one incremental pipeline run and return safe operational telemetry."""
        started = time.perf_counter()
        tickers = await self.load_tickers()
        all_prices: list[pd.DataFrame] = []
        fundamentals: list[dict[str, Any]] = []
        price_status: dict[str, Any] = {}
        for ticker in tickers:
            try:
                frame, provider = await self.fetch_prices(ticker)
                all_prices.append(frame)
                price_status[ticker] = {"ok": not frame.empty, "provider": provider, "rows": len(frame)}
            except Exception as exc:
                price_status[ticker] = {"ok": False, "error": type(exc).__name__}
                logger.error("Price retrieval failed for %s", ticker, exc_info=True)
            try:
                fundamentals.extend(await self.fetch_fundamentals(ticker))
            except Exception as exc:
                logger.warning("Fundamentals unavailable for %s (%s)", ticker, type(exc).__name__)
        successfully_priced = [ticker for ticker, status in price_status.items() if status.get("ok")]
        try:
            await self._activate_refreshed_stocks(successfully_priced)
        except Exception:
            logger.warning("Could not activate successfully refreshed symbols", exc_info=True)
        try:
            await self._complete_refresh_queue(successfully_priced)
        except Exception:
            logger.warning("Could not mark successful refresh-queue items complete", exc_info=True)
        price_frame = pd.concat(all_prices, ignore_index=True) if all_prices else pd.DataFrame()
        analytics = derive_market_analytics(price_frame)
        price_rows = _json_records(price_frame) if not price_frame.empty else []
        # Add point-in-time derived fields to observed prices only.
        if price_rows:
            log_returns = analytics["log_returns"]
            adv = analytics["adv"]
            for row in price_rows:
                day = pd.Timestamp(row["date"])
                row["log_return"] = _as_float(log_returns.at[day, row["ticker"]]) if day in log_returns.index else None
                row["adv"] = adv.get(row["ticker"])
                row["provider"] = row.get("provider") or "unknown"
        price_count = await self._upsert("prices", price_rows, PRICE_CONFLICT)
        try:
            await self._update_universe_liquidity(analytics.get("adv", {}))
        except Exception:
            # Prices are still safely persisted; missing this derived metadata
            # keeps symbols unverified rather than falsely liquid.
            logger.warning("Could not refresh USD universe liquidity flags", exc_info=True)
        macro_rows, macro_status = await self.fetch_macro()
        macro_count = await self._upsert("macro_indicators", macro_rows, MACRO_CONFLICT)
        fundamentals_count = await self._upsert("company_fundamentals", fundamentals, FUNDAMENTAL_CONFLICT)
        qualitative: dict[str, Any] = {"skipped": True}
        if include_qualitative:
            try:
                qualitative = await self.fetch_qualitative()
            except Exception as exc:
                qualitative = {"ok": False, "error": type(exc).__name__}
                logger.exception("Qualitative collector failed")
        prices_ok = all(value.get("ok", False) for value in price_status.values()) if tickers else True
        macro_ok = all(value.get("ok", False) for value in macro_status.values()) if macro_status else True
        qualitative_ok = qualitative.get("ok", True) is not False
        return {
            "ok": prices_ok and macro_ok and qualitative_ok,
            "snapshot_at": datetime.now(timezone.utc).isoformat(),
            "tickers_requested": len(tickers),
            "refresh_cohorts": {
                "high_priority": len(self.last_refresh_cohorts["high"]),
                "normal_priority": len(self.last_refresh_cohorts["normal"]),
            },
            "prices_upserted": price_count,
            "macro_indicators_upserted": macro_count,
            "company_fundamentals_upserted": fundamentals_count,
            "price_status": price_status,
            "macro_status": macro_status,
            "qualitative": qualitative,
            "analytics": {
                "input_shape": list(price_frame.shape),
                "log_returns_shape": list(analytics["log_returns"].shape),
                "rolling_covariance": _json_safe(analytics["rolling_covariance"].to_dict()),
                "adv": analytics["adv"],
                "calendar_forward_filled_rows": analytics.get("calendar_forward_filled_rows", 0),
            },
            "provider_attempts": [a.__dict__ for a in self.attempt_log],
            "provider_request_counts": self.rate_limiter.usage(),
            "execution_ms": round((time.perf_counter() - started) * 1000, 3),
        }


_ON_DEMAND_RATE_LIMITER = RateLimiter(
    daily_limits={
        "fmp": int(os.getenv("FMP_DAILY_LIMIT", "250")),
        "tiingo": int(os.getenv("TIINGO_DAILY_LIMIT", "1000")),
        # On-demand requests share a conservative per-process guard. This is
        # separate from the scheduled runner's quota accounting.
        "yfinance": int(os.getenv("ON_DEMAND_YFINANCE_DAILY_LIMIT", "120")),
    },
    hourly_limits={"tiingo": int(os.getenv("TIINGO_HOURLY_LIMIT", "50"))},
    provider_timezones={"fmp": "Etc/GMT+5", "tiingo": "Etc/GMT+5"},
    daily_reset_hours={"fmp": 15, "tiingo": 0},
    min_interval_seconds={"fred": 1.0, "tiingo": 0.1, "binance": 0.25, "yfinance": 0.2},
)


async def fetch_and_cache_eod_prices(ticker: str) -> dict[str, Any]:
    """Fetch one symbol's latest EOD bars and immediately upsert them to ``prices``.

    This narrow path is used when a symbol is searched or newly added to a
    saved portfolio. It uses the same provider fallback, normalization,
    analytics and ``ticker,date`` upsert contract as the daily pipeline, but
    deliberately skips fundamentals and macro jobs. Failed requests leave the
    high-priority refresh-queue item pending for the next scheduled attempt.
    Returns row count, selected provider and latest cached date. Complexity is
    ``O(T + W)`` for T returned bars and W rolling analytics observations.
    """
    normalized = str(ticker).strip().upper()
    if not normalized:
        raise ValueError("ticker must not be empty")
    started = time.perf_counter()
    pipeline = DataPipelineOrchestrator(
        ticker_list=[normalized],
        rate_limiter=_ON_DEMAND_RATE_LIMITER,
        fred_series={},
        qualitative_runner=lambda: {"skipped": True},
    )
    frame, provider = await pipeline.fetch_prices(normalized)
    if frame.empty:
        raise ProviderError(f"No end-of-day price observations are available for {normalized}")

    analytics = derive_market_analytics(frame)
    rows = _json_records(frame)
    log_returns = analytics["log_returns"]
    adv = analytics["adv"].get(normalized)
    for row in rows:
        day = pd.Timestamp(row["date"])
        row["log_return"] = _as_float(log_returns.at[day, normalized]) if day in log_returns.index else None
        row["adv"] = adv
        row["provider"] = row.get("provider") or provider
    await pipeline._upsert("prices", rows, PRICE_CONFLICT)
    try:
        await pipeline._update_universe_liquidity({normalized: adv} if adv is not None else {})
    except Exception:
        logger.info("Liquidity metadata update skipped for on-demand %s", normalized, exc_info=True)
    try:
        await pipeline._activate_refreshed_stocks([normalized])
        await pipeline._complete_refresh_queue([normalized])
    except Exception:
        logger.info("On-demand price rows cached but queue/stock status update failed for %s", normalized, exc_info=True)
    return {
        "ticker": normalized,
        "provider": provider,
        "prices_upserted": len(rows),
        "latest_price_date": str(pd.Timestamp(frame["date"].max()).date()),
        "status": "cached",
        "execution_ms": round((time.perf_counter() - started) * 1000.0, 3),
    }


async def run_pipeline() -> dict[str, Any]:
    """Convenience async entrypoint used by scheduled execution and tests."""
    return await DataPipelineOrchestrator().run()


def main() -> int:
    """CLI entrypoint; process exits non-zero when no price source succeeded."""
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    report = asyncio.run(run_pipeline())
    logger.info("Data pipeline complete: %s", json.dumps({k: v for k, v in report.items() if k != "analytics"}, default=str))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
