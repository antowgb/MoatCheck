"""Offline tests for constituent normalization, rotation, and queue safety."""

import asyncio
from datetime import date
from types import SimpleNamespace

import pandas as pd
import pytest

from app.services.universe_manager import (
    IndexSource,
    MAJOR_ETFS,
    UniverseManager,
    is_liquid_enough,
    load_portfolio_tickers,
    normalize_ticker,
    select_refresh_cohorts,
    select_refresh_tickers,
    stable_rotation_batch,
    trading_day_slot,
)


def test_normalize_ticker_converts_share_class_and_rejects_bad_input():
    assert normalize_ticker(" brk.b ") == "BRK-B"
    assert normalize_ticker("AAPL") == "AAPL"
    assert normalize_ticker("not a ticker!!!") is None


def test_weekday_rotation_skips_weekends_and_is_stable():
    friday = date(2026, 10, 16)
    monday = date(2026, 10, 19)
    assert trading_day_slot(monday) == (trading_day_slot(friday) + 1) % 3
    assert stable_rotation_batch("MSFT") == stable_rotation_batch("msft")


def test_universe_refresh_includes_core_portfolios_queue_and_one_extended_batch():
    slot = trading_day_slot(date(2026, 10, 19))
    rows = [
        {"ticker": "AAA", "tier": "core", "adv_usd": 900_000},
        {"ticker": "BBB", "tier": "core", "adv_usd": 150_000},
        {"ticker": "CCC", "tier": "extended", "rotation_batch": slot, "adv_usd": None},
        {"ticker": "DDD", "tier": "extended", "rotation_batch": (slot + 1) % 3},
    ]
    assert select_refresh_tickers(
        rows, portfolio_tickers=["BBB"], queued_tickers=["EEE"], run_date=date(2026, 10, 19)
    ) == ["AAA", "BBB", "EEE", "CCC"]


def test_core_indexes_major_etfs_and_portfolios_are_high_priority_each_day():
    slot = trading_day_slot(date(2026, 10, 19))
    memberships = [
        {"ticker": "SPX", "tier": "extended", "memberships": ["sp500"], "rotation_batch": (slot + 1) % 3},
        {"ticker": "NDX", "tier": "extended", "memberships": ["nasdaq100"], "rotation_batch": (slot + 1) % 3},
        {"ticker": "SPY", "tier": "extended", "memberships": ["major_etf"], "rotation_batch": (slot + 1) % 3},
        {"ticker": "EXT", "tier": "extended", "memberships": ["stoxx600"], "rotation_batch": slot, "adv_usd": 300_000},
    ]
    high, normal = select_refresh_cohorts(
        memberships,
        portfolio_tickers=["HELD"],
        run_date=date(2026, 10, 19),
    )
    assert high == ["SPX", "NDX", "SPY", "HELD"]
    assert normal == ["EXT"]


def test_unknown_and_below_threshold_adv_are_not_verified_liquid():
    assert not is_liquid_enough(None)
    assert not is_liquid_enough(199_999.99)
    assert not is_liquid_enough(200_000)
    assert is_liquid_enough(200_000.01)


class FakeQuery:
    def __init__(self, database, table):
        self.database = database
        self.name = table
        self.payload = None
        self.filters = {}

    def select(self, *_args):
        return self

    def order(self, *_args, **_kwargs):
        return self

    def range(self, *_args):
        return self

    def in_(self, key, values):
        self.filters[key] = values
        return self

    def upsert(self, payload, **_kwargs):
        self.payload = payload
        self.database.upserts.append((self.name, payload))
        return self

    def execute(self):
        if self.name == "portfolio_holdings":
            return SimpleNamespace(data=self.database.holdings)
        if self.name == "portfolios":
            return SimpleNamespace(data=self.database.portfolios)
        if self.name in {"stocks", "market_universe_memberships"} and "ticker" in self.filters:
            return SimpleNamespace(data=[])
        return SimpleNamespace(data=[])


class FakeDatabase:
    def __init__(self):
        self.upserts = []
        self.rpcs = []
        self.holdings = []
        self.portfolios = []

    def table(self, name):
        return FakeQuery(self, name)

    def rpc(self, name, params):
        self.rpcs.append((name, params))
        return FakeQuery(self, "rpc")


def test_fetch_universe_merges_memberships_and_curated_etfs():
    source = IndexSource("small_index", "https://invalid.test", ("Symbol",), ("Name",), True)

    async def fetch(_source):
        return pd.DataFrame({"Symbol": ["abc", "BRK.B"], "Name": ["Alpha", "Berkshire"]})

    manager = UniverseManager(sources=[source], source_fetchers={"small_index": fetch})
    universe = asyncio.run(manager.fetch_universe())
    assert universe["ABC"]["memberships"] == ["small_index"]
    assert universe["BRK-B"]["name"] == "Berkshire"
    assert universe["SPY"]["asset_type"] == "etf"


def test_fetch_universe_fails_closed_when_all_index_sources_fail():
    source = IndexSource("broken", "https://invalid.test", ("Symbol",))

    async def fail(_source):
        raise RuntimeError("offline")

    manager = UniverseManager(sources=[source], source_fetchers={"broken": fail})
    with pytest.raises(RuntimeError, match="ETF-only"):
        asyncio.run(manager.fetch_universe())


def test_http_constituent_response_uses_mocked_html_parser():
    source = IndexSource("sp500", "https://example.test/index", ("Symbol",), ("Company",), True)

    class Response:
        text = "<table><tr><th>Symbol</th><th>Company</th></tr><tr><td>ABC</td><td>Alpha Corp</td></tr></table>"

        def raise_for_status(self):
            return None

    class HttpClient:
        async def get(self, *_args, **_kwargs):
            return Response()

    manager = UniverseManager(http_client=HttpClient(), sources=[source], request_spacing_seconds=0)
    rows = asyncio.run(manager.fetch_source(source))
    assert rows == [{"ticker": "ABC", "name": "Alpha Corp", "membership": "sp500", "core_candidate": True}]


def test_csv_source_skips_metadata_non_equity_rows_and_normalizes_share_classes():
    source = IndexSource("russell2000", "https://example.test/index.csv", ("Ticker",), ("Name",))

    class Response:
        text = (
            "\n".join(["metadata"] * 8)
            + '\nTicker,Name,Asset Class\n"MOG A",Moog Inc,Equity\nCASH_USD,Cash,Cash\n'
        )

        def raise_for_status(self):
            return None

    class HttpClient:
        async def get(self, *_args, **_kwargs):
            return Response()

    manager = UniverseManager(http_client=HttpClient(), sources=[source], request_spacing_seconds=0)
    rows = asyncio.run(manager.fetch_source(source))
    assert [row["ticker"] for row in rows] == ["MOG-A"]


def test_sync_upserts_memberships_and_uses_idempotent_queue_rpc():
    source = IndexSource("sp500", "https://invalid.test", ("Symbol",), (), True)

    async def fetch(_source):
        return pd.DataFrame({"Symbol": ["AAA", "BBB"]})

    db = FakeDatabase()
    manager = UniverseManager(sources=[source], source_fetchers={"sp500": fetch}, db_client=db)
    report = asyncio.run(manager.sync())
    assert report["tickers"] == 2 + len(MAJOR_ETFS)
    stocks_payload = next(payload for table, payload in db.upserts if table == "stocks")
    assert all(row["status"] == "pending_refresh" for row in stocks_payload)
    assert any(table == "market_universe_memberships" for table, _ in db.upserts)
    assert any(name == "prune_market_universe_memberships" for name, _ in db.rpcs)
    queued = set().union(*(set(params["p_tickers"]) for name, params in db.rpcs if name == "enqueue_refresh_tickers"))
    assert {"AAA", "BBB", "SPY", "QQQ"}.issubset(queued)


def test_sync_queues_live_configured_and_held_tickers_as_high_priority():
    source = IndexSource("sp500", "https://invalid.test", ("Symbol",), (), True)

    async def fetch(_source):
        return pd.DataFrame({"Symbol": ["AAA"]})

    db = FakeDatabase()
    db.portfolios = [{"tickers": ["SAVED", "SHARED"]}]
    db.holdings = [{"ticker": "HELD"}, {"ticker": "SHARED"}]
    manager = UniverseManager(sources=[source], source_fetchers={"sp500": fetch}, db_client=db)
    report = asyncio.run(manager.sync())

    queued = set().union(*(set(params["p_tickers"]) for name, params in db.rpcs if name == "enqueue_refresh_tickers"))
    assert {"SAVED", "HELD", "SHARED"}.issubset(queued)
    assert report["portfolio_tickers"] == 3
    assert report["high_priority_tickers"] >= 3


def test_portfolio_ticker_reader_unions_targets_and_actual_holdings():
    db = FakeDatabase()
    db.portfolios = [{"tickers": ["AAA", "shared"]}]
    db.holdings = [{"ticker": "held"}, {"ticker": "SHARED"}]
    assert asyncio.run(load_portfolio_tickers(db)) == {"AAA", "HELD", "SHARED"}


def test_on_demand_ingestion_requires_exact_match_and_keeps_unknown_liquidity_unverified():
    async def lookup(ticker):
        assert ticker == "NEWCO"
        return {"symbol": "NEWCO", "name": "New Company", "currency": "usd", "asset_type": "equity", "adv_usd": None}

    db = FakeDatabase()
    manager = UniverseManager(db_client=db, ticker_lookup=lookup)
    result = asyncio.run(manager.ingest_on_demand("newco"))
    assert result["status"] == "pending_refresh"
    assert result["liquidity_verified"] is False
    assert result["currency"] == "USD"
    assert any(name == "enqueue_refresh_tickers" for name, _ in db.rpcs)


def test_on_demand_ingestion_rejects_unverified_search_match():
    async def lookup(_ticker):
        return {"symbol": "OTHER", "name": "Other Company"}

    manager = UniverseManager(db_client=FakeDatabase(), ticker_lookup=lookup)
    with pytest.raises(LookupError, match="exact market symbol"):
        asyncio.run(manager.ingest_on_demand("NEWCO"))


def test_on_demand_ingestion_does_not_upsert_missing_currency_as_null():
    async def lookup(_ticker):
        return {"symbol": "NEWCO", "name": "New Company", "currency": None}

    db = FakeDatabase()
    manager = UniverseManager(db_client=db, ticker_lookup=lookup)
    result = asyncio.run(manager.ingest_on_demand("NEWCO"))
    stock_payload = next(payload for table, payload in db.upserts if table == "stocks")
    assert result["currency"] is None
    assert "currency" not in stock_payload
