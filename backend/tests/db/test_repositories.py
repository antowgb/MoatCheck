"""Repository tests using mocked Supabase async fluent queries."""

import asyncio
from datetime import date
from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.models.backtest_run import BacktestRun
from app.models.portfolio import PortfolioConfig
from app.models.stress_test import StressTest
from app.repositories.backtests import BacktestRepository
from app.repositories.portfolios import PortfolioRepository
from app.repositories.stress_tests import StressTestRepository
from app.repositories._base import RepositoryError


class FakeQuery:
    def __init__(self, response):
        self.response = response
        self.operation = None
        self.payload = None
        self.filters = []

    def upsert(self, payload, **kwargs):
        self.operation, self.payload = "upsert", payload
        return self

    def insert(self, payload):
        self.operation, self.payload = "insert", payload
        return self

    def select(self, columns):
        self.operation = "select"
        return self

    def delete(self):
        self.operation = "delete"
        return self

    def update(self, payload):
        self.operation, self.payload = "update", payload
        return self

    def eq(self, key, value):
        self.filters.append((key, value))
        return self

    def order(self, key, desc=False):
        return self

    def limit(self, count):
        return self

    async def execute(self):
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class FakeSupabase:
    def __init__(self, response):
        self.response = response
        self.requested_table = None
        self.query = None

    def table(self, table_name):
        self.requested_table = table_name
        self.query = FakeQuery(self.response)
        return self.query


def portfolio_config() -> PortfolioConfig:
    return PortfolioConfig(
        name="Balanced",
        tickers=["AAA", "BBB"],
        target_weights={"AAA": 0.6, "BBB": 0.4},
        risk_constraints={"max_drawdown": 0.25},
        optimization_parameters={"objective": "max_sharpe", "risk_free_rate": 0.03},
    )


def test_portfolio_save_upserts_json_compatible_configuration() -> None:
    config = portfolio_config()
    sb = FakeSupabase(SimpleNamespace(data=[], error=None))

    saved = asyncio.run(PortfolioRepository(sb).save(config))

    assert saved.id == config.id
    assert sb.requested_table == "portfolios"
    assert sb.query.operation == "upsert"
    assert sb.query.payload["target_weights"] == {"AAA": 0.6, "BBB": 0.4}
    assert sb.query.payload["risk_constraints"] == {"max_drawdown": 0.25}
    assert isinstance(sb.query.payload["id"], str)


def test_portfolio_get_validates_record_and_returns_none_for_missing() -> None:
    config = portfolio_config()
    row = config.to_record()
    sb = FakeSupabase(SimpleNamespace(data=[row], error=None))
    repo = PortfolioRepository(sb)

    found = asyncio.run(repo.get(config.id))
    assert found is not None and found.id == config.id
    assert sb.query.filters == [("id", str(config.id))]

    sb.response = SimpleNamespace(data=[], error=None)
    assert asyncio.run(repo.get(config.id)) is None


def test_backtest_repository_saves_and_builds_tear_sheet() -> None:
    backtest = BacktestRun(
        strategy_name="quality-value",
        start_date=date(2022, 1, 1),
        end_date=date(2023, 12, 31),
        tickers=["AAA", "BBB"],
        net_pnl_series=[1200.0, 800.0],
        observation_dates=[date(2023, 12, 30), date(2023, 12, 31)],
        turnover_series=[0.2, 0.1],
        total_turnover=0.3,
        total_transaction_cost=35.0,
        total_transaction_drag_bps=3.5,
        metrics={"cagr": 0.12, "sharpe": 1.1},
        telemetry={"execution_time_ms": 18.0},
    )
    sb = FakeSupabase(SimpleNamespace(data=[backtest.to_record()], error=None))
    repo = BacktestRepository(sb)

    saved = asyncio.run(repo.save_run(backtest))
    assert saved.id == backtest.id
    assert sb.requested_table == "backtest_runs"
    assert sb.query.payload["net_pnl_series"] == [1200.0, 800.0]

    sheet = asyncio.run(repo.get_tear_sheet(backtest.id))
    assert sheet is not None
    assert sheet["metrics"] == {"cagr": 0.12, "sharpe": 1.1}
    assert sheet["total_transaction_drag_bps"] == 3.5


def test_stress_repository_inserts_and_reads_reverse_stress_result() -> None:
    scenario = StressTest(
        stress_type="reverse_stress",
        scenario_name="-30 percent portfolio loss",
        inputs={"target_loss_threshold": -0.3},
        results={"factor_shocks": {"rates": 0.02, "growth": -0.15}},
        portfolio_return=-0.3,
        minimum_shock_magnitude=0.151,
    )
    sb = FakeSupabase(SimpleNamespace(data=[scenario.to_record()], error=None))
    repo = StressTestRepository(sb)

    inserted = asyncio.run(repo.save(scenario))
    loaded = asyncio.run(repo.get(scenario.id))

    assert inserted.id == scenario.id
    assert loaded is not None
    assert loaded.results["factor_shocks"]["growth"] == -0.15


def test_models_reject_inconsistent_portfolio_or_backtest_series() -> None:
    with pytest.raises(ValueError, match="exactly one entry"):
        PortfolioConfig(name="Bad", tickers=["AAA", "BBB"], target_weights={"AAA": 1.0})

    with pytest.raises(ValueError, match="align with net_pnl_series"):
        BacktestRun(
            strategy_name="bad",
            start_date=date(2023, 1, 1),
            end_date=date(2023, 1, 2),
            tickers=["AAA"],
            net_pnl_series=[1.0, 2.0],
            observation_dates=[date(2023, 1, 2)],
            total_turnover=0.0,
        )

    with pytest.raises(ValueError, match="finite numbers"):
        PortfolioConfig(
            name="NaN",
            tickers=["AAA"],
            target_weights={"AAA": 1.0},
            risk_constraints={"limit": float("nan")},
        )


def test_repository_wraps_postgrest_failures_without_leaking_credentials() -> None:
    sb = FakeSupabase(RuntimeError("service access token rejected"))
    repo = PortfolioRepository(sb)

    with pytest.raises(RepositoryError, match="Supabase upsert failed"):
        asyncio.run(repo.save(portfolio_config()))
