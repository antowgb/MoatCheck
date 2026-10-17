import asyncio
from datetime import date
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest

from app.advisor.discovery import DiscoveryConfig, build_portfolio_discovery_proposals, find_replacement_candidates
from app.models.portfolio import PortfolioConfig
from app.services.portfolio_discovery import run_portfolio_discovery


@pytest.fixture
def synthetic_panel():
    rng = np.random.default_rng(202610)
    n = 400
    base = rng.normal(0.00015, 0.012, n)
    other = 0.65 * base + rng.normal(0.0001, 0.007, n)
    candidate = rng.normal(0.0022, 0.006, n)
    noise = rng.normal(0.0001, 0.009, n)
    return pd.DataFrame({"AAA": base, "BBB": other, "CCC": candidate, "DDD": noise},
                        index=pd.bdate_range(end="2026-10-01", periods=n))


def test_candidate_filter_requires_asset_and_portfolio_dominance(synthetic_panel):
    candidates = find_replacement_candidates(
        synthetic_panel, {"AAA": 0.5, "BBB": 0.5}, periods_per_year=252,
        minimum_observations=300, limit=10,
    )
    assert candidates
    assert all(item["asset_sharpe_after"] > item["asset_sharpe_before"] for item in candidates)
    assert all(item["portfolio_sharpe_after"] > item["portfolio_sharpe_before"] for item in candidates)
    assert all(item["portfolio_volatility_after"] < item["portfolio_volatility_before"] for item in candidates)
    assert all(item["absolute_correlation_to_other_holdings_after"] < item["absolute_correlation_to_other_holdings_before"] for item in candidates)


def test_candidate_filter_excludes_weak_or_sparse_names(synthetic_panel):
    panel = synthetic_panel.copy()
    panel["SPARSE"] = np.nan
    panel.loc[panel.index[:50], "SPARSE"] = 0.2
    candidates = find_replacement_candidates(panel, {"AAA": 0.5, "BBB": 0.5}, minimum_observations=100)
    assert all(item["with_ticker"] != "SPARSE" for item in candidates)


def test_build_proposals_contains_comparable_models_and_telemetry(synthetic_panel):
    result = build_portfolio_discovery_proposals(
        synthetic_panel, {"AAA": 0.5, "BBB": 0.5},
        config=DiscoveryConfig(periods_per_year=252, minimum_observations=300, model_universe_limit=4),
    )
    assert result["lookback_years"] == 3
    assert result["universe_assets_evaluated"] == 4
    assert result["telemetry"]["execution_time_ms"] >= 0
    assert result["telemetry"]["input_shape"]["return_panel"] == [400, 4]
    assert result["model_portfolios"]["maximum_sharpe"]["success"]
    assert result["model_portfolios"]["minimum_risk"]["success"]
    assert result["model_portfolios"]["maximum_sharpe"]["long_term_projection_per_unit"]
    assert all(abs(sum(model["weights"].values()) - 1) < 1e-6 for model in result["model_portfolios"].values())
    assert "swap_proposals" in result


def test_service_loads_indexed_universe_and_price_batches(monkeypatch, synthetic_panel):
    tickers = list(synthetic_panel.columns)

    class Query:
        def __init__(self, table): self.table_name = table; self.symbols = []; self.start_date = None; self.start = 0; self.end = 999
        def select(self, *_args): return self
        def range(self, start, end): self.start, self.end = start, end; return self
        def in_(self, _key, values): self.symbols = values; return self
        def gte(self, _key, value): self.start_date = value; return self
        def order(self, *_args): return self
        async def execute(self):
            if self.table_name == "stocks":
                values = [{"ticker": ticker} for ticker in tickers]
                return type("Response", (), {"data": values[self.start:self.end + 1]})()
            data = []
            for ticker in self.symbols:
                series = synthetic_panel[ticker]
                prices = 100 * (1 + series).cumprod()
                data.extend({"ticker": ticker, "date": day.date().isoformat(), "close": price, "adjusted_close": price}
                            for day, price in prices.items())
            return type("Response", (), {"data": data[self.start:self.end + 1]})()

    class Client:
        def table(self, name): return Query(name)

    portfolio = PortfolioConfig(id=uuid4(), owner_id=uuid4(), name="Test", tickers=["AAA", "BBB"],
                                target_weights={"AAA": .5, "BBB": .5})
    result = asyncio.run(run_portfolio_discovery(
        Client(), portfolio, as_of=date(2026, 10, 1),
        config=DiscoveryConfig(periods_per_year=252, minimum_observations=300, model_universe_limit=4),
    ))
    assert result["portfolio_id"] == str(portfolio.id)
    assert result["portfolio_name"] == "Test"
    assert result["as_of"] == "2026-10-01"
    assert "replacement_candidates" in result and "model_portfolios" in result


def test_advisor_discovery_api_get_and_apply_proposal(monkeypatch):
    from tests.asgi_client import TestClient
    from app.api.dependencies import AuthenticatedUser, require_current_user
    from app.main import app

    owner_id, portfolio_id = uuid4(), uuid4()
    existing = PortfolioConfig(id=portfolio_id, owner_id=owner_id, name="Long term", tickers=["AAA", "BBB"],
                               target_weights={"AAA": .5, "BBB": .5})

    class FakePortfolioRepository:
        def __init__(self, _client): self.value = existing
        async def get_for_owner(self, requested_id, requested_owner):
            assert requested_id == portfolio_id and requested_owner == owner_id
            return self.value
        async def save(self, portfolio):
            self.value = portfolio
            return portfolio

    async def fake_client(): return object()
    async def fake_discovery(_client, _portfolio):
        return {"portfolio_id": str(portfolio_id), "portfolio_name": "Long term", "as_of": "2026-10-01",
                "replacement_candidates": [], "swap_proposals": [],
                "model_portfolios": {"maximum_sharpe": {"weights": {"AAA": .4, "BBB": .6}},
                                     "minimum_risk": {"weights": {"AAA": .6, "BBB": .4}}},
                "telemetry": {"execution_time_ms": 1.0, "input_shape": {"return_panel": [300, 2]}}}

    monkeypatch.setattr("app.api.v1.endpoints.advisor.get_async_supabase_client", fake_client)
    monkeypatch.setattr("app.api.v1.endpoints.advisor.PortfolioRepository", FakePortfolioRepository)
    monkeypatch.setattr("app.api.v1.endpoints.advisor.run_portfolio_discovery", fake_discovery)
    app.dependency_overrides[require_current_user] = lambda: AuthenticatedUser(id=owner_id)
    try:
        with TestClient(app) as client:
            get_response = client.get(f"/api/v1/advisor/portfolios/{portfolio_id}/proposals")
            assert get_response.status_code == 200, get_response.text
            assert "maximum_sharpe" in get_response.json()["data"]["model_portfolios"]
            apply_response = client.post(f"/api/v1/advisor/portfolios/{portfolio_id}/apply-proposal", json={
                "proposal_id": "model:maximum-sharpe", "tickers": ["AAA", "BBB"],
                "target_weights": {"AAA": .4, "BBB": .6},
            })
        assert apply_response.status_code == 200, apply_response.text
        assert apply_response.json()["data"]["target_weights"] == {"AAA": .4, "BBB": .6}
        assert apply_response.json()["data"]["trade_execution"] == "not_executed"
    finally:
        app.dependency_overrides.pop(require_current_user, None)
