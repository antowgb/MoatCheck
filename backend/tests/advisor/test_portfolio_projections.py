from datetime import date, datetime, timedelta, timezone
from uuid import uuid4

import numpy as np
import pytest
from pydantic import ValidationError

from app.advisor.service import build_advisory_recommendation, calculate_portfolio_projections
from app.api.dependencies import AuthenticatedUser, get_optional_current_user
from app.main import app
from app.models.portfolio import PortfolioConfig
from app.schemas.advisor import PortfolioHoldingsInput, PortfolioTrackingSettingsInput
from tests.asgi_client import TestClient


def test_projection_matches_monthly_compound_interest_without_risk():
    result = calculate_portfolio_projections(10_000, 200, 0.06, 0.0, 5)
    monthly_rate = (1.06 ** (1 / 12)) - 1
    expected = 10_000 * 1.06**5 + 200 * sum((1 + monthly_rate) ** month for month in range(60))
    point = result[-1]
    assert point["year"] == 5
    assert point["pessimistic_value"] == pytest.approx(expected, rel=1e-10)
    assert point["median_value"] == pytest.approx(expected, rel=1e-10)
    assert point["optimistic_value"] == pytest.approx(expected, rel=1e-10)
    assert point["total_capital_deposited"] == 22_000
    assert point["median_net_gain"] == pytest.approx(expected - 22_000)


def test_projection_scenario_percentiles_are_ordered_and_deterministic():
    first = calculate_portfolio_projections(25_000, 300, 0.07, 0.18, 10)
    second = calculate_portfolio_projections(25_000, 300, 0.07, 0.18, 10)
    assert first == second
    assert [point["year"] for point in first] == list(range(1, 11))
    for point in first:
        assert point["pessimistic_value"] < point["median_value"] < point["optimistic_value"]
        assert point["pessimistic_net_gain"] == pytest.approx(point["pessimistic_value"] - point["total_capital_deposited"])


def test_projection_rejects_out_of_range_horizon():
    with pytest.raises(ValueError, match="between 1 and 20"):
        calculate_portfolio_projections(1000, 100, .05, .1, 21)


def test_advisor_values_holdings_and_creates_buy_only_dca(market_snapshot, advisor_policy):
    result = build_advisory_recommendation(
        portfolio_name="Tracked", tickers=["AAA", "BBB"],
        current_weights={"AAA": .5, "BBB": .5}, market=market_snapshot, policy=advisor_policy,
        holdings=[
            {"ticker": "AAA", "quantity": 10.0, "average_buy_price": 80.0, "currency": "EUR"},
            {"ticker": "BBB", "quantity": 20.0, "average_buy_price": 60.0, "currency": "EUR"},
        ], monthly_dca_amount=500, target_horizon_years=3,
    )
    assert result["valuation"]["total_market_value"] == 2_000
    assert result["valuation"]["total_cost_basis"] == 2_000
    assert result["valuation"]["unrealized_pnl"] == 0
    assert sum(order["amount"] for order in result["dca_plan"]["orders"]) == pytest.approx(500)
    assert all(order["side"] == "buy" for order in result["dca_plan"]["orders"])
    assert result["projections"]["initial_capital"] == 2_000
    assert result["projections"]["target_horizon_years"] == 3
    assert len(result["projections"]["years"]) == 3


def test_untracked_saved_portfolio_starts_with_cash_and_uses_only_monthly_dca(
    market_snapshot, advisor_policy
):
    monthly_dca = 500.0
    result = build_advisory_recommendation(
        portfolio_name="Starting from cash",
        tickers=["AAA", "BBB"],
        current_weights={"AAA": 0.0, "BBB": 0.0},
        market=market_snapshot,
        policy=advisor_policy,
        holdings=[],
        monthly_dca_amount=monthly_dca,
        target_horizon_years=3,
    )

    assert result["current_weights"] == {"AAA": 0.0, "BBB": 0.0, "CASH": 1.0}
    assert result["valuation"]["available"] is False
    assert result["projections"]["initial_capital"] == 0.0
    assert result["projections"]["years"][-1]["total_capital_deposited"] == 18_000.0
    invested_monthly_amount = monthly_dca * sum(
        result["target_weights"][ticker] for ticker in ("AAA", "BBB")
    )
    assert sum(order["amount"] for order in result["dca_plan"]["orders"]) == pytest.approx(invested_monthly_amount)
    assert result["dca_plan"]["unallocated_amount"] == pytest.approx(monthly_dca - invested_monthly_amount)
    assert result["execution_cost"]["total_traded_notional"] <= monthly_dca
    assert result["liquidity_adjusted_var_currency"] <= monthly_dca


def test_advisor_rejects_mixed_currency_holdings(market_snapshot, advisor_policy):
    market_snapshot.price_currencies = ["EUR", "USD"]
    with pytest.raises(ValueError, match="one currency"):
        build_advisory_recommendation(
            portfolio_name="Mixed", tickers=["AAA", "BBB"], current_weights={"AAA": .5, "BBB": .5},
            market=market_snapshot, policy=advisor_policy,
            holdings=[
                {"ticker": "AAA", "quantity": 1.0, "average_buy_price": 100.0, "currency": "EUR"},
                {"ticker": "BBB", "quantity": 1.0, "average_buy_price": 50.0, "currency": "USD"},
            ],
        )


def test_tracking_contract_validates_positions_and_settings():
    payload = PortfolioHoldingsInput(holdings=[{
        "ticker": "aaa", "quantity": 2.5, "average_buy_price": 100.0, "currency": "eur",
        "acquired_at": "2020-01-02",
    }])
    assert payload.holdings[0].ticker == "AAA"
    assert payload.holdings[0].currency == "EUR"
    assert payload.holdings[0].acquired_at == date(2020, 1, 2)
    legacy_payload = PortfolioHoldingsInput(holdings=[{
        "ticker": "AAA", "quantity": 1, "average_buy_price": 10, "currency": "",
    }])
    assert legacy_payload.holdings[0].currency == "EUR"
    with pytest.raises(ValidationError, match="ISO date"):
        PortfolioHoldingsInput(holdings=[{
            "ticker": "AAA", "quantity": 1, "average_buy_price": 10,
            "currency": "EUR", "acquired_at": "not-a-date",
        }])
    with pytest.raises(ValidationError):
        PortfolioHoldingsInput(holdings=[
            {"ticker": "AAA", "quantity": 1, "average_buy_price": 10, "currency": "EUR"},
            {"ticker": "AAA", "quantity": 2, "average_buy_price": 11, "currency": "EUR"},
        ])
    with pytest.raises(ValidationError):
        PortfolioTrackingSettingsInput(monthly_dca_amount=-1, target_horizon_years=5)
    assert PortfolioTrackingSettingsInput(monthly_dca_amount=250).target_horizon_years == 5


def test_tracking_settings_endpoint_is_owner_scoped_and_returns_contract(monkeypatch):
    owner_id, portfolio_id = uuid4(), uuid4()
    app.dependency_overrides[get_optional_current_user] = lambda: AuthenticatedUser(id=owner_id)
    # require_current_user resolves independently in the route dependency graph.
    from app.api.dependencies import require_current_user
    app.dependency_overrides[require_current_user] = lambda: AuthenticatedUser(id=owner_id)
    portfolio = PortfolioConfig(id=portfolio_id, owner_id=owner_id, name="Long term", tickers=["AAA"], target_weights={"AAA": 1.0})
    saved = []

    class FakePortfolios:
        def __init__(self, client): pass
        async def get_for_owner(self, requested_id, requested_owner):
            assert requested_id == portfolio_id and requested_owner == owner_id
            return portfolio

    class FakeTracking:
        def __init__(self, client): pass
        async def get_settings(self, requested_id):
            return type("Settings", (), {"model_copy": lambda self, update: type("Saved", (), {**update, "updated_at": datetime.now(timezone.utc), "model_dump": lambda self, **kwargs: {"monthly_dca_amount": self.monthly_dca_amount, "target_horizon_years": self.target_horizon_years}})()})()
        async def save_settings(self, settings):
            saved.append(settings)
            return settings

    async def fake_client(): return object()
    monkeypatch.setattr("app.api.v1.endpoints.advisor.get_async_supabase_client", fake_client)
    monkeypatch.setattr("app.api.v1.endpoints.advisor.PortfolioRepository", FakePortfolios)
    monkeypatch.setattr("app.api.v1.endpoints.advisor.PortfolioTrackingRepository", FakeTracking)
    try:
        with TestClient(app) as client:
            response = client.request("PUT", f"/api/v1/advisor/portfolios/{portfolio_id}/tracking-settings", json={"monthly_dca_amount": 400.0, "target_horizon_years": 7})
        assert response.status_code == 200, response.text
        assert response.json()["data"]["monthly_dca_amount"] == 400.0
        assert response.json()["data"]["target_horizon_years"] == 7
        assert len(saved) == 1
    finally:
        app.dependency_overrides.pop(get_optional_current_user, None)
        app.dependency_overrides.pop(require_current_user, None)


def test_holdings_endpoint_replaces_only_owned_portfolio_positions(monkeypatch):
    owner_id, portfolio_id = uuid4(), uuid4()
    from app.api.dependencies import require_current_user
    app.dependency_overrides[require_current_user] = lambda: AuthenticatedUser(id=owner_id)
    portfolio = PortfolioConfig(id=portfolio_id, owner_id=owner_id, name="Tracked", tickers=["AAA"], target_weights={"AAA": 1.0})
    saved = []

    class FakePortfolios:
        def __init__(self, client): pass
        async def get_for_owner(self, requested_id, requested_owner):
            assert requested_id == portfolio_id and requested_owner == owner_id
            return portfolio

    class FakeTracking:
        def __init__(self, client): pass
        async def list_holdings(self, requested_id):
            assert requested_id == portfolio_id
            return []
        async def replace_holdings(self, requested_id, holdings):
            assert requested_id == portfolio_id
            saved.extend(holdings)
            return holdings

    async def fake_client(): return object()
    async def fake_queue_refresh(_client, tickers): return [{"ticker": ticker, "status": "queued"} for ticker in tickers]
    monkeypatch.setattr("app.api.v1.endpoints.advisor.get_async_supabase_client", fake_client)
    monkeypatch.setattr("app.api.v1.endpoints.advisor.PortfolioRepository", FakePortfolios)
    monkeypatch.setattr("app.api.v1.endpoints.advisor.PortfolioTrackingRepository", FakeTracking)
    monkeypatch.setattr("app.api.v1.endpoints.advisor._queue_price_refreshes", fake_queue_refresh)
    try:
        with TestClient(app) as client:
            response = client.request("PUT", f"/api/v1/advisor/portfolios/{portfolio_id}/holdings", json={"holdings": [{"ticker": "AAA", "quantity": 3.0, "average_buy_price": 90.0, "currency": "", "acquired_at": "2020-01-02"}]})
        assert response.status_code == 200, response.text
        assert response.json()["data"]["holdings"][0]["ticker"] == "AAA"
        assert response.json()["data"]["holdings"][0]["currency"] == "EUR"
        assert saved[0].quantity == 3.0
        assert saved[0].acquired_at == date(2020, 1, 2)
    finally:
        app.dependency_overrides.pop(require_current_user, None)


@pytest.mark.parametrize("kwargs", [{"target_horizon_years": 2.5}, {"n_simulations": 1000.5}, {"target_horizon_years": True}])
def test_projection_rejects_fractional_or_boolean_counts(kwargs):
    with pytest.raises(ValueError):
        calculate_portfolio_projections(1000, 100, .05, .1, **kwargs)


def test_mixed_quote_currencies_rejected_even_without_holdings_or_dca(market_snapshot, advisor_policy):
    market_snapshot.price_currencies = ["EUR", "USD"]
    with pytest.raises(ValueError, match="one currency"):
        build_advisory_recommendation(
            portfolio_name="Mixed", tickers=["AAA", "BBB"],
            current_weights={"AAA": .5, "BBB": .5}, market=market_snapshot, policy=advisor_policy,
        )


def test_single_quote_currency_is_used_for_untracked_projections(market_snapshot, advisor_policy):
    market_snapshot.price_currencies = ["USD", "USD"]
    result = build_advisory_recommendation(
        portfolio_name="USD", tickers=["AAA", "BBB"], current_weights={"AAA": .5, "BBB": .5},
        market=market_snapshot, policy=advisor_policy,
    )
    assert result["valuation"]["currency"] == result["projections"]["currency"] == result["dca_plan"]["currency"] == "USD"


@pytest.mark.parametrize("capital, volatility", [(1e308, .1), (1000, 1e308)])
def test_projection_reports_overflow_as_validation_error_without_warnings(capital, volatility):
    with pytest.raises(ValueError, match="numerical range"):
        calculate_portfolio_projections(capital, 0, .25, volatility, 20)
