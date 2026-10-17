"""Contract checks for the frontend's stable v1 API route aliases.

These tests do not require a network listener or the optional TestClient
transport. They verify FastAPI registration and execute the same endpoint
adapters with validated Pydantic request models.
"""

from app.main import app
from app.api.v1.endpoints.backtest import run_backtest
from app.api.v1.endpoints.portfolio import optimize
from app.api.v1.endpoints.risk import evaluate_risk
from app.schemas.backtest import RebalanceSimulationRequest
from app.schemas.portfolio import PortfolioOptimizationRequest
from app.schemas.risk import PortfolioReturnsRequest


def test_frontend_route_aliases_are_registered() -> None:
    paths = app.openapi()["paths"]
    assert "post" in paths["/api/v1/portfolio/optimize"]
    assert "post" in paths["/api/v1/risk/evaluate"]
    assert "post" in paths["/api/v1/backtest/run"]


def test_frontend_route_alias_handlers_return_enveloped_results() -> None:
    portfolio_request = PortfolioOptimizationRequest(
        tickers=["AAA", "BBB"],
        expected_returns=[0.08, 0.12],
        covariance_matrix=[[0.04, 0.005], [0.005, 0.09]],
        points=4,
    )
    risk_request = PortfolioReturnsRequest(
        tickers=["AAA", "BBB"],
        weights=[0.5, 0.5],
        returns=[[0.01, 0.02], [-0.02, -0.01], [0.005, 0], [-0.01, -0.03]],
        confidence_level=0.75,
        method="historical",
    )
    backtest_request = RebalanceSimulationRequest(
        tickers=["AAA", "BBB"],
        initial_weights=[0.5, 0.5],
        target_weights=[[0.6, 0.4], [0.4, 0.6]],
        prices=[[100, 100], [101, 99], [100, 101]],
        execution_cost_bps=2.0,
    )

    optimized = optimize(portfolio_request)
    risk = evaluate_risk(risk_request)
    backtest = run_backtest(backtest_request)

    assert set(optimized) == {"data", "meta"}
    assert len(optimized["data"]["frontier"]) == 4
    assert risk["data"]["var"] >= 0
    assert risk["data"]["cvar"] >= risk["data"]["var"]
    assert len(backtest["data"]["wealth_path"]) == 3
    for result in (optimized, risk, backtest):
        assert result["meta"]["execution_time_ms"] >= 0
