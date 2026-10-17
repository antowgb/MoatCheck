"""FastAPI integration coverage for v1 schemas, adapters, and error mapping."""

import pytest
from tests.asgi_client import TestClient

from app.main import app

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_markowitz_endpoint_returns_weights_and_execution_metadata(client: TestClient) -> None:
    response = client.post(
        "/api/v1/portfolio/markowitz/min-variance",
        json={
            "tickers": ["AAA", "BBB"],
            "expected_returns": [0.08, 0.10],
            "covariance_matrix": [[0.04, 0.0], [0.0, 0.09]],
            "risk_free_rate": 0.03,
            "points": 10,
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body) == {"data", "meta"}
    assert set(body["data"]["weights"]) == {"AAA", "BBB"}
    assert abs(sum(body["data"]["weights"].values()) - 1.0) < 1e-8
    assert body["meta"]["execution_time_ms"] >= 0
    assert body["meta"]["input_shape"]["covariance_matrix"] == [2, 2]


def test_frontend_optimize_alias_returns_frontier(client: TestClient) -> None:
    response = client.post(
        "/api/v1/portfolio/optimize",
        json={
            "tickers": ["AAA", "BBB"],
            "expected_returns": [0.08, 0.12],
            "covariance_matrix": [[0.04, 0.005], [0.005, 0.09]],
            "risk_free_rate": 0.03,
            "points": 4,
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert len(body["data"]["frontier"]) == 4
    assert body["meta"]["execution_time_ms"] >= 0


def test_frontend_risk_evaluate_alias_returns_var_and_cvar(client: TestClient) -> None:
    response = client.post(
        "/api/v1/risk/evaluate",
        json={
            "tickers": ["AAA", "BBB"],
            "weights": [0.5, 0.5],
            "returns": [[0.01, 0.02], [-0.02, -0.01], [0.005, 0.0], [-0.01, -0.03]],
            "confidence_level": 0.75,
            "method": "historical",
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["data"]["var"] >= 0
    assert response.json()["data"]["cvar"] >= response.json()["data"]["var"]


def test_frontend_backtest_run_alias_returns_wealth_path(client: TestClient) -> None:
    response = client.post(
        "/api/v1/backtest/run",
        json={
            "tickers": ["AAA", "BBB"],
            "initial_weights": [0.5, 0.5],
            "target_weights": [[0.6, 0.4], [0.4, 0.6]],
            "prices": [[100, 100], [101, 99], [100, 101]],
            "execution_cost_bps": 2.0,
        },
    )
    assert response.status_code == 200, response.text
    assert len(response.json()["data"]["wealth_path"]) == 3


def test_bundled_crisis_catalog_and_replay_endpoints(client: TestClient) -> None:
    catalog_response = client.get("/api/v1/risk/stress/scenarios")
    assert catalog_response.status_code == 200, catalog_response.text
    catalog = catalog_response.json()["data"]["scenarios"]
    assert {item["id"] for item in catalog} == {"lehman-2008", "covid-2020"}

    lehman = client.post("/api/v1/risk/stress/scenarios/lehman-2008/replay", json={})
    assert lehman.status_code == 200, lehman.text
    lehman_data = lehman.json()["data"]
    assert lehman_data["replay"]["n_observations"] == 49
    assert lehman_data["tickers"] == ["SP500", "VT", "IEF"]
    assert "BTC-USD" not in lehman_data["tickers"]
    assert lehman_data["replay"]["worst_drawdown"] < -0.25

    covid = client.post("/api/v1/risk/stress/scenarios/covid-2020/replay", json={})
    assert covid.status_code == 200, covid.text
    covid_data = covid.json()["data"]
    assert covid_data["tickers"] == ["SP500", "URTH", "IEF", "BTC-USD"]
    assert covid_data["replay"]["n_observations"] == 51
    assert len(covid_data["rolling_volatility_path"]) == 51
    assert covid_data["rolling_volatility_path"][-1]["portfolio_annualized_volatility"] is not None


def test_portfolio_schema_rejects_dimension_mismatch_with_422(client: TestClient) -> None:
    response = client.post(
        "/api/v1/portfolio/markowitz/max-sharpe",
        json={"tickers": ["AAA", "BBB"], "expected_returns": [0.1], "covariance_matrix": [[0.1, 0], [0, 0.2]]},
    )
    assert response.status_code == 422


def test_risk_var_endpoint_translates_matrix_and_returns_positive_losses(client: TestClient) -> None:
    response = client.post(
        "/api/v1/risk/var",
        json={
            "tickers": ["AAA", "BBB"],
            "weights": [0.5, 0.5],
            "returns": [[0.01, 0.02], [-0.02, -0.01], [0.005, 0.0], [-0.01, -0.03]],
            "confidence_level": 0.75,
            "method": "historical",
        },
    )
    assert response.status_code == 200, response.text
    data = response.json()["data"]
    assert data["var"] >= 0
    assert data["cvar"] >= data["var"]
    assert response.json()["meta"]["input_shape"]["returns"] == [4, 2]


def test_liquidity_var_endpoint_returns_position_horizons(client: TestClient) -> None:
    response = client.post(
        "/api/v1/risk/liquidity-adjusted-var",
        json={
            "tickers": ["AAA"],
            "portfolio_returns": [0.01, -0.02, 0.005, -0.01],
            "position_values": [100000.0],
            "advs": [50000.0],
            "volatilities": [0.02],
            "spreads": [0.001],
            "confidence_level": 0.75,
            "max_participation_rate": 0.1,
            "method": "historical",
        },
    )
    assert response.status_code == 200, response.text
    data = response.json()["data"]
    assert data["max_liquidation_days"] == 20
    assert data["liquidity_adjusted_var"] >= data["standard_var"]


def test_historical_stress_endpoint_serializes_paths_and_timestamps(client: TestClient) -> None:
    response = client.post(
        "/api/v1/risk/stress/historical",
        json={
            "tickers": ["AAA", "BBB"],
            "weights": [0.5, 0.5],
            "dates": ["2020-03-01", "2020-03-02", "2020-03-03"],
            "asset_returns": [[-0.02, -0.04], [0.01, -0.01], [-0.03, -0.02]],
            "scenario_start": "2020-03-01",
            "scenario_end": "2020-03-03",
        },
    )
    assert response.status_code == 200, response.text
    data = response.json()["data"]
    assert len(data["portfolio_return_path"]) == 3
    assert set(data["asset_contributions"]) == {"AAA", "BBB"}


def test_backtest_execution_cost_and_rebalance_simulation_endpoints(client: TestClient) -> None:
    execution = client.post(
        "/api/v1/backtest/execution-costs",
        json={
            "trade_sizes": [50000.0], "prices": [100.0], "advs": [1000000.0],
            "volatilities": [0.02], "spreads": [0.001],
        },
    )
    assert execution.status_code == 200, execution.text
    assert execution.json()["data"]["total_execution_drag"] > 0

    simulation = client.post(
        "/api/v1/backtest/rebalance-simulation",
        json={
            "tickers": ["AAA", "BBB"],
            "initial_weights": [0.5, 0.5],
            "target_weights": [[0.6, 0.4], [0.4, 0.6]],
            "prices": [[100, 100], [101, 99], [100, 101]],
            "execution_cost_bps": 2.0,
        },
    )
    assert simulation.status_code == 200, simulation.text
    wealth = simulation.json()["data"]["wealth_path"]
    assert len(wealth) == 3
    # The first target differs from the initial allocation and is applied
    # before its matching price interval, so the initial rebalance has costs.
    assert wealth["0"] == pytest.approx(1_001_960)


def test_core_validation_error_maps_to_422(client: TestClient, monkeypatch) -> None:
    from app.api.v1.endpoints import portfolio

    def fail(*args, **kwargs):
        raise ValueError("invalid covariance input")

    monkeypatch.setattr(portfolio, "optimize_minimum_variance", fail)
    response = client.post(
        "/api/v1/portfolio/markowitz/min-variance",
        json={"tickers": ["AAA"], "expected_returns": [0.1], "covariance_matrix": [[0.04]]},
    )
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "invalid_quant_input"
