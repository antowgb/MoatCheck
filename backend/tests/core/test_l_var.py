import numpy as np
import pandas as pd
import pytest

from app.core.trading.l_var import (
    calculate_liquidation_horizon,
    calculate_liquidity_adjusted_var,
    evaluate_portfolio_liquidity_profile,
    liquidity_var_telemetry,
)


def test_infinite_liquidity_collapses_horizon_to_one_and_lvar_to_standard_var():
    returns = pd.Series([-0.02, 0.02, -0.01, 0.01])
    result = calculate_liquidity_adjusted_var(
        returns,
        position_values=np.array([1_000_000.0]),
        advs=np.array([np.inf]),
        volatilities=np.array([0.02]),
        spreads=np.array([0.0]),
    )
    assert result["max_liquidation_days"] == 1
    assert result["market_impact_liquidation_cost"] == 0.0
    assert result["spread_liquidation_cost"] == 0.0
    assert result["liquidity_adjusted_var"] == pytest.approx(result["standard_var"])
    assert liquidity_var_telemetry(result)["total_lvar_currency"] == pytest.approx(result["standard_var"])


def test_zero_position_has_zero_horizon_var_and_liquidation_costs():
    result = calculate_liquidity_adjusted_var(
        pd.Series([-0.03, 0.01, -0.01]),
        position_values=np.array([0.0]),
        advs=np.array([100_000.0]),
        volatilities=np.array([0.03]),
        spreads=np.array([0.02]),
    )
    assert result["max_liquidation_days"] == 0
    assert result["liquidity_adjusted_var"] == 0.0
    assert result["total_liquidation_cost"] == 0.0


def test_illiquid_large_position_extends_horizon_and_increases_lvar():
    result = calculate_liquidity_adjusted_var(
        pd.Series([-0.02, 0.02, -0.01, 0.01]),
        position_values=np.array([2_000_000.0]),
        advs=np.array([50_000.0]),
        volatilities=np.array([0.02]),
        spreads=np.array([0.001]),
        confidence_level=0.95,
    )
    assert result["max_liquidation_days"] == 400
    assert result["liquidity_adjusted_var"] > result["standard_var"]
    assert result["total_liquidation_cost"] > 0.0


def test_horizon_calculation_rounds_up_and_handles_zero_positions():
    horizons = calculate_liquidation_horizon(
        np.array([0.0, 100.0, 101.0]),
        np.array([10_000.0, 100.0, 100.0]),
        max_participation_rate=0.10,
    )
    assert horizons.tolist() == [0, 10, 11]


def test_liquidity_profile_reports_weighted_horizon_concentration_and_score():
    result = evaluate_portfolio_liquidity_profile(
        portfolio_weights=np.array([0.5, 0.5]),
        total_capital=2_000_000.0,
        advs=np.array([50_000.0, 20_000_000.0]),
        max_participation_rate=0.10,
    )
    assert result["liquidation_days_by_asset"].tolist() == [200, 1]
    assert result["weighted_average_liquidation_days"] == pytest.approx(100.5)
    assert result["illiquid_asset_concentration_ratio"] == pytest.approx(0.5)
    assert result["liquidity_score"] == pytest.approx(100.0 / 100.5)
    assert result["latency_ms"] >= 0.0


def test_historical_var_method_and_input_shapes_are_reported():
    returns = pd.Series([-0.05, -0.02, 0.0, 0.01, 0.03])
    result = calculate_liquidity_adjusted_var(
        returns,
        position_values=np.array([100_000.0, -50_000.0]),
        advs=np.array([1_000_000.0, 1_000_000.0]),
        volatilities=np.array([0.02, 0.03]),
        spreads=np.array([0.001, 0.002]),
        confidence_level=0.8,
        method="historical",
    )
    assert result["method"] == "historical"
    assert result["standard_var"] > 0
    assert result["input_shape"] == {"portfolio_returns": [5], "positions": [2]}


@pytest.mark.parametrize("bad_participation", [0.0, -0.1, 1.1, np.nan])
def test_invalid_participation_rates_raise(bad_participation):
    with pytest.raises(ValueError, match="max_participation_rate"):
        calculate_liquidation_horizon(np.array([1.0]), np.array([10.0]), bad_participation)


def test_mismatched_and_invalid_market_inputs_raise():
    with pytest.raises(ValueError, match="matching shapes"):
        calculate_liquidation_horizon(np.array([1.0]), np.array([10.0, 20.0]))
    with pytest.raises(ValueError, match="strictly positive"):
        calculate_liquidation_horizon(np.array([1.0]), np.array([0.0]))
    with pytest.raises(ValueError, match="at least two"):
        calculate_liquidity_adjusted_var(
            pd.Series([0.01]), np.array([100.0]), np.array([1000.0]),
            np.array([0.02]), np.array([0.001]), method="parametric",
        )
