import numpy as np
import pandas as pd
import pytest

from app.core.trading.execution_costs import (
    calculate_square_root_slippage,
    calculate_total_execution_drag,
    estimate_half_spread_cost,
    execution_cost_telemetry,
)


def test_half_spread_cost_is_half_the_full_quoted_spread():
    assert estimate_half_spread_cost(50_000.0, 0.001) == pytest.approx(25.0)
    assert estimate_half_spread_cost(0.0, 0.01) == 0.0


def test_infinite_liquidity_drives_square_root_impact_to_zero():
    assert calculate_square_root_slippage(50_000.0, np.inf, 0.02) == 0.0
    assert calculate_square_root_slippage(50_000.0, 1e32, 0.02) == pytest.approx(0.0, abs=1e-15)


def test_four_times_trade_value_doubles_square_root_impact_fraction():
    base = calculate_square_root_slippage(50_000.0, 1_000_000.0, 0.02)
    four_x = calculate_square_root_slippage(200_000.0, 1_000_000.0, 0.02)
    assert four_x == pytest.approx(2.0 * base)


def test_aggregate_execution_drag_breaks_down_all_cost_sources():
    result = calculate_total_execution_drag(
        trade_sizes=pd.Series([50_000.0]),
        prices=np.array([1.0]),
        advs=np.array([1_000_000.0]),
        volatilities=np.array([0.02]),
        spreads=np.array([0.001]),
        fixed_fee_per_trade=1.0,
        pct_broker_fee=0.0005,
    )
    impact_pct = 0.5 * 0.02 * np.sqrt(50_000.0 / 1_000_000.0)
    expected_impact = 50_000.0 * impact_pct
    assert result["cost_breakdown"]["spread_cost"] == pytest.approx(25.0)
    assert result["cost_breakdown"]["market_impact_cost"] == pytest.approx(expected_impact)
    assert result["cost_breakdown"]["fixed_broker_fees"] == pytest.approx(1.0)
    assert result["cost_breakdown"]["percentage_broker_fees"] == pytest.approx(25.0)
    assert result["total_execution_drag"] == pytest.approx(51.0 + expected_impact)
    assert result["total_friction_bps"] == pytest.approx(result["total_execution_drag"] / 50_000 * 10_000)
    assert result["input_shape"] == [1]
    assert result["latency_ms"] >= 0.0


def test_zero_order_and_infinite_adv_have_no_friction_or_market_impact():
    result = calculate_total_execution_drag(
        trade_sizes=np.array([0.0, 100.0]),
        prices=np.array([10.0, 10.0]),
        advs=np.array([1000.0, np.inf]),
        volatilities=np.array([0.02, 0.03]),
        spreads=np.array([0.01, 0.02]),
    )
    assert result["per_order_costs"][0] == 0.0
    assert result["impact_fraction_by_order"][1] == 0.0
    assert result["cost_breakdown"]["market_impact_cost"] == 0.0
    assert result["cost_breakdown"]["fixed_broker_fees"] == 1.0


def test_execution_telemetry_matches_aggregate_api():
    inputs = (np.array([10.0, -5.0]), np.array([20.0, 30.0]), np.array([1000.0, 2000.0]),
              np.array([0.01, 0.02]), np.array([0.001, 0.002]))
    aggregate = calculate_total_execution_drag(*inputs)
    telemetry = execution_cost_telemetry(*inputs)
    assert telemetry["total_execution_drag"] == pytest.approx(aggregate["total_execution_drag"])
    assert telemetry["input_shape"] == [2]
    assert set(telemetry["cost_breakdown"]) == {
        "spread_cost", "market_impact_cost", "fixed_broker_fees", "percentage_broker_fees"
    }


def test_mismatched_vectors_and_invalid_inputs_raise_clear_errors():
    with pytest.raises(ValueError, match="equal length"):
        calculate_total_execution_drag(
            np.array([1.0]), np.array([10.0, 20.0]), np.array([100.0]),
            np.array([0.01]), np.array([0.001]),
        )
    with pytest.raises(ValueError, match="strictly positive"):
        calculate_square_root_slippage(100.0, 0.0, 0.01)
    with pytest.raises(ValueError, match="positive"):
        calculate_total_execution_drag(
            np.array([1.0]), np.array([0.0]), np.array([100.0]),
            np.array([0.01]), np.array([0.001]),
        )


@pytest.mark.parametrize("bad_adv", [0.0, -1.0, np.nan, -np.inf])
def test_zero_negative_nan_or_negative_infinite_adv_is_rejected(bad_adv):
    with pytest.raises(ValueError):
        calculate_square_root_slippage(100.0, bad_adv, 0.01)
    with pytest.raises(ValueError):
        calculate_total_execution_drag(
            np.array([1.0]), np.array([100.0]), np.array([bad_adv]),
            np.array([0.01]), np.array([0.001]),
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"pct_broker_fee": np.nan},
        {"pct_broker_fee": np.inf},
        {"fixed_fee_per_trade": np.nan},
        {"fixed_fee_per_trade": np.inf},
    ],
)
def test_nonfinite_broker_cost_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        calculate_total_execution_drag(
            np.array([1.0]), np.array([100.0]), np.array([1000.0]),
            np.array([0.01]), np.array([0.001]), **kwargs,
        )


def test_nan_spread_and_volatility_vectors_are_rejected():
    for volatility, spread in ((np.nan, 0.001), (0.02, np.nan)):
        with pytest.raises(ValueError):
            calculate_total_execution_drag(
                np.array([1.0]), np.array([100.0]), np.array([1000.0]),
                np.array([volatility]), np.array([spread]),
            )
