import pytest

import numpy as np
import pandas as pd
import pytest

from app.core.trading.rebalance import (
    RebalancingEngine,
    build_rebalance_orders,
    calculate_portfolio_turnover,
    calculate_rebalancing_drag,
    rebalance_telemetry,
)


def test_rebalance_orders_and_telemetry_report_costs_and_turnover():
    current = {"A": 0.6, "B": 0.4}
    target = {"A": 0.5, "B": 0.5}
    prices = {"A": 100.0, "B": 100.0}
    result = build_rebalance_orders(
        current, target, prices, portfolio_value=1_000.0, integer_shares=False,
        average_daily_volume={"A": 10_000.0, "B": 10_000.0},
        volatility={"A": 0.02, "B": 0.02},
    )
    telemetry = rebalance_telemetry(
        current,
        target,
        prices,
        portfolio_value=1_000.0,
        integer_shares=False,
        average_daily_volume={"A": 10_000.0, "B": 10_000.0},
        volatility={"A": 0.02, "B": 0.02},
    )

    assert len(result["orders"]) == 2
    assert telemetry["input_shape"] == [2]
    assert telemetry["n_orders"] == 2
    assert telemetry["one_way_turnover"] == pytest.approx(0.1)
    assert telemetry["estimated_total_cost"] == pytest.approx(result["estimated_total_cost"])
    assert telemetry["latency_ms"] >= 0.0


def test_zero_weight_change_has_zero_turnover_and_zero_drag():
    weights = np.array([0.6, 0.4])
    assert calculate_portfolio_turnover(weights, weights.copy()) == 0.0
    turnover = pd.Series([0.0, 0.0, 0.0])
    assert calculate_rebalancing_drag(turnover, execution_cost_bps=10.0) == 0.0


def test_complete_portfolio_flip_has_gross_turnover_two():
    assert calculate_portfolio_turnover(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == 2.0


def test_monthly_turnover_drag_compounds_and_annualizes():
    drag = calculate_rebalancing_drag(pd.Series([0.2] * 12), execution_cost_bps=10.0)
    assert drag == pytest.approx(1.0 - (1.0 - 0.2 * 10.0 / 10_000.0) ** 12)


def test_rebalancing_engine_applies_turnover_cap_and_reports_paths():
    engine = RebalancingEngine(
        initial_weights=np.array([0.5, 0.5]),
        target_weights=np.array([[0.0, 1.0], [0.0, 1.0]]),
        prices=np.array([[100.0, 100.0], [100.0, 100.0], [100.0, 100.0]]),
        execution_cost_bps=0.0,
        max_turnover=0.5,
        initial_wealth=1_000_000.0,
    )
    result = engine.run()

    assert result["turnover_series"].to_list() == pytest.approx([0.5, 0.5])
    assert np.allclose(result["executed_target_weights"][0], [0.25, 0.75])
    assert np.allclose(result["executed_target_weights"][1], [0.0, 1.0])
    assert result["wealth_path"].iloc[0] == 1_000_000.0
    assert result["wealth_path"].iloc[-1] == pytest.approx(1_000_000.0)
    assert result["average_annual_turnover"] == pytest.approx(6.0)
    assert result["input_shape"]["prices"] == [3, 2]
    assert result["latency_ms"] >= 0.0


def test_engine_deducts_rebalancing_cost_and_records_negative_cash_flow():
    engine = RebalancingEngine(
        initial_weights=np.array([0.5, 0.5]),
        target_weights=np.array([[0.5, 0.5], [0.75, 0.25]]),
        prices=np.array([[100.0, 100.0], [110.0, 90.0], [110.0, 90.0]]),
        execution_cost_bps=5.0,
        initial_wealth=1_000_000.0,
    )
    result = engine.run()

    assert result["turnover_series"].iloc[0] == 0.0
    assert result["cash_flows"].loc[1, "turnover"] == pytest.approx(0.4)
    assert result["cash_flows"].loc[1, "transaction_cost"] == pytest.approx(200.0)
    assert result["cash_flows"].loc[1, "transaction_cost_cash_flow"] == pytest.approx(-200.0)
    assert result["wealth_path"].iloc[-1] == pytest.approx(999_800.0)
    assert result["total_transaction_drag_bps"] == pytest.approx(2.0)


def test_custom_transaction_model_receives_signed_order_quantities():
    seen = {}

    def model(order_quantities, current_prices):
        seen["quantities"] = order_quantities.copy()
        seen["prices"] = current_prices.copy()
        return 12.5

    engine = RebalancingEngine(
        np.array([1.0, 0.0]),
        np.array([[0.0, 1.0]]),
        np.array([[10.0, 20.0], [10.0, 20.0]]),
        transaction_costs_model=model,
        initial_wealth=1000.0,
    )
    result = engine.run()

    assert np.allclose(seen["quantities"], [-100.0, 50.0])
    assert np.allclose(seen["prices"], [10.0, 20.0])
    assert result["total_transaction_cost"] == pytest.approx(12.5)


def test_turnover_rejects_mismatched_or_nonfinite_weights():
    with pytest.raises(ValueError, match="same-length"):
        calculate_portfolio_turnover(np.array([1.0]), np.array([0.5, 0.5]))
    with pytest.raises(ValueError, match="finite"):
        calculate_portfolio_turnover(np.array([1.0]), np.array([np.nan]))


def test_engine_rejects_bad_period_shapes_and_invalid_turnover_caps():
    with pytest.raises(ValueError, match="one more time row"):
        RebalancingEngine(np.array([1.0]), np.array([[1.0]]), np.array([[100.0]]))
    with pytest.raises(ValueError, match="max_turnover"):
        RebalancingEngine(
            np.array([1.0]), np.array([[1.0]]), np.array([[100.0], [100.0]]),
            max_turnover=-0.1,
        )


@pytest.mark.parametrize("bad_adv", [0.0, -1.0, np.nan, np.inf])
def test_rebalance_orders_reject_invalid_adv_for_traded_assets(bad_adv):
    with pytest.raises(ValueError, match="average_daily_volume"):
        build_rebalance_orders(
            {"A": 1.0},
            {"A": 0.0, "B": 1.0},
            {"A": 100.0, "B": 100.0},
            portfolio_value=1_000.0,
            average_daily_volume={"A": 1_000.0, "B": bad_adv},
            volatility={"A": 0.02, "B": 0.02},
            integer_shares=False,
        )


def test_rebalance_orders_reject_missing_adv_and_nonfinite_transaction_fee():
    with pytest.raises(ValueError, match="average_daily_volume"):
        build_rebalance_orders(
            {"A": 1.0}, {"A": 0.0, "B": 1.0}, {"A": 100.0, "B": 100.0},
            portfolio_value=1_000.0,
            volatility={"A": 0.02, "B": 0.02}, integer_shares=False,
        )
    for invalid_fee in (np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError, match="transaction_cost_bps"):
            build_rebalance_orders(
                {"A": 1.0}, {"A": 1.0}, {"A": 100.0},
                portfolio_value=1_000.0, transaction_cost_bps=invalid_fee,
            )


def test_rebalance_orders_reject_missing_volatility_for_traded_asset():
    with pytest.raises(ValueError, match="volatility"):
        build_rebalance_orders(
            {"A": 1.0}, {"A": 0.0, "B": 1.0}, {"A": 100.0, "B": 100.0},
            portfolio_value=1_000.0,
            average_daily_volume={"A": 1_000.0, "B": 1_000.0},
            integer_shares=False,
        )
