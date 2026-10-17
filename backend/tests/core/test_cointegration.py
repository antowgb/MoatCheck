import numpy as np
import pandas as pd
import pytest

from app.core.trading.cointegration import (
    calculate_spread_zscore,
    cointegration_telemetry,
    estimate_point_in_time_hedge_ratio,
    simulate_stat_arb_strategy,
    test_engle_granger_cointegration as run_engle_granger_test,
)


def _price_pair(seed=17, n=500):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    common_walk = np.cumsum(rng.normal(0.0, 1.0, n)) + 100.0
    stationary_spread = np.zeros(n)
    noise = rng.normal(0.0, 0.7, n)
    for i in range(1, n):
        stationary_spread[i] = 0.65 * stationary_spread[i - 1] + noise[i]
    price_b = pd.Series(common_walk + 150.0, index=dates)
    price_a = pd.Series(1.4 * common_walk + stationary_spread + 200.0, index=dates)
    return price_a, price_b


def test_engle_granger_detects_stationary_spread_in_random_walk_pair():
    price_a, price_b = _price_pair()
    result = run_engle_granger_test(price_a, price_b)
    assert result["is_cointegrated"] is True
    assert result["p_value"] < 0.05
    assert result["hedge_ratio"] == pytest.approx(1.4, abs=0.05)
    assert np.isfinite(result["residual_adf_statistic"])
    assert result["input_shape"] == [500, 2]
    assert result["latency_ms"] >= 0.0


def test_independent_random_walks_are_not_flagged_cointegrated():
    rng = np.random.default_rng(812)
    dates = pd.date_range("2018-01-01", periods=700, freq="B")
    price_a = pd.Series(100.0 + np.cumsum(rng.normal(size=700)), index=dates)
    price_b = pd.Series(120.0 + np.cumsum(rng.normal(size=700)), index=dates)
    result = run_engle_granger_test(price_a, price_b)
    assert result["is_cointegrated"] is False
    assert result["p_value"] >= 0.05


def test_exact_linear_pair_is_detected_without_degenerate_adf_warning():
    dates = pd.date_range("2024-01-01", periods=20, freq="D")
    price_b = pd.Series(np.arange(100.0, 120.0), index=dates)
    price_a = 3.0 + 1.5 * price_b
    result = run_engle_granger_test(price_a, price_b)
    assert result["is_cointegrated"] is True
    assert result["p_value"] == 0.0
    assert result["hedge_ratio"] == pytest.approx(1.5)


def test_rolling_spread_zscore_uses_trailing_window_and_preserves_index():
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    price_b = pd.Series([100.0] * 5, index=dates)
    price_a = pd.Series([120.0, 121.0, 120.0, 122.0, 120.0], index=dates)
    score = calculate_spread_zscore(price_a, price_b, hedge_ratio=1.0, window=3)
    assert score.index.equals(dates)
    assert score.iloc[:2].isna().all()
    assert np.isfinite(score.iloc[2:]).all()


def test_zero_variance_spread_returns_nan_scores():
    dates = pd.date_range("2024-01-01", periods=8, freq="D")
    price_b = pd.Series(np.linspace(10.0, 17.0, 8), index=dates)
    price_a = 2.0 * price_b
    score = calculate_spread_zscore(price_a, price_b, hedge_ratio=2.0, window=3)
    assert score.isna().all()


def test_point_in_time_hedge_ratio_is_unchanged_by_future_observations():
    dates = pd.date_range("2024-01-01", periods=12, freq="D")
    price_b = pd.Series(np.arange(100.0, 112.0), index=dates)
    price_a = pd.Series(5.0 + 1.7 * price_b + np.sin(np.arange(12)), index=dates)
    original_beta = estimate_point_in_time_hedge_ratio(price_a, price_b)
    changed_a = price_a.copy()
    changed_b = price_b.copy()
    changed_a.iloc[8:] *= 1.8
    changed_b.iloc[8:] += 50.0
    revised_beta = estimate_point_in_time_hedge_ratio(changed_a, changed_b)
    assert original_beta.iloc[:8].equals(revised_beta.iloc[:8])
    assert original_beta.iloc[:3].isna().all()
    assert np.isfinite(original_beta.iloc[3:]).all()


def test_expanding_and_rolling_calibration_are_available_and_causal():
    dates = pd.date_range("2024-01-01", periods=15, freq="D")
    price_b = pd.Series(np.arange(100.0, 115.0), index=dates)
    price_a = pd.Series(10.0 + 1.2 * price_b, index=dates)
    price_a.iloc[8:] = 20.0 + 2.0 * price_b.iloc[8:]
    expanding_z = calculate_spread_zscore(price_a, price_b, window=3)
    rolling_z = calculate_spread_zscore(
        price_a, price_b, window=3, calibration_window=5
    )
    beta_expanding = expanding_z.attrs["hedge_ratios"]
    beta_rolling = rolling_z.attrs["hedge_ratios"]
    assert beta_expanding.iloc[:3].isna().all()
    assert beta_rolling.iloc[:3].isna().all()
    assert np.isfinite(beta_expanding.iloc[3:]).all()
    assert np.isfinite(beta_rolling.iloc[3:]).all()
    assert not np.allclose(beta_expanding.iloc[8:], beta_rolling.iloc[8:])
    with pytest.raises(ValueError, match="either"):
        calculate_spread_zscore(
            price_a, price_b, hedge_ratio=1.0, window=3, calibration_window=5
        )


def test_strategy_is_causal_enters_at_close_and_pnl_starts_next_period():
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    price_b = pd.Series([100.0] * 5, index=dates)
    price_a = pd.Series([120.0, 130.0, 128.0, 122.0, 120.0], index=dates)
    zscore = pd.Series([0.0, 2.2, 1.0, 0.4, 0.0], index=dates)
    result = simulate_stat_arb_strategy(
        price_a,
        price_b,
        zscore,
        entry_threshold=2.0,
        exit_threshold=0.5,
        stop_loss_threshold=3.5,
    )
    assert result["signals"].iloc[1] == "enter_short_spread"
    assert result["pnl_path"].iloc[1] == 0.0
    assert result["pnl_path"].iloc[2] == pytest.approx(2.0)
    assert result["pnl_path"].iloc[3] == pytest.approx(6.0)
    assert result["completed_trade_count"] == 1
    assert result["active_trade_count"] == 0
    assert result["win_rate"] == 1.0
    assert result["total_pnl"] == pytest.approx(8.0)
    assert result["input_shape"] == [5, 3]


def test_strategy_uses_prior_dynamic_hedge_ratio_for_interval_pnl():
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    price_a = pd.Series([120.0, 130.0, 128.0, 122.0, 120.0], index=dates)
    price_b = pd.Series([100.0, 101.0, 103.0, 104.0, 105.0], index=dates)
    zscore = pd.Series([0.0, 2.2, 1.0, 0.4, 0.0], index=dates)
    zscore.attrs["hedge_ratios"] = pd.Series([1.0, 1.5, 2.0, 2.0, 2.0], index=dates)
    result = simulate_stat_arb_strategy(price_a, price_b, zscore)
    assert result["pnl_path"].iloc[2] == pytest.approx(5.0)
    assert result["pnl_path"].iloc[3] == pytest.approx(8.0)
    assert result["total_pnl"] == pytest.approx(13.0)


def test_strategy_supports_open_positions_and_reports_telemetry():
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    price_a = pd.Series([120.0, 125.0, 126.0, 127.0], index=dates)
    price_b = pd.Series([100.0, 101.0, 99.0, 102.0], index=dates)
    zscore = pd.Series([0.0, 2.1, 2.2, 2.3], index=dates)
    result = simulate_stat_arb_strategy(price_a, price_b, zscore)
    assert result["active_trade_count"] == 1
    assert result["trade_count"] == 1
    assert result["trade_log"].iloc[0]["status"] == "open"
    assert cointegration_telemetry(price_a, price_b)["input_shape"] == [4, 2]


def test_invalid_inputs_raise_clear_errors():
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    a = pd.Series([10.0, 11.0, 12.0, 13.0], index=dates)
    b = pd.Series([20.0, 21.0, 22.0, 23.0], index=dates)
    with pytest.raises(ValueError, match="window"):
        calculate_spread_zscore(a, b, 1.0, window=1)
    with pytest.raises(ValueError, match="strictly positive"):
        run_engle_granger_test(a, pd.Series([20.0, 0.0, 22.0, 23.0], index=dates))
    with pytest.raises(ValueError, match="thresholds"):
        simulate_stat_arb_strategy(a, b, pd.Series([0.0, 2.0, 0.0, 0.0], index=dates), exit_threshold=2.0)


def test_stop_exit_does_not_reopen_on_the_same_close():
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    price_a = pd.Series([120.0, 130.0, 140.0, 145.0], index=dates)
    price_b = pd.Series([100.0, 101.0, 100.0, 102.0], index=dates)
    zscore = pd.Series([0.0, 2.1, 3.6, 2.2], index=dates)
    result = simulate_stat_arb_strategy(price_a, price_b, zscore)
    assert result["signals"].iloc[2] == "exit_stop_loss"
    assert result["positions"].iloc[2] == 0
    assert result["signals"].iloc[3] == "enter_short_spread"
