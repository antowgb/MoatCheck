"""Tests for pure daily-return portfolio performance metrics."""

import numpy as np
import pandas as pd
import pytest

from app.core.portfolio.metrics import (
    annualized_return,
    annualized_volatility,
    daily_return_statistics,
    performance_metrics_with_telemetry,
    sharpe_ratio,
)


def test_empty_returns_are_rejected_with_clear_errors() -> None:
    empty = pd.Series([], dtype=float)

    with pytest.raises(ValueError, match="two non-missing returns"):
        sharpe_ratio(empty)
    with pytest.raises(ValueError, match="two non-missing returns"):
        annualized_volatility(empty)
    with pytest.raises(ValueError, match="one non-missing return"):
        annualized_return(empty)
    with pytest.raises(ValueError, match="two non-missing returns"):
        daily_return_statistics(empty)


def test_flat_returns_have_zero_volatility_and_undefined_sharpe() -> None:
    flat_prices_returns = pd.Series([0.0, 0.0, 0.0, 0.0])

    assert annualized_volatility(flat_prices_returns) == 0.0
    assert np.isnan(sharpe_ratio(flat_prices_returns))
    assert annualized_return(flat_prices_returns) == 0.0


def test_sharpe_is_zero_when_mean_return_matches_risk_free_rate() -> None:
    returns = pd.Series([0.01, 0.03, 0.02, 0.02])
    annual_risk_free_rate = float(returns.mean() * 252)

    assert sharpe_ratio(returns, risk_free_rate=annual_risk_free_rate) == pytest.approx(0.0, abs=1e-14)


def test_volatility_scales_linearly_and_sharpe_is_scale_invariant() -> None:
    returns = np.array([-0.01, 0.005, 0.02, -0.002, 0.01])
    risk_free_rate = 0.03
    scale = 4.0

    base_volatility = annualized_volatility(returns)
    assert annualized_volatility(returns * scale) == pytest.approx(base_volatility * scale)
    assert sharpe_ratio(returns * scale, risk_free_rate=risk_free_rate * scale) == pytest.approx(
        sharpe_ratio(returns, risk_free_rate=risk_free_rate)
    )


def test_annualized_metrics_match_manual_calculations() -> None:
    returns = np.array([0.01, -0.02, 0.03, 0.005])
    trading_days = 4
    risk_free_rate = 0.02
    daily_mean = returns.mean()
    daily_std = returns.std(ddof=1)
    expected_sharpe = (daily_mean - risk_free_rate / trading_days) / daily_std * np.sqrt(trading_days)
    expected_volatility = daily_std * np.sqrt(trading_days)
    expected_cagr = np.prod(1.0 + returns) ** (trading_days / len(returns)) - 1.0

    assert sharpe_ratio(returns, risk_free_rate, trading_days) == pytest.approx(expected_sharpe)
    assert annualized_volatility(returns, trading_days) == pytest.approx(expected_volatility)
    assert annualized_return(returns, trading_days) == pytest.approx(expected_cagr)


def test_annualized_return_handles_total_loss_and_nan_observations() -> None:
    assert annualized_return(np.array([0.1, -1.0, 0.5])) == -1.0
    assert annualized_return(pd.Series([0.1, np.nan, 0.1]), trading_days=2) == pytest.approx(0.21)


@pytest.mark.parametrize("returns", [[-1.01, 0.02], [0.01, np.inf], [[0.01, 0.02], [0.03, 0.04]]])
def test_invalid_return_inputs_raise(returns: list[float]) -> None:
    with pytest.raises(ValueError):
        annualized_volatility(np.asarray(returns))


def test_trading_days_and_risk_free_rate_are_validated() -> None:
    returns = np.array([0.01, 0.02])

    with pytest.raises(ValueError, match="positive integer"):
        annualized_volatility(returns, 0)
    with pytest.raises(ValueError, match="finite"):
        sharpe_ratio(returns, risk_free_rate=float("nan"))


def test_performance_telemetry_exposes_latency_and_daily_intermediates() -> None:
    returns = pd.Series([0.01, 0.02, -0.01, 0.03])

    metrics, telemetry = performance_metrics_with_telemetry(returns, risk_free_rate=0.04)

    assert set(metrics) == {"sharpe_ratio", "annualized_volatility", "annualized_return"}
    assert telemetry["latency_ms"] >= 0.0
    assert telemetry["observations"] == 4
    assert telemetry["daily_mean"] == pytest.approx(returns.mean())
    assert telemetry["daily_std"] == pytest.approx(returns.std(ddof=1))
    assert telemetry["daily_risk_free_rate"] == pytest.approx(0.04 / 252)
    assert telemetry["daily_excess_mean"] == pytest.approx(returns.mean() - 0.04 / 252)
    assert telemetry["input_shape"] == [4]
