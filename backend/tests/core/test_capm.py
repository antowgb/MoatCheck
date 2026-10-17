"""Tests for pure CAPM beta, alpha, and regression diagnostics."""

import numpy as np
import pandas as pd
import pytest

from app.core.research.capm import (
    calculate_beta,
    calculate_capm_alpha,
    capm_telemetry,
    estimate_capm_metrics,
)


def test_asset_identical_to_market_has_beta_one_and_zero_alpha() -> None:
    market = pd.Series([-0.03, -0.01, 0.005, 0.02, 0.04], index=list("abcde"))

    assert calculate_beta(market, market) == pytest.approx(1.0)
    assert calculate_capm_alpha(market, market) == pytest.approx(0.0, abs=1e-12)
    metrics = estimate_capm_metrics(market, market)
    assert metrics["beta"] == pytest.approx(1.0)
    assert metrics["alpha"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["r_squared"] == pytest.approx(1.0)
    assert metrics["residual_volatility"] == pytest.approx(0.0, abs=1e-12)


def test_uncorrelated_asset_has_zero_beta() -> None:
    market = pd.Series([-0.02, -0.01, 0.01, 0.02])
    asset = pd.Series([0.02, -0.02, -0.02, 0.02])

    assert calculate_beta(asset, market) == pytest.approx(0.0, abs=1e-14)


def test_beta_and_regression_pairwise_drop_nan_and_align_indexes() -> None:
    market = pd.Series([-0.02, -0.01, 0.01, 0.02, 0.03], index=list("abcde"))
    asset = pd.Series([0.0, -0.01, np.nan, 0.02, 0.035], index=list("abcde"))
    expected_market = market.loc[["a", "b", "d", "e"]].to_numpy()
    expected_asset = asset.loc[["a", "b", "d", "e"]].to_numpy()
    expected_beta = np.cov(expected_asset, expected_market, ddof=1)[0, 1] / np.var(
        expected_market, ddof=1
    )

    assert calculate_beta(asset, market) == pytest.approx(expected_beta)

    reindexed_asset = pd.Series([0.02, 0.035, 0.0, -0.01], index=["d", "e", "a", "b"])
    assert calculate_beta(reindexed_asset, market) == pytest.approx(expected_beta)
    assert estimate_capm_metrics(asset, market)["aligned_observations"] == 4


def test_alpha_matches_manual_annualized_jensen_intercept() -> None:
    market = np.array([-0.03, -0.02, -0.01, 0.01, 0.02, 0.03])
    daily_rf = 0.04 / 252
    beta = 1.25
    daily_alpha = 0.0004
    asset = daily_rf + daily_alpha + beta * (market - daily_rf)
    expected = daily_alpha * 252

    result = calculate_capm_alpha(pd.Series(asset), pd.Series(market), risk_free_rate=0.04)

    assert result == pytest.approx(expected)


def test_alpha_changes_consistently_when_risk_free_rate_changes() -> None:
    market = np.array([-0.02, -0.01, 0.0, 0.01, 0.02])
    asset = 0.0005 + 0.7 * market
    market_series = pd.Series(market)
    asset_series = pd.Series(asset)

    alpha_at_zero_rf = calculate_capm_alpha(asset_series, market_series, risk_free_rate=0.0)
    alpha_at_six_percent = calculate_capm_alpha(asset_series, market_series, risk_free_rate=0.06)

    # The fitted intercept shifts by (beta - 1) * change_in_daily_rf.
    assert alpha_at_six_percent - alpha_at_zero_rf == pytest.approx((0.7 - 1.0) * 0.06)


def test_metrics_include_fit_quality_residual_risk_and_telemetry() -> None:
    market = pd.Series([-0.04, -0.02, -0.01, 0.01, 0.02, 0.04])
    asset = pd.Series(0.0002 + 1.2 * market + np.array([0.001, -0.001, 0.0, 0.0, 0.001, -0.001]))

    metrics = estimate_capm_metrics(asset, market)

    assert metrics["beta"] == pytest.approx(1.2, abs=0.02)
    assert 0.0 <= metrics["r_squared"] <= 1.0
    assert 0.0 <= metrics["p_value"] <= 1.0
    assert metrics["residual_volatility"] >= 0.0
    assert metrics["latency_ms"] >= 0.0
    assert metrics["asset_observations"] == len(asset)
    assert metrics["market_observations"] == len(market)
    assert metrics["aligned_observations"] == len(asset)
    assert metrics["input_shape"] == {"asset": [len(asset)], "market": [len(market)]}


def test_telemetry_can_be_requested_on_its_own() -> None:
    asset = pd.Series([0.01, np.nan, 0.02], index=[1, 2, 3])
    market = pd.Series([0.01, 0.02, 0.03], index=[1, 2, 3])

    telemetry = capm_telemetry(asset, market)

    assert telemetry["latency_ms"] >= 0.0
    assert telemetry["asset_observations"] == 2
    assert telemetry["market_observations"] == 3
    assert telemetry["aligned_observations"] == 2


@pytest.mark.parametrize(
    ("asset", "market", "message"),
    [
        (pd.Series([], dtype=float), pd.Series([], dtype=float), "two aligned"),
        (pd.Series([0.01, 0.02]), pd.Series([0.01, 0.01]), "non-zero variance"),
        (pd.Series([0.01, np.inf]), pd.Series([0.01, 0.02]), "finite"),
        (pd.Series([-1.01, 0.02]), pd.Series([0.01, 0.02]), "less than -100%"),
    ],
)
def test_invalid_or_undefined_regressions_raise(
    asset: pd.Series, market: pd.Series, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        calculate_beta(asset, market)


def test_non_series_and_duplicate_index_inputs_are_rejected() -> None:
    with pytest.raises(TypeError, match="pandas Series"):
        calculate_beta(np.array([0.01, 0.02]), pd.Series([0.01, 0.02]))  # type: ignore[arg-type]
    duplicate_index = pd.Series([0.01, 0.02, 0.03], index=[1, 1, 2])
    with pytest.raises(ValueError, match="index must be unique"):
        calculate_beta(duplicate_index, pd.Series([0.01, 0.02, 0.03], index=[1, 2, 3]))


def test_risk_free_rate_and_trading_days_are_validated() -> None:
    asset = pd.Series([0.01, -0.01, 0.02])
    market = pd.Series([-0.02, 0.01, 0.03])

    with pytest.raises(ValueError, match="finite"):
        calculate_capm_alpha(asset, market, risk_free_rate=float("nan"))
    with pytest.raises(ValueError, match="positive integer"):
        calculate_capm_alpha(asset, market, trading_days=0)
