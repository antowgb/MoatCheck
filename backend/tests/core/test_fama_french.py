"""Tests for pure Fama-French OLS estimation and style attribution."""

import numpy as np
import pandas as pd
import pytest

from app.core.research.fama_french import (
    decompose_style_attribution,
    fama_french_telemetry,
    fit_fama_french_3factor,
    fit_fama_french_5factor,
)

TRADING_DAYS = 252
RISK_FREE_RATE = 0.03
FACTORS_3 = ["MKT", "SMB", "HML"]
FACTORS_5 = [*FACTORS_3, "RMW", "CMA"]


def _orthogonal_factors(n: int, names: list[str]) -> pd.DataFrame:
    rng = np.random.default_rng(2026)
    centered = rng.normal(size=(n, len(names)))
    centered -= centered.mean(axis=0)
    orthogonal, _ = np.linalg.qr(centered)
    return pd.DataFrame(orthogonal * 0.01, columns=names)


def _portfolio_from_model(
    factors: pd.DataFrame,
    betas: list[float],
    annual_alpha: float = 0.01,
    risk_free_rate: float = RISK_FREE_RATE,
) -> pd.Series:
    daily_rf = risk_free_rate / TRADING_DAYS
    excess = annual_alpha / TRADING_DAYS + factors.to_numpy() @ np.asarray(betas)
    return pd.Series(daily_rf + excess, index=factors.index, name="portfolio")


def test_three_factor_recovers_perfect_market_exposure_and_alpha() -> None:
    factors = _orthogonal_factors(40, FACTORS_3)
    portfolio = _portfolio_from_model(factors, [1.5, 0.0, 0.0])

    result = fit_fama_french_3factor(portfolio, factors)

    assert result["factor_betas"]["MKT"] == pytest.approx(1.5)
    assert result["factor_betas"]["SMB"] == pytest.approx(0.0, abs=1e-12)
    assert result["factor_betas"]["HML"] == pytest.approx(0.0, abs=1e-12)
    assert result["alpha_annualized"] == pytest.approx(0.01)
    assert result["r_squared"] == pytest.approx(1.0)
    assert result["adjusted_r_squared"] == pytest.approx(1.0)
    assert result["residual_volatility"] == pytest.approx(0.0, abs=1e-12)


def test_five_factor_recovers_orthogonal_factor_loadings() -> None:
    factors = _orthogonal_factors(80, FACTORS_5)
    expected_betas = [1.2, 0.4, -0.3, 0.2, 0.5]
    portfolio = _portfolio_from_model(factors, expected_betas, annual_alpha=0.004)

    result = fit_fama_french_5factor(portfolio, factors)

    assert result["factor_betas"] == pytest.approx(dict(zip(FACTORS_5, expected_betas)))
    assert result["alpha_annualized"] == pytest.approx(0.004)
    assert result["n_observations"] == 80
    assert result["degrees_of_freedom"] == 74
    assert result["residual_volatility_daily"] == pytest.approx(0.0, abs=1e-12)
    assert result["latency_ms"] >= 0.0
    assert result["input_shape"]["aligned_design"] == [80, 5]


def test_missing_dates_and_nan_factor_rows_are_pairwise_aligned() -> None:
    factors = _orthogonal_factors(80, FACTORS_3)
    portfolio = _portfolio_from_model(factors, [0.8, 0.2, -0.1])
    portfolio = portfolio.drop(index=[0, 1])
    factors.loc[5, "SMB"] = np.nan

    result = fit_fama_french_3factor(portfolio, factors)

    assert result["n_observations"] == 77
    assert result["factor_betas"] == pytest.approx(
        {"MKT": 0.8, "SMB": 0.2, "HML": -0.1}
    )
    telemetry = fama_french_telemetry(portfolio, factors, tuple(FACTORS_3))
    assert telemetry["n_observations"] == 77
    assert telemetry["coverage_ratio"] == pytest.approx(77 / len(portfolio))
    assert telemetry["latency_ms"] >= 0.0


def test_zero_residual_variance_is_reported_as_zero() -> None:
    factors = _orthogonal_factors(30, FACTORS_3)
    portfolio = _portfolio_from_model(factors, [-0.5, 0.3, 1.1], annual_alpha=0.0)

    result = fit_fama_french_3factor(portfolio, factors)

    assert result["residual_volatility_daily"] == pytest.approx(0.0, abs=1e-12)
    assert result["residual_volatility"] == pytest.approx(0.0, abs=1e-12)
    assert result["r_squared"] == pytest.approx(1.0)


def test_style_attribution_multiplies_betas_by_factor_returns_exactly() -> None:
    betas = {"MKT": 1.0, "SMB": 0.5, "HML": -0.25}
    factor_returns = pd.Series({"MKT": 0.07, "SMB": 0.02, "HML": 0.04})

    attribution = decompose_style_attribution(betas, factor_returns)

    pd.testing.assert_series_equal(
        attribution,
        pd.Series(
            {"MKT": 0.07, "SMB": 0.01, "HML": -0.01},
            name="return_attribution",
        ),
    )
    assert attribution.sum() == pytest.approx(0.07)


@pytest.mark.parametrize(
    ("portfolio", "factors", "error", "message"),
    [
        (pd.Series([0.01]), pd.DataFrame(columns=FACTORS_3), ValueError, "aligned"),
        (
            pd.Series(np.arange(8, dtype=float)),
            pd.DataFrame({"MKT": np.arange(8), "SMB": np.arange(8), "HML": np.arange(8)}),
            ValueError,
            "rank-deficient",
        ),
        (
            pd.Series(np.arange(4, dtype=float)),
            pd.DataFrame({name: np.arange(4, dtype=float) for name in FACTORS_5}),
            ValueError,
            "at least 7 aligned",
        ),
    ],
)
def test_undefined_or_unidentifiable_models_raise(
    portfolio: pd.Series,
    factors: pd.DataFrame,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        if len(factors.columns) == 5:
            fit_fama_french_5factor(portfolio, factors)
        else:
            fit_fama_french_3factor(portfolio, factors)


def test_missing_factors_duplicate_indexes_and_invalid_values_raise() -> None:
    portfolio = pd.Series([0.01, 0.02, 0.03], index=[0, 1, 2])
    with pytest.raises(ValueError, match="missing required factors"):
        fit_fama_french_3factor(portfolio, pd.DataFrame(columns=["MKT"]))

    duplicate_portfolio = pd.Series([0.01, 0.02], index=[0, 0])
    factors = pd.DataFrame(np.ones((2, 3)), columns=FACTORS_3)
    with pytest.raises(ValueError, match="index must be unique"):
        fit_fama_french_3factor(duplicate_portfolio, factors)

    invalid = factors.copy()
    invalid.loc[0, "MKT"] = np.inf
    with pytest.raises(ValueError, match="finite"):
        fit_fama_french_3factor(pd.Series([0.01, 0.02]), invalid)


def test_attribution_rejects_mismatched_keys_and_nonfinite_inputs() -> None:
    with pytest.raises(ValueError, match="exactly match"):
        decompose_style_attribution({"MKT": 1.0}, pd.Series({"SMB": 0.02}))
    with pytest.raises(ValueError, match="finite"):
        decompose_style_attribution({"MKT": np.inf}, pd.Series({"MKT": 0.02}))
