"""Pure OLS estimation and style attribution for Fama-French factor models.

Portfolio and factor returns are periodic simple returns in decimal units,
indexed by observation date. Inputs are aligned in memory and the annual
risk-free rate is converted to a daily rate using 252 trading days.
"""

from __future__ import annotations

import math
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

_TRADING_DAYS = 252
_THREE_FACTORS = ("MKT", "SMB", "HML")
_FIVE_FACTORS = (*_THREE_FACTORS, "RMW", "CMA")


def _validate_risk_free_rate(risk_free_rate: float) -> float:
    r"""Validate the annual simple risk-free rate.

    The annual input :math:`R_f` is expressed as a decimal and the daily
    regression rate is :math:`r_{f,d}=R_f/252`; for example, 3% per year is
    approximately :math:`0.000119` per trading day. Complexity: :math:`O(1)`
    time and space.
    """
    try:
        rate = float(risk_free_rate)
    except (TypeError, ValueError) as exc:
        raise TypeError("risk_free_rate must be numeric") from exc
    if not math.isfinite(rate) or rate <= -1.0:
        raise ValueError("risk_free_rate must be finite and greater than -100%")
    return rate


def _aligned_model_data(
    portfolio_returns: pd.Series,
    factor_df: pd.DataFrame,
    factor_names: tuple[str, ...],
    risk_free_rate: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, list[int]]]:
    r"""Validate and pairwise-align a portfolio series and required factors.

    For each common date, this returns excess portfolio return
    :math:`y_t=R_{p,t}-R_f/252` and a row of the requested factors
    :math:`X_t`. For example, if one date lacks HML, that date is removed from
    both the response and every factor so OLS uses a consistent sample.
    Complexity: :math:`O(n\log n+nk)` time and :math:`O(nk)` space for
    ``n`` observations and ``k`` factors.
    """
    if not isinstance(portfolio_returns, pd.Series):
        raise TypeError("portfolio_returns must be a pandas Series")
    if not isinstance(factor_df, pd.DataFrame):
        raise TypeError("factor_df must be a pandas DataFrame")
    if not portfolio_returns.index.is_unique:
        raise ValueError("portfolio_returns index must be unique")
    if not factor_df.index.is_unique:
        raise ValueError("factor_df index must be unique")
    if not factor_df.columns.is_unique:
        raise ValueError("factor_df columns must be unique")
    missing_factors = [name for name in factor_names if name not in factor_df.columns]
    if missing_factors:
        raise ValueError(f"factor_df is missing required factors: {missing_factors}")

    try:
        portfolio_values = portfolio_returns.to_numpy(dtype=float, na_value=np.nan)
        factor_values = factor_df.loc[:, list(factor_names)].to_numpy(
            dtype=float, na_value=np.nan
        )
    except (TypeError, ValueError) as exc:
        raise TypeError("portfolio returns and factors must be numeric") from exc

    if np.isinf(portfolio_values).any() or np.isinf(factor_values).any():
        raise ValueError("portfolio returns and factors must be finite wherever present")
    observed_portfolio = portfolio_values[~np.isnan(portfolio_values)]
    if np.any(observed_portfolio < -1.0):
        raise ValueError("portfolio simple returns cannot be less than -100%")

    aligned = pd.concat(
        [portfolio_returns.rename("portfolio"), factor_df.loc[:, list(factor_names)]],
        axis=1,
        join="inner",
    ).dropna(subset=["portfolio", *factor_names])
    if aligned.empty:
        raise ValueError("at least one aligned, non-missing observation is required")
    values = aligned.to_numpy(dtype=float, copy=True)
    daily_risk_free = risk_free_rate / _TRADING_DAYS
    y = values[:, 0] - daily_risk_free
    x = values[:, 1:]
    input_shapes = {
        "portfolio_returns": [int(len(portfolio_returns))],
        "factor_df": [int(factor_df.shape[0]), int(factor_df.shape[1])],
        "aligned_design": [int(x.shape[0]), int(x.shape[1])],
    }
    return y, x, input_shapes


def _fit_fama_french(
    portfolio_returns: pd.Series,
    factor_df: pd.DataFrame,
    risk_free_rate: float,
    factor_names: tuple[str, ...],
    model_name: str,
) -> dict[str, Any]:
    r"""Fit a full-rank OLS design and package model diagnostics.

    OLS estimates :math:`\hat\theta=(X^TX)^{-1}X^Ty`, where the design
    includes an intercept and the response is excess portfolio return. For
    instance, a 1% alpha plus a 7% factor contribution gives an 8% modeled
    excess return before residual noise. A unique solution requires more
    observations than coefficients and a full-rank design. Complexity:
    :math:`O(nk^2+k^3)` time and :math:`O(nk+k^2)` space.
    """
    start = perf_counter()
    rate = _validate_risk_free_rate(risk_free_rate)
    y, x_values, input_shapes = _aligned_model_data(
        portfolio_returns, factor_df, factor_names, rate
    )
    coefficient_count = len(factor_names) + 1
    if y.size <= coefficient_count:
        raise ValueError(
            f"at least {coefficient_count + 1} aligned observations are required "
            "to estimate coefficients and residual variance"
        )
    design = np.column_stack((np.ones(y.size, dtype=float), x_values))
    if np.linalg.matrix_rank(design) != coefficient_count:
        raise ValueError("factor design matrix is rank-deficient; OLS coefficients are not identifiable")

    fitted = sm.OLS(y, design, hasconst=True, missing="raise").fit()
    residual_variance = max(float(fitted.mse_resid), 0.0)
    daily_residual_volatility = math.sqrt(residual_variance)
    daily_alpha = float(fitted.params[0])
    factor_betas = {
        name: float(value)
        for name, value in zip(factor_names, fitted.params[1:], strict=True)
    }
    return {
        "model": model_name,
        "alpha": daily_alpha * _TRADING_DAYS,
        "alpha_daily": daily_alpha,
        "alpha_annualized": daily_alpha * _TRADING_DAYS,
        "factor_betas": factor_betas,
        "r_squared": float(fitted.rsquared),
        "adjusted_r_squared": float(fitted.rsquared_adj),
        "residual_volatility_daily": daily_residual_volatility,
        "residual_volatility": daily_residual_volatility * math.sqrt(_TRADING_DAYS),
        "n_observations": int(fitted.nobs),
        "degrees_of_freedom": int(fitted.df_resid),
        "latency_ms": float((perf_counter() - start) * 1000.0),
        "input_shape": input_shapes,
    }


def fit_fama_french_3factor(
    portfolio_returns: pd.Series,
    factor_df: pd.DataFrame,
    risk_free_rate: float = 0.03,
) -> dict[str, Any]:
    r"""Fit the Fama-French three-factor OLS model to daily returns.

    The regression is :math:`R_{p,t}-R_f=\alpha+\beta_m MKT_t+s SMB_t+
    h HML_t+\epsilon_t`. The annual risk-free input is divided by 252 and
    factors are expected in the same decimal units as portfolio returns. A
    portfolio with 1% annualized pure alpha and 7% annualized factor tilts has
    modeled excess performance of 8% before residual noise. The returned
    diagnostics include R-squared, adjusted R-squared, annualized residual
    volatility, and OLS latency. Complexity: :math:`O(nk^2+k^3)` time and
    :math:`O(nk+k^2)` space, with :math:`k=3` factors.
    """
    return _fit_fama_french(
        portfolio_returns,
        factor_df,
        risk_free_rate,
        _THREE_FACTORS,
        "Fama-French 3-Factor",
    )


def fit_fama_french_5factor(
    portfolio_returns: pd.Series,
    factor_df: pd.DataFrame,
    risk_free_rate: float = 0.03,
) -> dict[str, Any]:
    r"""Fit the Fama-French five-factor model to daily portfolio returns.

    The model is :math:`R_i-R_f=\alpha+\beta_m(MKT)+s(SMB)+h(HML)+
    r(RMW)+c(CMA)+e`. Market, size, value, profitability, and investment
    exposures separate common style tilts from residual alpha. For example,
    if factor tilts explain 7% annualized return and the intercept adds 1%,
    total modeled excess performance is 8%. This includes adjusted
    R-squared, annualized residual volatility, sample diagnostics, and latency.
    Complexity: :math:`O(nk^2+k^3)` time and :math:`O(nk+k^2)` space, with
    :math:`k=5` factors.
    """
    return _fit_fama_french(
        portfolio_returns,
        factor_df,
        risk_free_rate,
        _FIVE_FACTORS,
        "Fama-French 5-Factor",
    )


def decompose_style_attribution(
    factor_betas: dict[str, float], factor_returns: pd.Series
) -> pd.Series:
    r"""Multiply factor exposures by factor returns for exact attribution.

    For factor :math:`j`, attribution is :math:`A_j=\beta_jF_j`; summing
    :math:`A_j` gives the modeled factor contribution in the same decimal
    return units as ``factor_returns``. For example, :math:`\beta_{MKT}=1`
    and a 7% market-factor return contribute 7%; a separate 1% alpha remains
    the regression intercept and is not included in the factor series. Factor
    keys must match exactly to prevent silent omission. Complexity: :math:`O(k)`
    time and output space for :math:`k` factors.
    """
    if not isinstance(factor_betas, dict):
        raise TypeError("factor_betas must be a dictionary")
    if not isinstance(factor_returns, pd.Series):
        raise TypeError("factor_returns must be a pandas Series")
    if not factor_returns.index.is_unique:
        raise ValueError("factor_returns index must be unique")
    if set(factor_betas) != set(factor_returns.index):
        raise ValueError("factor_betas keys must exactly match factor_returns index")
    try:
        betas = np.asarray([factor_betas[name] for name in factor_returns.index], dtype=float)
        returns = factor_returns.to_numpy(dtype=float, na_value=np.nan)
    except (TypeError, ValueError) as exc:
        raise TypeError("factor betas and returns must be numeric") from exc
    if not np.isfinite(betas).all() or not np.isfinite(returns).all():
        raise ValueError("factor betas and returns must contain only finite values")
    attribution = betas * returns
    return pd.Series(attribution, index=factor_returns.index, name="return_attribution")


def fama_french_telemetry(
    portfolio_returns: pd.Series,
    factor_df: pd.DataFrame,
    factor_names: tuple[str, ...] = _FIVE_FACTORS,
    risk_free_rate: float = 0.03,
) -> dict[str, int | float | list[int]]:
    r"""Report aligned observation coverage, input shapes, and processing time.

    The coverage ratio is :math:`n_{aligned}/n_{portfolio}` after intersecting
    dates and dropping incomplete factor rows. For example, 90 aligned points
    from 100 portfolio observations yield 0.90 coverage. This helper performs
    only in-memory validation and counting. Complexity: :math:`O(n\log n+nk)`
    time and :math:`O(nk)` space.
    """
    start = perf_counter()
    rate = _validate_risk_free_rate(risk_free_rate)
    _, x, shapes = _aligned_model_data(
        portfolio_returns, factor_df, factor_names, rate
    )
    elapsed_ms = (perf_counter() - start) * 1000.0
    observations = int(x.shape[0])
    portfolio_count = int(len(portfolio_returns))
    return {
        "latency_ms": float(elapsed_ms),
        "n_observations": observations,
        "coverage_ratio": observations / portfolio_count if portfolio_count else 0.0,
        "input_shape": shapes,
    }
