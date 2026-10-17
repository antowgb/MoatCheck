"""Pure CAPM and market-beta calculations for aligned daily return series.

Inputs are daily simple returns in decimal units (``0.01`` means 1%). Missing
observations are excluded pairwise after aligning the two Pandas indexes. The
annual risk-free rate is an arithmetic annual decimal rate.
"""

from __future__ import annotations

import math
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import linregress


def _validate_risk_free_rate(risk_free_rate: float) -> float:
    r"""Validate and convert the annual risk-free rate to a finite float.

    The CAPM calculations use :math:`r_{f,d}=R_f/D`, so rejecting NaN and
    infinity here prevents invalid expected returns from contaminating the
    regression. Complexity: :math:`O(1)` time and :math:`O(1)` space.
    """
    try:
        rate = float(risk_free_rate)
    except (TypeError, ValueError) as exc:
        raise TypeError("risk_free_rate must be numeric") from exc
    if not math.isfinite(rate):
        raise ValueError("risk_free_rate must be finite")
    return rate


def _aligned_returns(
    asset_returns: pd.Series, market_returns: pd.Series
) -> tuple[np.ndarray, np.ndarray]:
    r"""Validate and pairwise-align observed asset and market returns.

    Observations are matched by Pandas index, and any row containing NaN is
    dropped pairwise. Values must be finite and represent valid simple returns
    (:math:`r\ge-1`). Duplicate indexes are rejected because they make the
    date-to-date pairing ambiguous. Complexity: :math:`O(n\log n)` time for
    index alignment and :math:`O(n)` temporary space.
    """
    if not isinstance(asset_returns, pd.Series) or not isinstance(market_returns, pd.Series):
        raise TypeError("asset_returns and market_returns must be pandas Series")
    for name, series in (("asset_returns", asset_returns), ("market_returns", market_returns)):
        if not series.index.is_unique:
            raise ValueError(f"{name} index must be unique for unambiguous alignment")
        try:
            source_values = series.to_numpy(dtype=float, na_value=np.nan)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} must contain numeric values") from exc
        observed = source_values[~np.isnan(source_values)]
        if not np.isfinite(observed).all():
            raise ValueError("returns must be finite wherever present")
        if np.any(observed < -1.0):
            raise ValueError("simple returns cannot be less than -100%")

    aligned = pd.concat(
        [asset_returns.rename("asset"), market_returns.rename("market")],
        axis=1,
        join="inner",
    ).dropna()
    try:
        values = aligned.to_numpy(dtype=float, copy=True)
    except (TypeError, ValueError) as exc:
        raise TypeError("return series must contain numeric values") from exc
    if not np.isfinite(values).all():
        raise ValueError("returns must be finite wherever present")
    if np.any(values < -1.0):
        raise ValueError("simple returns cannot be less than -100%")
    if values.shape[0] < 2:
        raise ValueError("at least two aligned, non-missing observations are required")
    return values[:, 0], values[:, 1]


def _regression(
    asset_returns: pd.Series,
    market_returns: pd.Series,
    risk_free_rate: float,
    trading_days: int,
):
    r"""Fit the daily excess-return CAPM using a one-predictor OLS regression.

    :math:`R_{i,t}-r_f=\alpha_d+\beta(R_{m,t}-r_f)+\epsilon_t`. A defensive
    check rejects a market series with no variance, for which beta is
    undefined. Complexity: :math:`O(n)` time and :math:`O(n)` space.
    """
    if isinstance(trading_days, bool) or not isinstance(trading_days, (int, np.integer)) or trading_days <= 0:
        raise ValueError("trading_days must be a positive integer")
    asset, market = _aligned_returns(asset_returns, market_returns)
    daily_rf = risk_free_rate / int(trading_days)
    market_excess = market - daily_rf
    asset_excess = asset - daily_rf
    if float(np.var(market_excess, ddof=1)) == 0.0:
        raise ValueError("market returns must have non-zero variance to estimate beta")
    regression = linregress(market_excess, asset_excess)
    return asset, market, regression


def calculate_beta(asset_returns: pd.Series, market_returns: pd.Series) -> float:
    r"""Estimate asset market beta as covariance divided by market variance.

    The formula is :math:`\beta_i=\frac{\operatorname{Cov}(R_i,R_m)}
    {\operatorname{Var}(R_m)}` using aligned daily simple returns and sample
    covariance. Subtracting the same constant risk-free return from both
    series leaves the slope unchanged. A beta of 1.2 means a Tech stock has
    historically moved about 1.2% for a 1% market move; a defensive stock at
    0.6 has moved about 0.6% for the same market move. Complexity:
    :math:`O(n)` time and :math:`O(n)` space.
    """
    asset, market = _aligned_returns(asset_returns, market_returns)
    market_variance = float(np.var(market, ddof=1))
    if market_variance == 0.0:
        raise ValueError("market returns must have non-zero variance to estimate beta")
    return float(np.cov(asset, market, ddof=1)[0, 1] / market_variance)


def calculate_capm_alpha(
    asset_returns: pd.Series,
    market_returns: pd.Series,
    risk_free_rate: float = 0.03,
    trading_days: int = 252,
) -> float:
    r"""Calculate annualized Jensen alpha from daily CAPM excess returns.

    The daily intercept is :math:`\alpha_d=\overline{(R_i-r_f)-\beta(R_m-r_f)}`;
    this function annualizes it arithmetically as
    :math:`\alpha_{ann}=D\alpha_d`. It measures average return beyond the
    CAPM-predicted return after accounting for market exposure. For example,
    if a Tech stock with :math:`\beta=1.2` earns more than the risk-free rate
    plus 1.2 times the market premium, the excess is positive Jensen alpha;
    a defensive stock with :math:`\beta=0.6` has a lower market hurdle. The
    daily intercept convention matches the requested annualization; compounding
    an estimated daily intercept would answer a different question.
    Complexity: :math:`O(n)` time and :math:`O(n)` space.
    """
    rate = _validate_risk_free_rate(risk_free_rate)
    _, _, regression = _regression(asset_returns, market_returns, rate, trading_days)
    return float(regression.intercept * int(trading_days))


def estimate_capm_metrics(
    asset_returns: pd.Series,
    market_returns: pd.Series,
    risk_free_rate: float = 0.03,
) -> dict[str, float | int | dict[str, int | list[int]]]:
    r"""Return CAPM beta, annual Jensen alpha, fit statistics, and telemetry.

    The fitted relationship is :math:`R_i-r_f=\alpha_d+\beta(R_m-r_f)+\epsilon`.
    The result contains annualized alpha, beta, regression :math:`R^2`, slope
    p-value, and annualized sample residual volatility. It also reports elapsed
    milliseconds and original/aligned series lengths so callers can expose the
    data coverage behind an estimate. A beta of 1.2 describes a more market
    sensitive Tech stock; beta 0.6 describes a defensive stock. Complexity:
    :math:`O(n)` time and :math:`O(n)` space for aligned observations.
    """
    start = perf_counter()
    rate = _validate_risk_free_rate(risk_free_rate)
    asset, market, regression = _regression(asset_returns, market_returns, rate, 252)
    residuals = (asset - rate / 252) - (
        regression.intercept + regression.slope * (market - rate / 252)
    )
    residual_volatility = float(np.std(residuals, ddof=1) * np.sqrt(252))
    telemetry = capm_telemetry(asset_returns, market_returns, len(asset))
    telemetry["latency_ms"] = (perf_counter() - start) * 1000.0
    return {
        "beta": float(regression.slope),
        "alpha": float(regression.intercept * 252),
        "r_squared": float(regression.rvalue**2),
        "p_value": float(regression.pvalue),
        "residual_volatility": residual_volatility,
        **telemetry,
    }


def capm_telemetry(
    asset_returns: pd.Series,
    market_returns: pd.Series,
    aligned_observations: int | None = None,
) -> dict[str, float | int | dict[str, int | list[int]]]:
    r"""Measure CAPM input shape and execution latency without external I/O.

    The returned diagnostics include the input lengths, aligned observation
    count :math:`n=|I_a\cap I_m|` after dropping missing pairs, input shapes,
    and time spent producing the summary in milliseconds
    :math:`1000(t_{end}-t_{start})`.
    In ``estimate_capm_metrics`` latency is measured around the regression and
    diagnostics work as a whole. This standalone helper provides shape details
    for a caller that only needs an input audit. Index intersection takes
    :math:`O(n)` time and :math:`O(n)` temporary space.
    """
    start = perf_counter()
    if aligned_observations is None:
        aligned = pd.concat(
            [asset_returns.rename("asset"), market_returns.rename("market")],
            axis=1,
            join="inner",
        ).dropna()
        aligned_count = int(len(aligned))
    else:
        aligned_count = int(aligned_observations)
    elapsed_ms = (perf_counter() - start) * 1000.0
    return {
        "latency_ms": elapsed_ms,
        "asset_observations": int(asset_returns.notna().sum()),
        "market_observations": int(market_returns.notna().sum()),
        "aligned_observations": aligned_count,
        "input_shape": {
            "asset": [int(len(asset_returns))],
            "market": [int(len(market_returns))],
        },
    }
