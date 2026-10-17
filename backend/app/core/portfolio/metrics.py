"""Pure, database-independent performance metrics for daily simple returns.

Inputs are one-dimensional daily simple returns in decimal units (``0.01`` is
1%). Missing observations are omitted; non-finite values and returns below
-100% are rejected. Risk-free rates are annual decimals and are converted to
daily rates by dividing by ``trading_days``.
"""

from __future__ import annotations

import math
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd


def _validate_trading_days(trading_days: int) -> None:
    r"""Require a positive whole-number annualization period count.

    Annualization scales a daily statistic by :math:`D` or :math:`\sqrt D`,
    so :math:`D` must be a positive integer. Rejecting booleans avoids Python's
    surprising treatment of ``True`` as the integer one.

    Complexity: :math:`O(1)` time and :math:`O(1)` space.
    """
    if (
        isinstance(trading_days, bool)
        or not isinstance(trading_days, (int, np.integer))
        or trading_days <= 0
    ):
        raise ValueError("trading_days must be a positive integer")


def _coerce_returns(returns: pd.Series | np.ndarray) -> np.ndarray:
    r"""Validate and extract observed one-dimensional simple returns.

    Observations must satisfy :math:`-1\le r_t<\infty`; a return below
    :math:`-100\%` is impossible for a fully invested long position. NaNs are
    omitted so partial data can still be analyzed, while infinities are rejected
    instead of silently biasing risk estimates.

    Complexity: :math:`O(n)` time and :math:`O(n)` space for ``n`` observations.
    """
    if not isinstance(returns, (pd.Series, np.ndarray)):
        raise TypeError("returns must be a pandas Series or NumPy ndarray")
    try:
        values = returns.to_numpy(dtype=float, na_value=np.nan, copy=True) if isinstance(returns, pd.Series) else np.asarray(returns, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("returns must contain numeric values") from exc
    if values.ndim != 1:
        raise ValueError("returns must be one-dimensional")
    if np.isinf(values).any():
        raise ValueError("returns must be finite wherever present")
    observed = values[~np.isnan(values)]
    if np.any(observed < -1.0):
        raise ValueError("simple returns cannot be less than -100%")
    return observed


def _annual_risk_free_rate(risk_free_rate: float) -> float:
    r"""Validate the annual risk-free rate and return it as a finite float.

    The input is an annual decimal rate :math:`R_f`; metric calculations use
    :math:`r_f=R_f/D` per day. Keeping this conversion in one helper ensures
    Sharpe calculations interpret a stated annual rate consistently.

    Complexity: :math:`O(1)` time and :math:`O(1)` space.
    """
    try:
        rate = float(risk_free_rate)
    except (TypeError, ValueError) as exc:
        raise TypeError("risk_free_rate must be numeric") from exc
    if not math.isfinite(rate):
        raise ValueError("risk_free_rate must be finite")
    return rate


def annualized_volatility(
    returns: pd.Series | np.ndarray, trading_days: int = 252
) -> float:
    r"""Compute annualized sample volatility from daily simple returns.

    For daily returns :math:`r_t`,
    :math:`\sigma_{ann}=s_r\sqrt D`, where :math:`s_r` is the sample standard
    deviation (``ddof=1``) and :math:`D` is ``trading_days``. The square-root
    scaling reflects variance adding across independent periods. For example,
    doubling every daily return doubles volatility, while annualization scales
    daily volatility by :math:`\sqrt{252}` by default.

    At least two observed returns are needed to estimate sample volatility.
    Complexity: :math:`O(n)` time and :math:`O(n)` temporary space.
    """
    _validate_trading_days(trading_days)
    values = _coerce_returns(returns)
    if values.size < 2:
        raise ValueError("at least two non-missing returns are required to estimate volatility")
    return float(np.std(values, ddof=1) * np.sqrt(trading_days))


def sharpe_ratio(
    returns: pd.Series | np.ndarray,
    risk_free_rate: float = 0.03,
    trading_days: int = 252,
) -> float:
    r"""Compute the annualized Sharpe ratio from daily simple returns.

    The daily convention is
    :math:`S=\frac{\bar r-r_f/D}{s_r}\sqrt D`, where :math:`\bar r` is the
    arithmetic daily mean, :math:`r_f` is the annual risk-free rate, and
    :math:`s_r` is sample daily volatility. This measures excess return per
    unit of total risk. For intuition, a 12% annual return at 8% volatility
    with a 3% risk-free rate has Sharpe :math:`(12-3)/8=1.125`; 15% return at
    25% volatility has Sharpe :math:`(15-3)/25=0.48`, despite its higher
    headline return. If volatility is zero, the ratio is undefined and NaN is
    returned, including for a flat-price series.

    Complexity: :math:`O(n)` time and :math:`O(n)` temporary space.
    """
    _validate_trading_days(trading_days)
    risk_free_rate = _annual_risk_free_rate(risk_free_rate)
    values = _coerce_returns(returns)
    if values.size < 2:
        raise ValueError("at least two non-missing returns are required to estimate Sharpe ratio")
    daily_std = float(np.std(values, ddof=1))
    if daily_std == 0.0:
        return float("nan")
    daily_excess_mean = float(np.mean(values) - risk_free_rate / trading_days)
    return float((daily_excess_mean / daily_std) * np.sqrt(trading_days))


def annualized_return(
    returns: pd.Series | np.ndarray, trading_days: int = 252
) -> float:
    r"""Compute compound annual growth rate (CAGR) from daily simple returns.

    For :math:`n` observed daily returns,
    :math:`R_{ann}=\left(\prod_{t=1}^{n}(1+r_t)\right)^{D/n}-1`. Compounding
    reflects how wealth actually grows; it differs from simply multiplying
    the average daily return by :math:`D`. The calculation uses summed
    :math:`\log(1+r_t)` terms for numerical stability. If wealth reaches zero
    because any period return is -100%, CAGR is -100%.

    Complexity: :math:`O(n)` time and :math:`O(n)` temporary space.
    """
    _validate_trading_days(trading_days)
    values = _coerce_returns(returns)
    if values.size == 0:
        raise ValueError("at least one non-missing return is required to estimate annualized return")
    if np.any(values == -1.0):
        return -1.0
    mean_log_growth = float(np.log1p(values).sum() / values.size)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        result = np.expm1(mean_log_growth * trading_days)
    return float(result)


def daily_return_statistics(
    returns: pd.Series | np.ndarray, trading_days: int = 252
) -> dict[str, float | int]:
    r"""Summarize daily return inputs and their annualized intermediate metrics.

    The summary reports :math:`n`, :math:`\bar r`, sample standard deviation
    :math:`s_r`, and annualized volatility :math:`s_r\sqrt D`. These daily
    values make annualized outputs auditable and help distinguish missing data
    from a genuinely low-risk return stream. At least two observations are
    required for the standard deviation.

    Complexity: :math:`O(n)` time and :math:`O(n)` temporary space.
    """
    _validate_trading_days(trading_days)
    values = _coerce_returns(returns)
    if values.size < 2:
        raise ValueError("at least two non-missing returns are required for daily statistics")
    daily_mean = float(np.mean(values))
    daily_std = float(np.std(values, ddof=1))
    return {
        "observations": int(values.size),
        "daily_mean": daily_mean,
        "daily_std": daily_std,
        "annualized_volatility": float(daily_std * np.sqrt(trading_days)),
    }


def performance_metrics_with_telemetry(
    returns: pd.Series | np.ndarray,
    risk_free_rate: float = 0.03,
    trading_days: int = 252,
) -> tuple[dict[str, float], dict[str, Any]]:
    r"""Calculate core performance metrics with latency and daily diagnostics.

    Returns Sharpe ratio, annualized volatility and CAGR, plus telemetry for
    :math:`n`, daily mean, sample daily standard deviation, daily risk-free
    rate, daily excess mean, and elapsed milliseconds
    :math:`1000(t_{end}-t_{start})`. For example, the telemetry reveals the
    daily quantities underlying an annual Sharpe value, making the scaling
    inspectable by an API or UI without introducing I/O in this module.

    Complexity: :math:`O(n)` time and :math:`O(n)` temporary space.
    """
    _validate_trading_days(trading_days)
    risk_free_rate = _annual_risk_free_rate(risk_free_rate)
    values = _coerce_returns(returns)
    start = perf_counter()
    metrics = {
        "sharpe_ratio": sharpe_ratio(values, risk_free_rate, trading_days),
        "annualized_volatility": annualized_volatility(values, trading_days),
        "annualized_return": annualized_return(values, trading_days),
    }
    elapsed_ms = (perf_counter() - start) * 1000.0
    daily_stats = daily_return_statistics(values, trading_days)
    return metrics, {
        "latency_ms": elapsed_ms,
        **daily_stats,
        "daily_risk_free_rate": risk_free_rate / trading_days,
        "daily_excess_mean": daily_stats["daily_mean"] - risk_free_rate / trading_days,
        "input_shape": [int(values.size)],
    }
