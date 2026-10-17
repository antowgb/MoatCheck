"""Pure downside-risk and drawdown metrics for simple periodic returns.

Returns are decimal simple returns (for example ``0.02`` means +2%). Annual
metrics assume 252 observations per year. Drawdowns are expressed as negative
fractions from the running wealth peak.
"""

from __future__ import annotations

import math
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

_TRADING_DAYS = 252


def _return_series(returns: pd.Series | np.ndarray) -> pd.Series:
    """Validate a one-dimensional finite simple-return series; O(n) time/space."""
    if not isinstance(returns, (pd.Series, np.ndarray)):
        raise TypeError("returns must be a pandas Series or NumPy array")
    try:
        vals = (returns.to_numpy(dtype=float, na_value=np.nan, copy=True)
                if isinstance(returns, pd.Series) else np.asarray(returns, dtype=float))
    except (TypeError, ValueError) as exc:
        raise TypeError("returns must contain numeric values") from exc
    if vals.ndim != 1:
        raise ValueError("returns must be one-dimensional")
    if vals.size == 0:
        raise ValueError("returns must contain at least one observation")
    if not np.isfinite(vals).all():
        raise ValueError("returns must contain only finite values")
    if np.any(vals < -1.0):
        raise ValueError("simple returns cannot be less than -100%")
    index = returns.index if isinstance(returns, pd.Series) else pd.RangeIndex(vals.size)
    return pd.Series(vals, index=index, name=getattr(returns, "name", None))


def calculate_downside_deviation(
    returns: pd.Series | np.ndarray, target_return: float = 0.0
) -> float:
    r"""Return annualized downside deviation below a per-period target.

    .. math:: DD=\sqrt{252\,\frac{1}{N}\sum_{t=1}^N\min(R_t-T,0)^2}.

    The denominator is the full number of periods, so upside observations
    contribute zero downside rather than being omitted. If returns are
    ``[+2%, -1%, -3%]`` and target is zero, only the two losses enter the
    squared shortfall sum. Complexity: :math:`O(N)` time and :math:`O(N)` space.
    """
    values = _return_series(returns).to_numpy()
    target = float(target_return)
    if not math.isfinite(target):
        raise ValueError("target_return must be finite")
    shortfalls = np.minimum(values - target, 0.0)
    return float(np.sqrt(_TRADING_DAYS * np.mean(shortfalls**2)))


def calculate_sortino_ratio(
    returns: pd.Series | np.ndarray,
    risk_free_rate: float = 0.03,
    target_return: float = 0.0,
) -> float:
    r"""Calculate the annualized Sortino ratio with annual risk-free rate.

    .. math:: Sortino=\frac{252\bar R-R_f}{DD_T},
    where :math:`DD_T` is annualized downside deviation below per-period
    target :math:`T`. Unlike Sharpe, upside volatility is excluded: for
    example, two portfolios with equal average returns can have different
    Sharpe ratios if one has large upside swings, while Sortino penalizes only
    harmful downside swings. A zero downside deviation returns ``inf`` for a
    positive excess return, ``-inf`` for a negative one, and ``nan`` for zero.
    Complexity: :math:`O(N)` time and :math:`O(N)` space.
    """
    values = _return_series(returns).to_numpy()
    rf = float(risk_free_rate)
    target = float(target_return)
    if not math.isfinite(rf) or not math.isfinite(target):
        raise ValueError("risk_free_rate and target_return must be finite")
    downside = calculate_downside_deviation(values, target)
    excess = float(np.mean(values) * _TRADING_DAYS - rf)
    if downside == 0.0:
        if excess > 0.0:
            return math.inf
        if excess < 0.0:
            return -math.inf
        return math.nan
    return float(excess / downside)


def calculate_drawdown_series(returns: pd.Series | np.ndarray) -> pd.Series:
    r"""Compute each period's percentage distance below the prior wealth peak.

    Wealth starts at :math:`W_0=1`, evolves as
    :math:`W_t=W_{t-1}(1+R_t)`, and drawdown is
    :math:`D_t=W_t/\max_{0\le s\le t}W_s-1`. For example, returns ``[+10%,
    -20%]`` give wealth ``[1.10, 0.88]`` and drawdown ``[0%, -20%]``. The
    original pandas index is retained. Complexity: :math:`O(N)` time and
    :math:`O(N)` space.
    """
    series = _return_series(returns)
    wealth = (1.0 + series).cumprod()
    peaks = np.maximum.accumulate(np.concatenate(([1.0], wealth.to_numpy())))[1:]
    result = wealth.to_numpy() / peaks - 1.0
    return pd.Series(result, index=series.index, name="drawdown")


def calculate_max_drawdown(returns: pd.Series | np.ndarray) -> dict[str, Any]:
    r"""Return maximum peak-to-trough decline and its dates/indices and recovery.

    The reported maximum is :math:`\min_t D_t` (a negative fraction). The
    peak is the latest running high before the trough. Recovery duration is
    the number of observations from the trough to the first later point whose
    wealth regains that peak; if never recovered it is ``None``. For example,
    a fall from wealth 1.20 to 0.90 is a -25% drawdown. Sharpe treats upside
    and downside volatility alike; Sortino uses only downside, while drawdown
    describes the realized peak-to-trough path. Complexity: :math:`O(N)` time
    and :math:`O(N)` space.
    """
    series = _return_series(returns)
    wealth = np.concatenate(([1.0], (1.0 + series).cumprod().to_numpy()))
    running_peaks = np.maximum.accumulate(wealth)
    drawdowns = wealth / running_peaks - 1.0
    trough_position = int(np.argmin(drawdowns[1:]) + 1)
    magnitude = float(drawdowns[trough_position])
    peak_position = int(np.argmax(wealth[:trough_position + 1]))
    recovery_position = next(
        (i for i in range(trough_position + 1, wealth.size)
         if wealth[i] >= wealth[peak_position]
         or np.isclose(wealth[i], wealth[peak_position], rtol=1e-12, atol=1e-15)), None
    )
    # Position zero is the initial wealth before the first return; expose it
    # using the first input label for stable, user-friendly date/index fields.
    def label(position: int) -> Any:
        return series.index[0] if position == 0 else series.index[position - 1]

    return {
        "max_drawdown": magnitude,
        "peak_date": label(peak_position),
        "trough_date": label(trough_position),
        "recovery_duration": (
            int(recovery_position - trough_position)
            if recovery_position is not None else None
        ),
    }


def downside_risk_telemetry(
    returns: pd.Series | np.ndarray, target_return: float = 0.0
) -> dict[str, Any]:
    r"""Report downside-risk execution telemetry in a compact dictionary.

    Includes latency, input shape, annualized downside deviation, and maximum
    drawdown magnitude. For instance a -25% maximum drawdown means a 25% loss
    from the observed peak. Complexity: :math:`O(N)` time and :math:`O(N)` space.
    """
    started = perf_counter()
    series = _return_series(returns)
    downside = calculate_downside_deviation(series, target_return)
    drawdown = calculate_max_drawdown(series)
    return {
        "latency_ms": float((perf_counter() - started) * 1000.0),
        "input_shape": [int(series.size)],
        "n_observations": int(series.size),
        "downside_deviation": downside,
        "max_drawdown": drawdown["max_drawdown"],
    }
