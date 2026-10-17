"""Pure, database-independent return calculations for portfolio analytics.

Rows represent consecutive observations and columns represent assets. Prices
must be positive wherever present; missing observations are preserved through
the arithmetic and removed by the returned DataFrame's ``dropna`` operation.
"""

from __future__ import annotations

from collections.abc import Callable
from time import perf_counter
from typing import Any, TypeVar

import numpy as np
import pandas as pd

T = TypeVar("T")


def _validate_prices(prices_df: pd.DataFrame) -> None:
    """Validate price inputs without mutating them.

    The domain condition is :math:`P_{i,t}>0` for every observed price.
    Positive prices are required because the logarithm is undefined for zero
    and negative prices, and simple returns cannot use a zero denominator.
    Missing observations are permitted and handled by the public functions;
    checking this condition up front gives callers a clear error at the source
    of invalid return values.

    Complexity: :math:`O(nm)` for ``n`` rows and ``m`` asset columns, matching
    the size of the input that must be inspected.
    """
    if not isinstance(prices_df, pd.DataFrame):
        raise TypeError("prices_df must be a pandas DataFrame")
    try:
        values = prices_df.to_numpy(dtype=float, copy=False)
    except (TypeError, ValueError) as exc:
        raise TypeError("prices_df must contain numeric prices") from exc
    observed = values[~np.isnan(values)]
    if not np.isfinite(observed).all():
        raise ValueError("prices must be finite wherever present")
    if np.any(observed <= 0):
        raise ValueError("prices must be strictly positive wherever present")


def to_log_returns(prices_df: pd.DataFrame) -> pd.DataFrame:
    r"""Compute adjacent-period continuously compounded returns.

    For asset :math:`i` at time :math:`t`,
    :math:`r_{i,t}=\ln(P_{i,t}/P_{i,t-1})`. Log returns are additive across
    time: summing them gives the log of the cumulative price ratio, which
    makes multi-period aggregation and many statistical models convenient.
    For example, a move from 100 to 102 gives
    :math:`\ln(102)-\ln(100)\approx0.0198`. Subtracting logarithms directly
    also avoids forming a price ratio that could overflow or underflow. The
    first observation has no prior price and is removed, as are rows with
    missing return values.

    Complexity: :math:`O(nm)` time and :math:`O(nm)` output space for ``n``
    observations and ``m`` assets.
    """
    _validate_prices(prices_df)
    return (np.log(prices_df) - np.log(prices_df.shift(1))).dropna()


def to_simple_returns(prices_df: pd.DataFrame) -> pd.DataFrame:
    r"""Compute adjacent-period arithmetic returns.

    For asset :math:`i` at time :math:`t`,
    :math:`R_{i,t}=(P_{i,t}-P_{i,t-1})/P_{i,t-1}`. Simple returns express
    the directly realized percentage change and combine across assets in a
    portfolio for a single period. Unlike log returns, they do not add exactly
    across time. For example, a move from 100 to 102 is a simple return of
    :math:`(102-100)/100=0.02`, or 2%. The first observation and rows with
    missing return values are removed.

    Complexity: :math:`O(nm)` time and :math:`O(nm)` output space for ``n``
    observations and ``m`` assets.
    """
    _validate_prices(prices_df)
    return prices_df.pct_change(fill_method=None).dropna()


def annualized_mean_returns(
    log_returns_df: pd.DataFrame, trading_days: int = 252
) -> pd.Series:
    r"""Annualize the per-period arithmetic mean of log returns.

    For asset :math:`i`, :math:`\mu_i^{(ann)}=D\,\bar r_i`, where
    :math:`D` is ``trading_days`` and :math:`\bar r_i` is the sample mean of
    its periodic log returns. This estimates expected annual log growth under
    a stable-return assumption; log returns are used because their time sums
    aggregate exactly. The result remains in log-return units.

    Complexity: :math:`O(nm)` time and :math:`O(m)` output space.
    """
    if not isinstance(log_returns_df, pd.DataFrame):
        raise TypeError("log_returns_df must be a pandas DataFrame")
    _validate_trading_days(trading_days)
    return log_returns_df.mean(axis=0) * trading_days


def annualized_covariance_matrix(
    log_returns_df: pd.DataFrame, trading_days: int = 252
) -> pd.DataFrame:
    r"""Annualize the sample covariance matrix of periodic log returns.

    For assets :math:`i,j`, :math:`\Sigma_{ij}^{(ann)}=D\,s_{ij}`, where
    :math:`s_{ij}` is the sample covariance (denominator :math:`n-1`) and
    :math:`D` is ``trading_days``. Under the usual weak-stationarity
    assumption, covariance of sums of uncorrelated periods scales linearly
    with the number of periods. Log returns make those sums exact across time.

    Complexity: :math:`O(nm^2)` time and :math:`O(m^2)` output space for
    ``n`` observations and ``m`` assets. For example, a daily covariance of
    :math:`0.0001` annualizes to :math:`0.0252` at 252 trading days.
    """
    if not isinstance(log_returns_df, pd.DataFrame):
        raise TypeError("log_returns_df must be a pandas DataFrame")
    _validate_trading_days(trading_days)
    columns = log_returns_df.columns
    values = log_returns_df.to_numpy(dtype=float, copy=False)
    n_assets = len(columns)
    covariance = np.full((n_assets, n_assets), np.nan, dtype=float)

    # Matrix products preserve pairwise-NaN semantics without Python loops:
    # each (i,j) entry uses only rows where both assets have finite returns.
    valid = np.isfinite(values).astype(float)
    finite_values = np.where(valid, values, 0.0)
    pair_counts = valid.T @ valid
    pair_sums = finite_values.T @ valid
    pair_cross_products = finite_values.T @ finite_values
    centered_cross_products = pair_cross_products - np.divide(
        pair_sums * pair_sums.T,
        pair_counts,
        out=np.zeros_like(pair_cross_products),
        where=pair_counts > 0,
    )
    np.divide(
        centered_cross_products * trading_days,
        pair_counts - 1,
        out=covariance,
        where=pair_counts > 1,
    )

    return pd.DataFrame(covariance, index=columns, columns=columns)


def log_to_simple_return(log_return: float) -> float:
    r"""Convert a cumulative log return to a simple percentage return.

    The conversion is :math:`R=\exp(r)-1`, where :math:`r` is a cumulative
    log return. Exponentiation reverses the logarithm of the gross return and
    is therefore the right conversion when reporting cumulative performance
    as an ordinary percentage change.

    Complexity: :math:`O(1)` time and :math:`O(1)` space.
    """
    value = float(log_return)
    if not np.isfinite(value):
        raise ValueError("log_return must be finite")
    return float(np.expm1(value))


def data_shape_info(data: pd.DataFrame | pd.Series) -> dict[str, Any]:
    r"""Return compact shape and missing-value telemetry for tabular data.

    The reported shape is :math:`(n,m)`, with a Series represented as
    :math:`(n,1)`. The result contains row/column counts (a Series is reported as one
    column), shape, and missing-value count. Shape visibility helps callers
    explain how much input reached a calculation and how much output it
    produced without coupling this math module to an API or logging system.

    Complexity: :math:`O(nm)` time to count missing values and :math:`O(1)`
    result space.
    """
    if isinstance(data, pd.Series):
        rows, columns = len(data), 1
    elif isinstance(data, pd.DataFrame):
        rows, columns = data.shape
    else:
        raise TypeError("data must be a pandas DataFrame or Series")
    return {
        "rows": int(rows),
        "columns": int(columns),
        "shape": [int(rows), int(columns)],
        "missing_values": int(data.isna().to_numpy().sum()),
    }


def measure_execution_latency(
    function: Callable[..., T], *args: Any, **kwargs: Any
) -> tuple[T, dict[str, Any]]:
    r"""Run a calculation and return its result with latency/shape telemetry.

    Latency is :math:`1000(t_{end}-t_{start})` milliseconds, measured with a
    monotonic high-resolution clock. If the first positional argument is a DataFrame
    or Series, its shape is included; DataFrame/Series results contribute
    output shape as well. This helper performs no I/O, so callers decide how
    telemetry is displayed or recorded.

    Complexity: the wrapped function's complexity plus :math:`O(nm)` for
    input/output missing-value counts; telemetry storage is :math:`O(1)`.
    """
    input_data = args[0] if args else kwargs.get("prices_df", kwargs.get("log_returns_df"))
    input_shape = data_shape_info(input_data) if isinstance(input_data, (pd.DataFrame, pd.Series)) else None
    start = perf_counter()
    result = function(*args, **kwargs)
    elapsed_ms = (perf_counter() - start) * 1000.0
    output_shape = data_shape_info(result) if isinstance(result, (pd.DataFrame, pd.Series)) else None
    return result, {
        "latency_ms": elapsed_ms,
        "input_shape": input_shape,
        "output_shape": output_shape,
    }


def _validate_trading_days(trading_days: int) -> None:
    r"""Require a positive integer annualization factor.

    Annualized statistics use :math:`x_{ann}=D x_{period}`, so :math:`D` must
    be a positive whole number of periods. This prevents nonsensical scaling
    factors and excludes booleans, even though Python treats them as integers.

    Complexity: :math:`O(1)` time and space.
    """
    # Annualization uses D * statistic, so D must be a positive period count.
    if isinstance(trading_days, bool) or not isinstance(trading_days, (int, np.integer)) or trading_days <= 0:
        raise ValueError("trading_days must be a positive integer")
