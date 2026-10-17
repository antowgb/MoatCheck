"""Pure time-series stationarity and cointegration diagnostics."""

from __future__ import annotations

import math
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint

_SIGNIFICANCE = 0.05


def _as_series(series: pd.Series | np.ndarray, name: str = "series") -> pd.Series:
    """Validate a finite, non-empty one-dimensional numeric series in O(n)."""
    if not isinstance(series, (pd.Series, np.ndarray)):
        raise TypeError(f"{name} must be a pandas Series or NumPy array")
    try:
        values = (series.to_numpy(dtype=float, na_value=np.nan, copy=True)
                  if isinstance(series, pd.Series) else np.asarray(series, dtype=float))
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain numeric values") from exc
    if values.ndim != 1 or values.size < 3:
        raise ValueError(f"{name} must be one-dimensional with at least 3 observations")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must contain only finite observations")
    index = series.index if isinstance(series, pd.Series) else pd.RangeIndex(values.size)
    return pd.Series(values, index=index, name=getattr(series, "name", name))


def _validate_regression_type(regression_type: str) -> str:
    """Validate statsmodels ADF deterministic-term option in O(1)."""
    if regression_type not in {"c", "ct", "ctt", "n"}:
        raise ValueError("regression_type must be one of: 'c', 'ct', 'ctt', 'n'")
    return regression_type


def test_augmented_dickey_fuller(
    series: pd.Series | np.ndarray, regression_type: str = "c"
) -> dict[str, Any]:
    r"""Run the Augmented Dickey-Fuller unit-root test.

    The fitted regression is
    :math:`\Delta y_t=\alpha+\beta t+\gamma y_{t-1}+
    \sum_{i=1}^{p}\delta_i\Delta y_{t-i}+\epsilon_t`, including only the
    deterministic terms selected by ``regression_type``. The null is a unit
    root (:math:`\gamma=0`); a p-value below 0.05 rejects it. Price levels
    such as a random walk often fail this test while their first differences
    (returns/changes) may pass. Complexity is approximately :math:`O(n p^2)`
    for the lagged least-squares regression, with :math:`O(np)` storage.
    """
    values = _as_series(series).to_numpy()
    regression = _validate_regression_type(regression_type)
    try:
        result = adfuller(
            values, regression=regression, autolag="AIC", result_object=True
        )
    except (ValueError, np.linalg.LinAlgError) as exc:
        raise ValueError(f"ADF test could not be estimated: {exc}") from exc
    p_value = float(result.pvalue)
    return {
        "adf_statistic": float(result.statistic),
        "p_value": p_value,
        "critical_values": {key: float(value) for key, value in result.critical_values.items()},
        "used_lags": int(result.lags),
        "n_observations": int(result.nobs),
        "regression_type": regression,
        "is_stationary": bool(p_value < _SIGNIFICANCE),
    }


def make_series_stationary(
    series: pd.Series, max_differencing: int = 2
) -> tuple[pd.Series, int]:
    r"""Difference a series repeatedly until ADF rejects a unit root.

    First differencing maps levels to changes, :math:`\Delta y_t=y_t-y_{t-1}`;
    a random-walk price path may need :math:`d=1`, while an already stationary
    AR(1) process should return :math:`d=0`. In a cointegrated pair, individual
    price levels can be nonstationary even though their fitted spread is
    stationary. The stopping rule uses ADF p-value < 0.05. Complexity is at
    most :math:`O(d n p^2)` time and :math:`O(n)` working space.
    """
    if not isinstance(series, pd.Series):
        raise TypeError("series must be a pandas Series")
    if isinstance(max_differencing, bool) or not isinstance(max_differencing, int) or max_differencing < 0:
        raise ValueError("max_differencing must be a non-negative integer")
    current = _as_series(series)
    for degree in range(max_differencing + 1):
        if test_augmented_dickey_fuller(current)["is_stationary"]:
            return current.rename(series.name), degree
        if degree < max_differencing:
            current = current.diff().dropna()
            if current.size < 3:
                break
    raise ValueError(
        f"series did not achieve stationarity within {max_differencing} differences"
    )


def test_cointegration_pair(
    series_a: pd.Series, series_b: pd.Series
) -> dict[str, Any]:
    r"""Test whether two price series share a stationary linear combination.

    Engle-Granger estimates :math:`y_t=\alpha+\beta x_t+u_t`, then tests the
    residual :math:`u_t` for a unit root using cointegration-specific critical
    values. For example, two prices each wandering around 100 and 50 can still
    be cointegrated if :math:`y_t-2x_t` fluctuates around a stable spread.
    Missing dates are aligned by index intersection; complexity is
    approximately :math:`O(n p^2)` time and :math:`O(n)` space.
    """
    if not isinstance(series_a, pd.Series) or not isinstance(series_b, pd.Series):
        raise TypeError("series_a and series_b must be pandas Series")
    aligned = pd.concat(
        [series_a.rename("a"), series_b.rename("b")], axis=1, join="inner"
    ).dropna()
    a = _as_series(aligned["a"], "series_a")
    b = _as_series(aligned["b"], "series_b")
    try:
        statistic, p_value, critical = coint(a.to_numpy(), b.to_numpy(), trend="c", autolag="aic")
    except (ValueError, np.linalg.LinAlgError) as exc:
        raise ValueError(f"cointegration test could not be estimated: {exc}") from exc
    p_value = float(p_value)
    return {
        "test_statistic": float(statistic),
        "p_value": p_value,
        "critical_values": {
            "1%": float(critical[0]), "5%": float(critical[1]), "10%": float(critical[2])
        },
        "is_cointegrated": bool(p_value < _SIGNIFICANCE),
        "n_observations": int(len(aligned)),
    }


def stationarity_telemetry(
    series: pd.Series | np.ndarray, regression_type: str = "c"
) -> dict[str, Any]:
    r"""Return ADF diagnostics plus execution latency and input-shape telemetry.

    The result includes :math:`p`, the stationarity flag, and input shape; for
    example the levels of a random-walk price path may have ``p > 0.05`` even
    when its first differences are stationary. Complexity follows ADF:
    approximately :math:`O(n p^2)` time and :math:`O(np)` storage.
    """
    started = perf_counter()
    values = _as_series(series)
    result = test_augmented_dickey_fuller(values, regression_type)
    return {
        **result,
        "latency_ms": float((perf_counter() - started) * 1000.0),
        "input_shape": [int(values.size)],
    }
