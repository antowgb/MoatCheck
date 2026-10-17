"""Pure Rescaled Range (R/S) diagnostics for trend persistence."""

from __future__ import annotations

import math
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import linregress


def _validated_values(series: pd.Series) -> np.ndarray:
    r"""Return finite observations from a numeric Series in original order.

    :math:`X_1,\ldots,X_T` are the ordered observations used by R/S analysis.
    For instance, KO and PEP valuation observations must remain chronological
    when measuring persistence. Complexity: :math:`O(T)` time and space.
    """
    if not isinstance(series, pd.Series):
        raise TypeError("series must be a pandas Series")
    if not series.index.is_unique:
        raise ValueError("series index must be unique")
    try:
        values = series.to_numpy(dtype=float, na_value=np.nan, copy=True)
    except (TypeError, ValueError) as exc:
        raise TypeError("series must contain numeric values") from exc
    if not np.isfinite(values).all():
        raise ValueError("series must contain only finite, non-missing values")
    return values


def _fit_scaling(
    values: np.ndarray, max_lag: int, input_type: str
) -> tuple[float, float, int]:
    r"""Estimate the log-log slope of standard deviation over time scales.

    For level data, use :math:`\Delta_\ell X_t=X_{t+\ell}-X_t`; for returns,
    use :math:`A_{t,\ell}=\sum_{j=1}^{\ell}r_{t+j}`. Regress
    :math:`\log(\operatorname{std}(\Delta_\ell X))=c+H\log(\ell)+\epsilon`;
    the variance-time scaling law is :math:`\operatorname{std}(A_\ell)\propto
    \ell^H`. Complexity is :math:`O(TL)` time and :math:`O(T)` space for
    ``T`` observations and ``L`` tested lags.
    """
    count = values.size
    upper_lag = min(max_lag, count // 2)
    if upper_lag < 4:
        return float("nan"), float("nan"), 0
    n_scales = min(20, upper_lag)
    lags = np.unique(np.rint(np.geomspace(1, upper_lag, num=n_scales)).astype(int))
    log_lags: list[float] = []
    log_scales: list[float] = []
    cumulative_returns = (
        np.concatenate(([0.0], np.cumsum(values))) if input_type == "returns" else None
    )
    for lag in lags:
        if input_type == "levels":
            increments = values[int(lag) :] - values[: -int(lag)]
        else:
            if cumulative_returns is None:
                raise RuntimeError("return input requires cumulative returns")
            increments = cumulative_returns[int(lag) :] - cumulative_returns[: -int(lag)]
        scale = float(np.std(increments, ddof=1))
        if math.isfinite(scale) and scale > 0.0:
            log_lags.append(math.log(float(lag)))
            log_scales.append(math.log(scale))

    if len(log_lags) < 3 or np.ptp(log_lags) == 0.0:
        return float("nan"), float("nan"), len(log_lags)
    regression = linregress(log_lags, log_scales)
    return float(regression.slope), float(regression.rvalue**2), len(log_lags)


def calculate_hurst_exponent(
    series: pd.Series, max_lag: int = 100, *, input_type: str = "levels"
) -> float:
    r"""Estimate the Hurst exponent with variance-time scaling.

    ``input_type="levels"`` (the default) expects price/path levels and
    measures :math:`X_{t+\ell}-X_t`; ``input_type="returns"`` expects discrete
    returns and measures their overlapping sums. Neither input is implicitly
    differenced twice. The scaling law is
    :math:`\operatorname{std}(A_\ell)\propto\ell^H`; a Brownian price path or
    iid returns should give :math:`H\approx0.5`, while persistent growth gives
    a higher slope. A KO/PEP spread with :math:`H=0.28` indicates
    mean-reverting behavior. Complexity: :math:`O(TL)` time and :math:`O(T)`
    space. Returns NaN if fewer than three usable scales remain.
    """
    if isinstance(max_lag, (bool, np.bool_)) or not isinstance(max_lag, (int, np.integer)):
        raise ValueError("max_lag must be an integer of at least 4")
    if max_lag < 4:
        raise ValueError("max_lag must be an integer of at least 4")
    if input_type not in {"levels", "returns"}:
        raise ValueError("input_type must be 'levels' or 'returns'")
    values = _validated_values(series)
    if values.size < 8:
        return float("nan")
    exponent, _, _ = _fit_scaling(values, int(max_lag), input_type)
    return exponent


def classify_market_regime(hurst_exponent: float) -> str:
    r"""Classify persistence as mean reverting, random walk, or trending.

    The thresholds are :math:`H<0.45` (mean reverting),
    :math:`0.45\le H\le0.55` (random walk), and :math:`H>0.55` (trending).
    Thus a KO/PEP spread with :math:`H=0.28` is classified as mean reverting,
    while a series with :math:`H=0.62` is classified as trending. Complexity:
    :math:`O(1)` time and space.
    """
    try:
        value = float(hurst_exponent)
    except (TypeError, ValueError) as exc:
        raise TypeError("hurst_exponent must be a finite number") from exc
    if not math.isfinite(value):
        raise ValueError("hurst_exponent must be finite")
    if value < 0.45:
        return "mean_reverting"
    if value <= 0.55:
        return "random_walk"
    return "trending"


def calculate_rolling_hurst(
    series: pd.Series,
    window: int = 252,
    max_lag: int = 50,
    *,
    input_type: str = "levels",
) -> pd.Series:
    r"""Calculate trailing Hurst estimates with explicit level/return semantics.

    At each time :math:`t`, estimate :math:`H_t` from the latest ``window``
    observations using the same variance-time regression. ``input_type`` is
    ``"levels"`` for price/path levels and ``"returns"`` for discrete returns;
    return windows aggregate returns over each lag and never difference them.
    A KO/PEP spread whose
    rolling estimate moves from :math:`0.28` toward :math:`0.5` is shifting
    from mean-reverting behavior toward random-walk-like behavior. Initial
    windows are NaN; windows with too few usable scales also return NaN.
    Complexity: :math:`O(TWL)` time and :math:`O(W)` temporary space for
    ``T`` observations, window ``W``, and at most 20 lag scales.
    """
    if not isinstance(series, pd.Series):
        raise TypeError("series must be a pandas Series")
    if isinstance(window, (bool, np.bool_)) or not isinstance(window, (int, np.integer)):
        raise ValueError("window must be an integer of at least 8")
    if window < 8:
        raise ValueError("window must be an integer of at least 8")
    if isinstance(max_lag, (bool, np.bool_)) or not isinstance(max_lag, (int, np.integer)):
        raise ValueError("max_lag must be an integer of at least 4")
    if max_lag < 4:
        raise ValueError("max_lag must be an integer of at least 4")
    if input_type not in {"levels", "returns"}:
        raise ValueError("input_type must be 'levels' or 'returns'")
    if not series.index.is_unique:
        raise ValueError("series index must be unique")
    try:
        values = series.to_numpy(dtype=float, na_value=np.nan, copy=True)
    except (TypeError, ValueError) as exc:
        raise TypeError("series must contain numeric values") from exc
    if np.isinf(values).any():
        raise ValueError("series must be finite wherever present")
    result = np.full(len(series), np.nan, dtype=float)
    # Preserve positional rolling windows and only emit an estimate when the
    # original window is complete and contains no missing observations.
    for end in range(int(window) - 1, len(values)):
        segment = values[end - int(window) + 1 : end + 1]
        if segment.size == int(window) and not np.isnan(segment).any():
            result[end] = _fit_scaling(segment, int(max_lag), input_type)[0]
    return pd.Series(result, index=series.index, name=series.name, dtype=float)


def hurst_telemetry(
    series: pd.Series, max_lag: int = 100, *, input_type: str = "levels"
) -> dict[str, Any]:
    r"""Return variance-time fit quality, regime, and processing latency.

    Telemetry includes :math:`H`, the :math:`R^2` of
    :math:`\log(\operatorname{std}(A_\ell))` versus :math:`\log(\ell)`, tested
    scale count, processing time in milliseconds, and regime label. Use
    ``input_type="levels"`` for price paths or ``"returns"`` for discrete
    returns. For example, :math:`H=0.28` for KO/PEP yields ``mean_reverting``.
    Complexity:
    :math:`O(TL)` time and :math:`O(T)` space.
    """
    start = perf_counter()
    if isinstance(max_lag, (bool, np.bool_)) or not isinstance(max_lag, (int, np.integer)) or max_lag < 4:
        raise ValueError("max_lag must be an integer of at least 4")
    if input_type not in {"levels", "returns"}:
        raise ValueError("input_type must be 'levels' or 'returns'")
    values = _validated_values(series)
    exponent, r_squared, n_scales = (
        _fit_scaling(values, int(max_lag), input_type)
        if values.size >= 8
        else (float("nan"), float("nan"), 0)
    )
    regime = classify_market_regime(exponent) if math.isfinite(exponent) else "insufficient_data"
    return {
        "hurst_exponent": float(exponent),
        "r_squared": float(r_squared),
        "regime": regime,
        "n_observations": int(values.size),
        "n_scales": int(n_scales),
        "latency_ms": float((perf_counter() - start) * 1000.0),
    }
