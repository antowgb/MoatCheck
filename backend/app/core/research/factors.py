"""Pure, database-independent factor transformations and aggregation."""

from __future__ import annotations

import math
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats.mstats import winsorize


def _validate_factor_series(series: pd.Series) -> np.ndarray:
    r"""Convert a numeric factor series to a finite float array.

    Observed values must satisfy :math:`x_i\in\mathbb{R}`; NaNs are allowed
    because real factor panels often have unavailable fundamentals. Explicit
    validation prevents infinities or text values from silently contaminating
    cross-sectional cutoffs and standardized scores.

    Complexity: :math:`O(n)` time and :math:`O(n)` temporary space.
    """
    if not isinstance(series, pd.Series):
        raise TypeError("series must be a pandas Series")
    try:
        values = series.to_numpy(dtype=float, na_value=np.nan, copy=True)
    except (TypeError, ValueError) as exc:
        raise TypeError("factor series must contain numeric values") from exc
    observed = values[~np.isnan(values)]
    if not np.isfinite(observed).all():
        raise ValueError("factor values must be finite wherever present")
    return values


def _validate_limits(limits: tuple[float, float]) -> tuple[float, float]:
    r"""Validate lower and upper tail fractions for winsorization.

    Limits represent :math:`(p_l,p_u)`, fractions of observations clipped in
    the lower and upper tails. Both must be in :math:`[0,1]` and their sum
    strictly below one so a non-empty central sample remains. This catches
    invalid parameters before SciPy calculates cutoffs.

    Complexity: :math:`O(1)` time and space.
    """
    if not isinstance(limits, tuple) or len(limits) != 2:
        raise ValueError("limits must be a (lower, upper) tuple")
    try:
        lower, upper = (float(value) for value in limits)
    except (TypeError, ValueError) as exc:
        raise ValueError("limits must contain finite numeric fractions") from exc
    if (
        not math.isfinite(lower)
        or not math.isfinite(upper)
        or lower < 0
        or upper < 0
        or lower > 1
        or upper > 1
        or lower + upper >= 1
    ):
        raise ValueError("limits must be finite fractions in [0, 1] that sum to less than 1")
    return lower, upper


def winsorize_factor(
    series: pd.Series, limits: tuple[float, float] = (0.01, 0.01)
) -> pd.Series:
    r"""Clip extreme factor observations to empirical tail cutoffs.

    For sorted non-missing values :math:`x_{(1)},\ldots,x_{(n)}`, the lower
    :math:`p_l` fraction is set to its lower cutoff and the upper
    :math:`p_u` fraction to its upper cutoff. Winsorization limits the influence
    of a single extreme financial ratio (such as an anomalous growth rate)
    without discarding the asset observation. NaNs and the original Pandas
    index/name are preserved. For example, with values
    :math:`[1,2,3,4,5,150]` and upper limit :math:`0.2`, the debt-ratio value
    150 is clipped to 5.0 rather than allowed to dominate downstream scaling.

    SciPy's ``mstats.winsorize`` uses inclusive counts by default. Complexity:
    :math:`O(n\log n)` time for cutoff selection and :math:`O(n)` space.
    """
    values = _validate_factor_series(series)
    lower, upper = _validate_limits(limits)
    valid_mask = ~np.isnan(values)
    if not valid_mask.any():
        return pd.Series(values, index=series.index, name=series.name)

    clipped = winsorize(values[valid_mask], limits=(lower, upper), inclusive=(True, True))
    result = values.copy()
    result[valid_mask] = np.asarray(clipped, dtype=float)
    return pd.Series(result, index=series.index, name=series.name)


def z_score_standardize(series: pd.Series) -> pd.Series:
    r"""Center and scale factor observations to unit population variance.

    For each observed value :math:`x_i`,
    :math:`z_i=(x_i-\mu)/\sigma`, with population mean
    :math:`\mu=\frac{1}{n}\sum_i x_i` and standard deviation
    :math:`\sigma=\sqrt{\frac{1}{n}\sum_i(x_i-\mu)^2}`. Standardized factors
    share a common scale, so a high-magnitude ratio does not dominate a
    composite merely because of its units. NaNs remain missing. If
    :math:`\sigma=0` (including a single observed value), observed results are
    set to zero because the series contains no relative information. For
    example, :math:`[1,2,3]` becomes approximately
    :math:`[-1.225,0,1.225]` under population-standard-deviation scaling.

    Complexity: :math:`O(n)` time and :math:`O(n)` output space.
    """
    values = _validate_factor_series(series)
    valid_mask = ~np.isnan(values)
    result = np.full(values.shape, np.nan, dtype=float)
    if not valid_mask.any():
        return pd.Series(result, index=series.index, name=series.name)

    observed = values[valid_mask]
    mean = float(np.mean(observed))
    std = float(np.std(observed, ddof=0))
    result[valid_mask] = 0.0 if std == 0 else (observed - mean) / std
    return pd.Series(result, index=series.index, name=series.name)


def preprocess_factor(
    series: pd.Series, limits: tuple[float, float] = (0.01, 0.01)
) -> pd.Series:
    r"""Winsorize then standardize a factor as one reusable pipeline.

    The pipeline is :math:`z_i=(w_i-\mu_w)/\sigma_w`, where :math:`w_i` is
    the winsorized observation. Clipping first prevents extreme values from
    setting an unrepresentative mean and standard deviation; standardizing
    afterward makes the bounded factor comparable with other signals. Index,
    name, and missing-value positions are retained. For example,
    :math:`[1,2,3,4,5,150]` is first winsorized (150 is capped at 5.0 with an
    upper limit of 0.2), then centered and scaled.

    Complexity: :math:`O(n\log n)` time and :math:`O(n)` space, dominated by
    winsorization.
    """
    return z_score_standardize(winsorize_factor(series, limits=limits))


def factor_stats(series: pd.Series) -> dict[str, float | int]:
    r"""Summarize observed factor values for transparent telemetry.

    Returns non-missing :math:`n`, mean :math:`\mu`, population standard
    deviation :math:`\sigma`, minimum and maximum. Empty or all-NaN series
    report ``count=0`` and NaN statistics. These summaries let callers compare
    the input and transformed factor without adding logging or I/O here.

    Complexity: :math:`O(n)` time and :math:`O(1)` output space.
    """
    values = _validate_factor_series(series)
    observed = values[~np.isnan(values)]
    if not observed.size:
        return {"count": 0, "mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "count": int(observed.size),
        "mean": float(np.mean(observed)),
        "std": float(np.std(observed, ddof=0)),
        "min": float(np.min(observed)),
        "max": float(np.max(observed)),
    }


def preprocess_factor_with_telemetry(
    series: pd.Series, limits: tuple[float, float] = (0.01, 0.01)
) -> tuple[pd.Series, dict[str, Any]]:
    r"""Preprocess a factor and return latency plus before/after statistics.

    Processing latency is :math:`1000(t_{end}-t_{start})` milliseconds from a
    monotonic high-resolution clock. The telemetry includes input and output
    :func:`factor_stats` summaries and row counts. This allows an API or UI to
    expose transformations while this module remains pure and side-effect
    free.

    Complexity: :math:`O(n\log n)` time and :math:`O(n)` working space.
    """
    before = factor_stats(series)
    start = perf_counter()
    result = preprocess_factor(series, limits=limits)
    elapsed_ms = (perf_counter() - start) * 1000.0
    return result, {
        "latency_ms": elapsed_ms,
        "before": before,
        "after": factor_stats(result),
        "input_shape": [len(series)],
        "output_shape": [len(result)],
    }


def combine_factor_scores(
    factor_scores: dict[str, dict[str, float]], factor_weights: dict[str, float]
) -> dict[str, object]:
    r"""Aggregate aligned per-asset factor scores with normalized weights.

    For asset :math:`a`, :math:`S_a=\sum_{k=1}^{K}\tilde w_k s_{k,a}`,
    where :math:`\tilde w_k=w_k/\sum_jw_j`. Normalizing weights makes the
    aggregate independent of their original scale while requiring each factor
    to cover the same asset universe.

    Complexity: :math:`O(KA)` time and :math:`O(A)` output space for ``K``
    factors and ``A`` assets.
    """
    if not factor_scores or set(factor_scores) != set(factor_weights):
        raise ValueError("factor score and weight keys must match and be non-empty")
    if any(not math.isfinite(value) for value in factor_weights.values()):
        raise ValueError("factor weights must be finite")
    weight_sum = sum(factor_weights.values())
    if weight_sum <= 0:
        raise ValueError("factor weights must sum to a positive value")
    assets = sorted({asset for values in factor_scores.values() for asset in values})
    if not assets or any(set(values) != set(assets) for values in factor_scores.values()):
        raise ValueError("every factor must provide scores for the same asset universe")
    normalized = {name: weight / weight_sum for name, weight in factor_weights.items()}
    for factor, scores in factor_scores.items():
        if any(not math.isfinite(score) for score in scores.values()):
            raise ValueError(f"factor scores for {factor} must be finite")
    composite = {
        asset: sum(normalized[factor] * factor_scores[factor][asset] for factor in normalized)
        for asset in assets
    }
    return {"scores": composite, "factor_weights": normalized, "factors": list(normalized)}
