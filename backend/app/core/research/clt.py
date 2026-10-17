"""Pure CLT simulation and distribution diagnostics for asset returns."""

from __future__ import annotations

import math
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, shapiro, skew


def _coerce_returns(returns: pd.Series | np.ndarray) -> np.ndarray:
    r"""Validate one-dimensional numeric observations and drop missing values.

    NaNs are excluded before estimation; infinities and nonnumeric values are
    rejected because they make distribution moments undefined. For example,
    daily Q4 return samples can be aggregated after omitting missing days, but
    missing values are never replaced by fabricated zeros. Complexity:
    :math:`O(n)` time and :math:`O(n)` space.
    """
    if not isinstance(returns, (pd.Series, np.ndarray)):
        raise TypeError("returns must be a pandas Series or NumPy ndarray")
    try:
        values = (
            returns.to_numpy(dtype=float, na_value=np.nan, copy=True)
            if isinstance(returns, pd.Series)
            else np.asarray(returns, dtype=float)
        )
    except (TypeError, ValueError) as exc:
        raise TypeError("returns must contain numeric values") from exc
    if values.ndim != 1:
        raise ValueError("returns must be one-dimensional")
    values = values[~np.isnan(values)]
    if not np.isfinite(values).all():
        raise ValueError("returns must be finite wherever present")
    return values


def _validate_positive_integer(value: int, name: str) -> int:
    r"""Require a positive integer simulation parameter.

    A sample size and a simulation count must satisfy :math:`k,m\in\mathbb{N}^+`;
    booleans are rejected even though Python treats them as integers. For
    instance, a monthly aggregation should specify 21 observations rather than
    ``True``. Complexity: :math:`O(1)` time and space.
    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def simulate_clt_aggregation(
    daily_returns: pd.Series,
    sample_size: int = 21,
    n_simulations: int = 1000,
    *,
    random_state: int | None = None,
) -> np.ndarray:
    r"""Bootstrap daily returns into repeated multi-day aggregate sums.

    For daily observations :math:`X_i` with mean :math:`\mu` and variance
    :math:`\sigma^2`, the sample mean converges as
    :math:`\bar{X}_n\approx N(\mu,\sigma^2/n)`; equivalently, the sum has
    approximate distribution :math:`\sum_{i=1}^nX_i\approx N(n\mu,n\sigma^2)`.
    Each output is a with-replacement bootstrap sum of ``sample_size`` daily
    returns. Daily data may be strongly skewed, while 21-day (roughly monthly)
    sums tend toward a more bell-shaped distribution. ``random_state`` makes
    experiments repeatable. Complexity: :math:`O(mk)` time and
    :math:`O(m+k\min(m,\lceil10^6/k\rceil))` working space for ``m``
    simulations and aggregation size ``k``.
    """
    if not isinstance(daily_returns, pd.Series):
        raise TypeError("daily_returns must be a pandas Series")
    sample_size = _validate_positive_integer(sample_size, "sample_size")
    n_simulations = _validate_positive_integer(n_simulations, "n_simulations")
    values = _coerce_returns(daily_returns)
    if values.size == 0:
        raise ValueError("daily_returns must contain at least one non-missing value")
    if random_state is not None and (
        isinstance(random_state, bool) or not isinstance(random_state, (int, np.integer))
    ):
        raise ValueError("random_state must be an integer or None")

    rng = np.random.default_rng(random_state)
    aggregates = np.empty(n_simulations, dtype=float)
    batch_size = max(1, min(n_simulations, 1_000_000 // sample_size))
    for start in range(0, n_simulations, batch_size):
        stop = min(start + batch_size, n_simulations)
        draws = rng.choice(values, size=(stop - start, sample_size), replace=True)
        aggregates[start:stop] = draws.sum(axis=1)
    return aggregates


def test_normality_shapiro(
    returns: pd.Series | np.ndarray,
) -> dict[str, float | int | bool]:
    r"""Run Shapiro-Wilk normality testing and report higher-shape diagnostics.

    The null hypothesis is that observations follow a normal distribution;
    :math:`W` near one and a large p-value indicate insufficient evidence to
    reject it at the chosen threshold. The result includes :math:`W`, p-value,
    sample skewness, and excess kurtosis. A daily return series can be skewed,
    while its 21-day aggregate may have lower absolute skew and look closer to
    normal, illustrating the CLT. At least three observations are required.
    SciPy notes that p-values may be inaccurate for samples above 5,000.
    Complexity: :math:`O(n\log n)` time and :math:`O(n)` space.
    """
    values = _coerce_returns(returns)
    if values.size < 3:
        raise ValueError("at least three non-missing observations are required for Shapiro-Wilk")
    if np.ptp(values) == 0.0:
        raise ValueError("Shapiro-Wilk is undefined for constant returns")
    result = shapiro(values)
    skewness = float(skew(values, bias=False)) if values.size >= 3 else float("nan")
    excess_kurtosis = float(kurtosis(values, fisher=True, bias=False)) if values.size >= 4 else float("nan")
    p_value = float(result.pvalue)
    return {
        "statistic": float(result.statistic),
        "p_value": p_value,
        "skewness": skewness,
        "kurtosis": excess_kurtosis,
        "normality_not_rejected": bool(p_value >= 0.05),
        "observations": int(values.size),
    }


def calculate_higher_moments(
    returns: pd.Series | np.ndarray,
) -> dict[str, float | int]:
    r"""Calculate mean, sample variance, skewness, and excess kurtosis.

    For observations :math:`X_i`, return :math:`\mu`, sample variance
    :math:`s^2=\sum_i(X_i-\bar X)^2/(n-1)`, adjusted skewness, and adjusted
    excess kurtosis :math:`\kappa=E[((X-\mu)/\sigma)^4]-3`. A normal model has
    skewness and excess kurtosis near zero; positive excess kurtosis signals
    heavier tails. For example, comparing daily returns to 21-day sums shows
    how aggregation can reduce skewness and bring kurtosis toward zero.
    Complexity: :math:`O(n)` time and :math:`O(n)` space.
    """
    values = _coerce_returns(returns)
    if values.size < 2:
        raise ValueError("at least two non-missing observations are required for variance")
    if values.size < 3:
        raise ValueError("at least three non-missing observations are required for skewness")
    if np.ptp(values) == 0.0:
        return {
            "mean": float(np.mean(values)),
            "variance": 0.0,
            "skewness": float("nan"),
            "kurtosis": float("nan"),
            "observations": int(values.size),
        }
    return {
        "mean": float(np.mean(values)),
        "variance": float(np.var(values, ddof=1)),
        "skewness": float(skew(values, bias=False)),
        "kurtosis": (
            float(kurtosis(values, fisher=True, bias=False))
            if values.size >= 4
            else float("nan")
        ),
        "observations": int(values.size),
    }


def simulate_clt_aggregation_with_telemetry(
    daily_returns: pd.Series,
    sample_size: int = 21,
    n_simulations: int = 1000,
    *,
    random_state: int | None = None,
) -> tuple[np.ndarray, dict[str, float | int | list[int]]]:
    r"""Simulate aggregate returns and report latency plus sample dimensions.

    Telemetry includes elapsed milliseconds :math:`1000(t_1-t_0)`, the number
    of usable daily observations, ``sample_size``, ``n_simulations``, and
    output shape. A daily-vs-monthly comparison can inspect these metrics next
    to :math:`\bar{X}_n\approx N(\mu,\sigma^2/n)`. Complexity:
    :math:`O(mk)` time and bounded batch working space plus the output vector.
    """
    start = perf_counter()
    aggregates = simulate_clt_aggregation(
        daily_returns,
        sample_size=sample_size,
        n_simulations=n_simulations,
        random_state=random_state,
    )
    elapsed_ms = (perf_counter() - start) * 1000.0
    observations = int(_coerce_returns(daily_returns).size)
    return aggregates, {
        "latency_ms": elapsed_ms,
        "input_observations": observations,
        "sample_size": int(sample_size),
        "n_simulations": int(n_simulations),
        "output_shape": [int(aggregates.size)],
    }


def analyze_returns_with_telemetry(
    returns: pd.Series | np.ndarray,
) -> tuple[dict[str, float | int], dict[str, float | int | bool]]:
    r"""Return moment and Shapiro diagnostics with execution telemetry.

    The telemetry reports elapsed milliseconds, observation count, a Shapiro
    normality flag, skewness, and excess kurtosis. For example, comparing this
    result for daily values and bootstrapped 21-day sums shows whether monthly
    aggregation makes a skewed daily distribution more normal. The normality
    flag means only that the test did not reject at :math:`\alpha=0.05`; it
    does not prove normality. Complexity: :math:`O(n\log n)` time and
    :math:`O(n)` space, dominated by Shapiro-Wilk.
    """
    start = perf_counter()
    moments = calculate_higher_moments(returns)
    normality = test_normality_shapiro(returns)
    elapsed_ms = (perf_counter() - start) * 1000.0
    telemetry: dict[str, float | int | bool] = {
        "latency_ms": elapsed_ms,
        "observations": int(moments["observations"]),
        "normality_not_rejected": bool(normality["normality_not_rejected"]),
        "shapiro_p_value": float(normality["p_value"]),
        "skewness": float(moments["skewness"]),
        "kurtosis": float(moments["kurtosis"]),
    }
    return moments, telemetry


def normal_distribution_reference(
    mean: float, variance: float, points: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    r"""Build x and density arrays for a normal approximation reference curve.

    The CLT reference for aggregated sample means is
    :math:`\bar X_n\sim N(\mu,\sigma^2/n)`. This helper samples its PDF so a
    caller can compare daily and monthly histograms without plotting or I/O in
    the math engine. A daily distribution can be skewed even when its monthly
    mean distribution has this bell-shaped approximation. Complexity:
    :math:`O(p)` time and space for ``p`` points.
    """
    try:
        mean_value = float(mean)
        variance_value = float(variance)
    except (TypeError, ValueError) as exc:
        raise TypeError("mean and variance must be numeric") from exc
    if not math.isfinite(mean_value):
        raise ValueError("mean must be finite")
    if not math.isfinite(variance_value) or variance_value <= 0.0:
        raise ValueError("variance must be finite and positive")
    points = _validate_positive_integer(points, "points")
    std = math.sqrt(variance_value)
    x = np.linspace(mean_value - 4 * std, mean_value + 4 * std, points)
    return x, norm.pdf(x, loc=mean_value, scale=std)
