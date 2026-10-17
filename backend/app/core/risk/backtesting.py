"""Pure statistical backtests for signed-return VaR forecasts."""

from __future__ import annotations

import math
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2

_SIGNIFICANCE = 0.05


def _aligned_observations(
    returns: pd.Series | np.ndarray, var_series: pd.Series | np.ndarray
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Pair finite returns and signed VaR thresholds; O(n) time and memory."""
    if not isinstance(returns, (pd.Series, np.ndarray)) or not isinstance(var_series, (pd.Series, np.ndarray)):
        raise TypeError("returns and var_series must be pandas Series or NumPy arrays")
    try:
        if isinstance(returns, pd.Series) and isinstance(var_series, pd.Series):
            paired = pd.concat([returns.rename("returns"), var_series.rename("var")], axis=1, join="inner")
            values = paired.to_numpy(dtype=float)
            shape = [int(paired.shape[0])]
        else:
            r = returns.to_numpy(dtype=float, na_value=np.nan, copy=True) if isinstance(returns, pd.Series) else np.asarray(returns, dtype=float)
            v = var_series.to_numpy(dtype=float, na_value=np.nan, copy=True) if isinstance(var_series, pd.Series) else np.asarray(var_series, dtype=float)
            if r.ndim != 1 or v.ndim != 1 or r.size != v.size:
                raise ValueError("returns and var_series must be one-dimensional and have equal length")
            values = np.column_stack((r, v))
            shape = [int(r.size)]
    except (TypeError, ValueError) as exc:
        if isinstance(exc, ValueError) and "one-dimensional" in str(exc):
            raise
        raise ValueError("returns and VaR thresholds must be aligned numeric vectors") from exc
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("returns and VaR thresholds must form paired one-dimensional vectors")
    if np.isinf(values).any():
        raise ValueError("returns and VaR thresholds must not contain infinities")
    values = values[~np.isnan(values).any(axis=1)]
    if values.shape[0] < 2:
        raise ValueError("at least two paired finite observations are required")
    return values[:, 0], values[:, 1], shape


def _confidence_level(confidence_level: float) -> float:
    """Validate an exception confidence probability in O(1)."""
    try:
        confidence = float(confidence_level)
    except (TypeError, ValueError) as exc:
        raise ValueError("confidence_level must be a finite probability") from exc
    if not math.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("confidence_level must be strictly between 0 and 1")
    return confidence


def _xlogx(count: int, probability: float) -> float:
    """Evaluate the binomial log-likelihood term with 0 log 0 = 0; O(1)."""
    return 0.0 if count == 0 else count * math.log(probability)


def _kupiec_result(exceptions: np.ndarray, confidence: float) -> dict[str, Any]:
    """Compute Bernoulli unconditional-coverage LR diagnostics; O(n)."""
    n = int(exceptions.size)
    x = int(np.sum(exceptions))
    probability = 1.0 - confidence
    observed_rate = x / n
    log_likelihood_null = _xlogx(n - x, 1.0 - probability) + _xlogx(x, probability)
    log_likelihood_unrestricted = _xlogx(n - x, 1.0 - observed_rate) + _xlogx(x, observed_rate)
    statistic = max(0.0, -2.0 * (log_likelihood_null - log_likelihood_unrestricted))
    p_value = float(chi2.sf(statistic, df=1))
    return {
        "test_statistic": float(statistic),
        "p_value": p_value,
        "total_observations": n,
        "exception_count": x,
        "expected_exception_count": float(n * probability),
        "exception_rate": float(observed_rate),
        "reject_null": bool(p_value < _SIGNIFICANCE),
    }


def kupiec_pof_test(
    returns: pd.Series | np.ndarray,
    var_series: pd.Series | np.ndarray,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    r"""Test VaR exception frequency with Kupiec's proportion-of-failures LR.

    A breach is :math:`I_t=1[R_t<VaR_t]`, since the supplied VaR is a signed
    lower-tail return threshold. The null exception probability is
    :math:`p=1-c`. With :math:`x=\sum I_t` and :math:`\hat p=x/n`,
    :math:`LR_{POF}=-2\log\frac{(1-p)^{n-x}p^x}
    {(1-\hat p)^{n-x}\hat p^x}\sim\chi^2_1`. In a 500-day example, about
    25 exceptions at 95% can have correct coverage; this test checks their
    count, not whether they cluster in time. Complexity: :math:`O(n)` time
    and :math:`O(n)` storage for pairing observations.
    """
    r, var, _ = _aligned_observations(returns, var_series)
    confidence = _confidence_level(confidence_level)
    return _kupiec_result(r < var, confidence)


def _christoffersen_result(exceptions: np.ndarray) -> dict[str, Any]:
    """Compute first-order Markov independence LR from transition counts; O(n)."""
    if exceptions.size < 2:
        raise ValueError("at least two observations are required for independence testing")
    previous, current = exceptions[:-1].astype(bool), exceptions[1:].astype(bool)
    n00 = int(np.sum(~previous & ~current))
    n01 = int(np.sum(~previous & current))
    n10 = int(np.sum(previous & ~current))
    n11 = int(np.sum(previous & current))
    n0, n1 = n00 + n01, n10 + n11
    total = n0 + n1
    pi = (n01 + n11) / total if total else 0.0
    pi01 = n01 / n0 if n0 else 0.0
    pi11 = n11 / n1 if n1 else 0.0

    def bernoulli_loglik(successes: int, trials: int, probability: float) -> float:
        if trials == 0:
            return 0.0
        failures = trials - successes
        # For an observed boundary transition rate, the zero-count side has
        # zero log-likelihood contribution by continuity.
        return _xlogx(successes, probability) + _xlogx(failures, 1.0 - probability)

    ll_independent = bernoulli_loglik(n01 + n11, total, pi)
    ll_markov = bernoulli_loglik(n01, n0, pi01) + bernoulli_loglik(n11, n1, pi11)
    statistic = max(0.0, -2.0 * (ll_independent - ll_markov))
    p_value = float(chi2.sf(statistic, df=1))
    return {
        "test_statistic": float(statistic),
        "p_value": p_value,
        "n00": n00,
        "n01": n01,
        "n10": n10,
        "n11": n11,
        "transition_probability_01": float(pi01),
        "transition_probability_11": float(pi11),
        "reject_independence": bool(p_value < _SIGNIFICANCE),
    }


def christoffersen_independence_test(
    returns: pd.Series | np.ndarray,
    var_series: pd.Series | np.ndarray,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    r"""Test whether VaR exceptions are independent rather than clustered.

    Let :math:`n_{ij}` count transitions from exception state :math:`i` to
    :math:`j`. Under independence, both transition probabilities equal
    :math:`\hat\pi=(n_{01}+n_{11})/n`; the Markov alternative estimates
    :math:`\hat\pi_{01}=n_{01}/(n_{00}+n_{01})` and
    :math:`\hat\pi_{11}=n_{11}/(n_{10}+n_{11})`. The statistic is
    :math:`LR_{ind}=-2(\ell_{ind}-\ell_{Markov})\sim\chi^2_1`. Two 500-day
    models may each have 25 failures, but one can be evenly spaced while the
    other fails in a burst; Kupiec sees equal coverage and this test detects
    clustering. Complexity: :math:`O(n)` time and storage.
    """
    r, var, _ = _aligned_observations(returns, var_series)
    _confidence_level(confidence_level)  # validates a shared public argument
    return _christoffersen_result(r < var)


def _traffic_light(exception_count: int, expected: float) -> str:
    """Map exception count into normalized Basel 4/9 bands in O(1)."""
    green_limit = 1.6 * expected  # 4 / 2.5 exceptions at the Basel 99%, 250-day reference
    yellow_limit = 3.6 * expected  # 9 / 2.5 at the same reference
    if exception_count <= green_limit:
        return "Green"
    if exception_count <= yellow_limit:
        return "Yellow"
    return "Red"


def evaluate_var_model_backtest(
    returns: pd.Series | np.ndarray,
    var_series: pd.Series | np.ndarray,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    r"""Combine coverage, independence, and Basel-style traffic-light results.

    The joint evaluation reports Kupiec :math:`LR_{POF}` and Christoffersen
    :math:`LR_{ind}` so correct coverage does not conceal clustered failures.
    Basel's reference 99%/250-day bands are 0--4 green, 5--9 yellow, and 10+
    red; these limits are scaled by expected exception count for other sample
    sizes/confidence levels. For example, 25 evenly spaced breaches in 500
    days at 95% is Green (calibrated coverage), whereas the same 25 consecutive
    failures may pass Kupiec but fail independence. Complexity: :math:`O(n)`
    time and space.
    """
    started = perf_counter()
    r, var, input_shape = _aligned_observations(returns, var_series)
    confidence = _confidence_level(confidence_level)
    exceptions = r < var
    kupiec = _kupiec_result(exceptions, confidence)
    christoffersen = _christoffersen_result(exceptions)
    zone = _traffic_light(kupiec["exception_count"], kupiec["expected_exception_count"])
    return {
        "kupiec": kupiec,
        "christoffersen": christoffersen,
        "traffic_light_zone": zone,
        "latency_ms": float((perf_counter() - started) * 1000.0),
        "exception_rate": kupiec["exception_rate"],
        "input_shape": input_shape,
        "total_observations": kupiec["total_observations"],
    }


def var_backtest_telemetry(
    returns: pd.Series | np.ndarray,
    var_series: pd.Series | np.ndarray,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    r"""Expose backtest runtime, exception rate, traffic light, and input shape.

    At 95% confidence, the expected exception rate is :math:`5\%`; the
    telemetry makes the observed rate and both coverage/clustering tests
    available to reporting systems. Complexity: :math:`O(n)` time and space.
    """
    return evaluate_var_model_backtest(returns, var_series, confidence_level)
