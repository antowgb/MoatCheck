"""Pure portfolio risk metrics with explicit signed-return conventions.

Existing portfolio aggregation functions report VaR/ES as positive loss
fractions. The scalar estimators added below report the lower-tail return
quantile and expected return in signed decimal units (negative values are
losses).
"""

from __future__ import annotations

import math
from datetime import date
from time import perf_counter
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew

Method = Literal["historical", "parametric"]


def as_loss_amount(return_quantile: float) -> float:
    r"""Convert a signed return quantile to a positive loss amount/fraction.

    Scalar VaR/CVaR functions return signed lower-tail returns, so
    :math:`-0.021` means a loss of :math:`0.021` (2.1%). This helper applies
    :math:`L=-q_R` without changing units; multiply a fraction by portfolio
    value to obtain currency. Positive return quantiles become negative loss
    amounts, correctly representing gains. Complexity: :math:`O(1)` time and
    space.
    """
    try:
        value = float(return_quantile)
    except (TypeError, ValueError) as exc:
        raise ValueError("return_quantile must be finite") from exc
    if not math.isfinite(value):
        raise ValueError("return_quantile must be finite")
    return -value


def as_return_quantile(loss_amount: float) -> float:
    r"""Convert a loss amount/fraction to the corresponding signed return.

    Portfolio ``portfolio_var_cvar`` reports positive loss fractions; this
    helper maps :math:`L=0.021` to the signed return quantile
    :math:`q_R=-0.021`. Units are preserved, so a currency loss remains in
    currency and a decimal fraction remains a fraction. Complexity:
    :math:`O(1)` time and space.
    """
    try:
        value = float(loss_amount)
    except (TypeError, ValueError) as exc:
        raise ValueError("loss_amount must be finite") from exc
    if not math.isfinite(value):
        raise ValueError("loss_amount must be finite")
    return -value


def _validated_return_vector(
    returns: pd.Series | np.ndarray,
    *,
    minimum_observations: int = 2,
) -> np.ndarray:
    r"""Validate and extract finite one-dimensional simple returns.

    The sample :math:`r_1,\ldots,r_n` is expressed in decimal units; NaNs are
    omitted, while infinities and multidimensional arrays are rejected. For a
    €1,000,000 portfolio, a return of :math:`-0.021` represents a €21,000
    loss. Complexity: :math:`O(n)` time and :math:`O(n)` temporary space.
    """
    if not isinstance(returns, (pd.Series, np.ndarray)):
        raise TypeError("returns must be a pandas Series or NumPy array")
    try:
        values = returns.to_numpy(dtype=float, na_value=np.nan, copy=True) if isinstance(returns, pd.Series) else np.asarray(returns, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("returns must contain numeric values") from exc
    if values.ndim != 1:
        raise ValueError("returns must be one-dimensional")
    if np.isinf(values).any():
        raise ValueError("returns must be finite wherever present")
    values = values[~np.isnan(values)]
    if values.size < minimum_observations:
        raise ValueError(f"at least {minimum_observations} non-missing returns are required")
    return values


def _validate_confidence_level(confidence_level: float) -> float:
    r"""Validate a tail confidence level strictly inside :math:`(0,1)`.

    A 95% confidence level evaluates the lower 5% return tail, which gives a
    €21,000 loss threshold when VaR is :math:`-2.1\%` on a €1,000,000
    portfolio. Complexity: :math:`O(1)` time and space.
    """
    try:
        confidence = float(confidence_level)
    except (TypeError, ValueError) as exc:
        raise ValueError("confidence_level must be a finite number") from exc
    if not math.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("confidence_level must be strictly between 0 and 1")
    return confidence


def _cornish_fisher_quantile(
    values: np.ndarray, confidence_level: float
) -> tuple[float, float, float]:
    r"""Return a lower-tail Cornish-Fisher quantile and sample moments.

    The adjusted standard-normal quantile is
    :math:`z_{CF}=z+\frac{z^2-1}{6}S+\frac{z^3-3z}{24}K-
    \frac{2z^3-5z}{36}S^2`, where :math:`S` is skewness and :math:`K` is
    excess kurtosis. A heavier left tail may move the 99% quantile below the
    Gaussian quantile; at 95%, correction direction depends on moments.
    Complexity: :math:`O(n)` time and :math:`O(1)` additional space.
    """
    if values.size < 4:
        raise ValueError("Cornish-Fisher VaR requires at least 4 observations")
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1))
    if std == 0.0:
        return mean, 0.0, 0.0
    sample_skew = float(skew(values, bias=False))
    excess_kurtosis = float(kurtosis(values, fisher=True, bias=False))
    if not math.isfinite(sample_skew) or not math.isfinite(excess_kurtosis):
        raise ValueError("returns must have finite skewness and kurtosis")
    z = float(norm.ppf(1.0 - confidence_level))
    z_squared = z * z
    adjusted_z = (
        z
        + (z_squared - 1.0) * sample_skew / 6.0
        + (z**3 - 3.0 * z) * excess_kurtosis / 24.0
        - (2.0 * z**3 - 5.0 * z) * sample_skew**2 / 36.0
    )
    return mean + std * adjusted_z, sample_skew, excess_kurtosis


def calculate_parametric_var(
    returns: pd.Series | np.ndarray, confidence_level: float = 0.95
) -> float:
    r"""Calculate Gaussian VaR as a signed lower-tail portfolio return.

    Assuming :math:`R\sim N(\mu,\sigma^2)`, return
    :math:`VaR_c=\mu+\sigma\Phi^{-1}(1-c)`. Thus a result of :math:`-0.021`
    means a €21,000 loss threshold on a €1,000,000 portfolio at 95% confidence.
    A CVaR of :math:`-0.036` means the average tail loss is €36,000. These
    The return type is a signed decimal return, negative for a loss (not a
    positive loss amount); use :func:`as_loss_amount` to convert conventions.
    ``portfolio_var_cvar`` instead returns positive loss fractions.
    Complexity:
    :math:`O(n)` time and :math:`O(n)` input space.
    """
    values = _validated_return_vector(returns)
    confidence = _validate_confidence_level(confidence_level)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1))
    return float(mean + std * norm.ppf(1.0 - confidence))


def calculate_historical_var(
    returns: pd.Series | np.ndarray, confidence_level: float = 0.95
) -> float:
    r"""Calculate empirical VaR as the sample's lower-tail return quantile.

    The result is the signed decimal return quantile
    :math:`VaR_c=Q_{1-c}(R)`, usually negative in a loss tail. For a
    €1,000,000 portfolio, :math:`VaR_{0.95}=-0.021` corresponds to a €21,000
    loss at the empirical 5th-percentile boundary; use :func:`as_loss_amount`
    for its positive loss fraction. An average tail return of
    :math:`-0.036` corresponds to a €36,000 CVaR. Complexity:
    :math:`O(n\log n)` time for quantile selection and :math:`O(n)` space.
    """
    values = _validated_return_vector(returns, minimum_observations=1)
    confidence = _validate_confidence_level(confidence_level)
    return float(np.quantile(values, 1.0 - confidence, method="linear"))


def calculate_cornish_fisher_var(
    returns: pd.Series | np.ndarray, confidence_level: float = 0.95
) -> float:
    r"""Calculate signed VaR with skewness and excess-kurtosis adjustment.

    It returns the signed decimal return quantile :math:`\mu+\sigma z_{CF}`
    (negative for a loss; convert using :func:`as_loss_amount`), where
    :math:`z_{CF}=z+\frac{z^2-1}{6}S+\frac{z^3-3z}{24}K-
    \frac{2z^3-5z}{36}S^2` and :math:`z=\Phi^{-1}(1-c)`. A more severe
    left-tail estimate is more negative: on a €1,000,000 portfolio, moving
    from :math:`-2.1\%` Gaussian VaR to :math:`-3.6\%` adjusted VaR moves the
    loss threshold from €21,000 to €36,000. Complexity: :math:`O(n)` time and
    :math:`O(n)` input space.
    """
    values = _validated_return_vector(returns, minimum_observations=4)
    confidence = _validate_confidence_level(confidence_level)
    quantile, _, _ = _cornish_fisher_quantile(values, confidence)
    return float(quantile)


def calculate_cvar_expected_shortfall(
    returns: pd.Series | np.ndarray,
    confidence_level: float = 0.95,
    method: str = "historical",
) -> float:
    r"""Calculate signed expected shortfall beyond a selected lower-tail VaR.

    The return value is a signed decimal return (typically negative), not a
    positive loss amount; use :func:`as_loss_amount` to convert it.
    Historical CVaR is :math:`E[R\mid R\le Q_{1-c}(R)]`. Under a Gaussian
    model it is :math:`\mu-\sigma\,\phi(z)/(1-c)`, with
    :math:`z=\Phi^{-1}(1-c)`. For a €1,000,000 portfolio, VaR :math:`-2.1\%`
    marks a €21,000 threshold while CVaR :math:`-3.6\%` means the average loss
    in the worst 5% is €36,000. ``method`` accepts ``historical``,
    ``parametric``, or ``cornish_fisher``. Complexity is :math:`O(n\log n)`
    for historical quantiles and :math:`O(n)` for model-based methods.
    """
    values = _validated_return_vector(returns, minimum_observations=1)
    confidence = _validate_confidence_level(confidence_level)
    if method not in {"historical", "parametric", "cornish_fisher"}:
        raise ValueError("method must be 'historical', 'parametric', or 'cornish_fisher'")
    if method == "parametric":
        if values.size < 2:
            raise ValueError("parametric CVaR requires at least 2 observations")
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1))
        z = float(norm.ppf(1.0 - confidence))
        if std == 0.0:
            return mean
        return float(mean - std * norm.pdf(z) / (1.0 - confidence))

    if method == "historical":
        threshold = float(np.quantile(values, 1.0 - confidence, method="linear"))
    else:
        threshold, _, _ = _cornish_fisher_quantile(values, confidence)
    tail = values[values <= threshold]
    if tail.size == 0:
        return float(np.min(values))
    return float(np.mean(tail))


def risk_metrics_telemetry(
    returns: pd.Series | np.ndarray,
) -> dict[str, Any]:
    r"""Summarize return shape, skewness, excess kurtosis, and processing time.

    The helper reports :math:`S`, excess :math:`K`, observation shape, and
    latency in milliseconds so a caller can explain distribution-adjusted
    risk. For example, on a €1,000,000 portfolio a −2.1% VaR is a €21,000
    threshold and −3.6% expected shortfall is €36,000. Complexity:
    :math:`O(n)` time and :math:`O(n)` temporary space.
    """
    start = perf_counter()
    values = _validated_return_vector(returns, minimum_observations=2)
    sample_std = float(np.std(values, ddof=1))
    if sample_std == 0.0:
        sample_skew = 0.0
        excess_kurtosis = 0.0
    else:
        sample_skew = float(skew(values, bias=False)) if values.size >= 3 else float("nan")
        excess_kurtosis = (
            float(kurtosis(values, fisher=True, bias=False))
            if values.size >= 4
            else float("nan")
        )
    return {
        "latency_ms": float((perf_counter() - start) * 1000.0),
        "input_shape": [int(values.size)],
        "n_observations": int(values.size),
        "mean_return": float(np.mean(values)),
        "std_return": sample_std,
        "skewness": sample_skew,
        "excess_kurtosis": excess_kurtosis,
    }


def _matrix(returns: pd.DataFrame | np.ndarray) -> tuple[np.ndarray, list[str], pd.Index | None]:
    r"""Validate a time-by-asset matrix of simple returns.

    The input is :math:`R\in\mathbb{R}^{T\times A}`; every value must be a
    finite simple return greater than :math:`-1`. For example, a column value
    of :math:`-0.021` is a −2.1% daily return. Complexity: :math:`O(TA)` time
    and space.
    """
    if isinstance(returns, pd.DataFrame):
        labels = [str(c) for c in returns.columns]
        index = returns.index
        values = returns.to_numpy(dtype=float, copy=True)
    else:
        values = np.asarray(returns, dtype=float)
        labels = [f"asset_{i}" for i in range(values.shape[1])] if values.ndim == 2 else []
        index = None
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
        raise ValueError("returns must be a 2D observations-by-assets matrix with at least 2 observations")
    if not np.isfinite(values).all() or np.any(values <= -1):
        raise ValueError("returns must be finite and greater than -100%")
    return values, labels, index


def _weights(weights: list[float] | np.ndarray, n_assets: int) -> np.ndarray:
    r"""Validate fully invested portfolio weights.

    The constraint is :math:`\sum_{i=1}^A w_i=1` with one finite weight per
    asset. Equal weights across four assets are :math:`[0.25,0.25,0.25,0.25]`.
    Complexity: :math:`O(A)` time and space.
    """
    w = np.asarray(weights, dtype=float)
    if w.shape != (n_assets,) or not np.isfinite(w).all() or not np.isclose(w.sum(), 1.0, atol=1e-6):
        raise ValueError("weights must be finite, one per asset, and sum to 1")
    return w


def portfolio_var_cvar(
    returns: pd.DataFrame | np.ndarray,
    weights: list[float] | np.ndarray,
    *,
    confidence: float = 0.95,
    method: Method = "historical",
) -> dict[str, float | str | int]:
    r"""Calculate one-period VaR and CVaR as positive loss fractions.

    Historical VaR is :math:`Q_c(-R_p)` and historical CVaR is the mean loss
    in that tail. Parametric VaR is :math:`-\mu_p+z_c\sigma_p`; its Gaussian
    expected shortfall is :math:`-\mu_p+\sigma_p\phi(z_c)/(1-c)`. For a
    €1,000,000 portfolio, a VaR of ``0.021`` means a €21,000 loss threshold.
    Estimates are floored at zero when the entire modeled tail is profitable.
    This function returns non-negative loss fractions (e.g. ``0.021`` means a
    2.1% loss); unlike the scalar functions above, it does not return a signed
    return quantile. Convert with :func:`as_return_quantile` when needed.
    Complexity: :math:`O(TA+T\log T)` for historical
    and :math:`O(TA)` for parametric calculations, with :math:`O(TA)` space.
    """
    values, _, _ = _matrix(returns)
    w = _weights(weights, values.shape[1])
    if not 0 < confidence < 1:
        raise ValueError("confidence must be strictly between 0 and 1")
    if method not in ("historical", "parametric"):
        raise ValueError(f"unsupported method: {method}")
    portfolio_returns = values @ w
    losses = -portfolio_returns
    if method == "historical":
        var = float(np.quantile(losses, confidence, method="linear"))
        tail = losses[losses >= var]
        cvar = float(tail.mean()) if tail.size else var
    else:
        mean = float(portfolio_returns.mean())
        std = float(portfolio_returns.std(ddof=1))
        z = float(norm.ppf(confidence))
        var = -mean + z * std
        cvar = -mean + std * float(norm.pdf(z)) / (1.0 - confidence)
    return {"var": max(0.0, var), "cvar": max(0.0, cvar), "confidence": confidence, "method": method, "observations": int(values.shape[0])}


def marginal_contribution_to_risk(
    returns: pd.DataFrame | np.ndarray,
    weights: list[float] | np.ndarray,
    *,
    periods_per_year: int = 1,
) -> dict[str, object]:
    r"""Calculate marginal and component contributions to portfolio volatility.

    With covariance :math:`\Sigma`, volatility is
    :math:`\sigma_p=\sqrt{w^T\Sigma w}` and marginal contribution is
    :math:`MCR_i=(\Sigma w)_i/\sigma_p`; component contribution is
    :math:`w_iMCR_i`. For example, a 20%-volatility asset at a 50% weight has
    a 10% component contribution when held alone at that weight. Annualization
    multiplies covariance by ``periods_per_year``. Complexity: :math:`O(TA^2
    +A^2)` time and :math:`O(A^2)` space.
    """
    values, labels, _ = _matrix(returns)
    w = _weights(weights, values.shape[1])
    if periods_per_year < 1:
        raise ValueError("periods_per_year must be positive")
    covariance = np.cov(values, rowvar=False, ddof=1) * periods_per_year
    covariance = np.atleast_2d(covariance)
    portfolio_variance = float(w @ covariance @ w)
    volatility = float(np.sqrt(max(portfolio_variance, 0.0)))
    marginal = covariance @ w / volatility if volatility > 0 else np.zeros_like(w)
    component = w * marginal
    return {
        "portfolio_volatility": volatility,
        "marginal_contribution": dict(zip(labels, marginal.tolist())),
        "component_contribution": dict(zip(labels, component.tolist())),
        "component_fraction": dict(zip(labels, (component / volatility).tolist() if volatility else np.zeros_like(w).tolist())),
    }


def stress_test(
    returns: pd.DataFrame,
    weights: list[float] | np.ndarray,
    scenarios: dict[str, tuple[str | date, str | date]],
) -> dict[str, object]:
    r"""Replay fixed portfolio weights over inclusive historical date windows.

    Each scenario maps to ``(start_date, end_date)`` and the input DataFrame
    index must be date-like. The compounded return is
    :math:`\prod_t(1+R_{p,t})-1`; drawdown is the minimum of
    :math:`W_t/\max_{s\le t}W_s-1`. For a €1,000,000 portfolio, a scenario
    total return of −10% represents a €100,000 loss. This is a replay, not a
    forecast. Complexity: :math:`O(TA+ST)` for ``S`` scenarios and ``T`` rows;
    output and intermediate space are :math:`O(T+S)`.
    """
    values, labels, index = _matrix(returns)
    if not isinstance(returns, pd.DataFrame) or index is None:
        raise ValueError("stress_test requires a DataFrame with a date-like index")
    w = _weights(weights, values.shape[1])
    series = pd.Series(values @ w, index=pd.to_datetime(index), name="portfolio_return").sort_index()
    output: dict[str, object] = {}
    for name, (start, end) in scenarios.items():
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        if start_ts > end_ts:
            raise ValueError(f"scenario {name!r} has start_date after end_date")
        window = series.loc[start_ts:end_ts]
        if window.empty:
            output[name] = {"observations": 0, "total_return": None, "max_drawdown": None, "error": "no observations in scenario window"}
            continue
        wealth = (1.0 + window).cumprod()
        drawdown = wealth / wealth.cummax() - 1.0
        output[name] = {
            "start_date": window.index[0].date().isoformat(),
            "end_date": window.index[-1].date().isoformat(),
            "observations": int(len(window)),
            "total_return": float(wealth.iloc[-1] - 1.0),
            "max_drawdown": float(drawdown.min()),
        }
    return {"assets": labels, "scenarios": output}
