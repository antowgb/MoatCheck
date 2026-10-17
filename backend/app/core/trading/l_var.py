"""Pure liquidity-adjusted VaR and portfolio liquidity diagnostics."""

from __future__ import annotations

import math
from time import perf_counter
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.stats import norm

VaRMethod = Literal["parametric", "historical"]


def _as_vector(values: np.ndarray, name: str, *, allow_positive_infinity: bool = False) -> np.ndarray:
    """Convert an input to a checked one-dimensional array in O(N)."""
    try:
        raw = values.to_numpy(dtype=float, copy=True) if isinstance(values, pd.Series) else values
        result = np.asarray(raw, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain numeric values") from exc
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    finite = np.isfinite(result)
    if allow_positive_infinity:
        finite |= np.isposinf(result)
    if not finite.all():
        raise ValueError(f"{name} must contain only finite values" + (" or positive infinity" if allow_positive_infinity else ""))
    return result


def _check_participation(rate: float) -> float:
    """Validate a daily ADV participation fraction in O(1)."""
    try:
        value = float(rate)
    except (TypeError, ValueError) as exc:
        raise ValueError("max_participation_rate must be in (0, 1]") from exc
    if not math.isfinite(value) or not 0.0 < value <= 1.0:
        raise ValueError("max_participation_rate must be in (0, 1]")
    return value


def calculate_liquidation_horizon(
    position_sizes: np.ndarray,
    advs: np.ndarray,
    max_participation_rate: float = 0.10,
) -> np.ndarray:
    r"""Calculate whole trading days needed to liquidate each position.

    For position size :math:`Q_i`, average daily volume :math:`ADV_i`, and
    maximum participation :math:`p`, the horizon is
    :math:`H_i=\lceil |Q_i|/(p\,ADV_i)\rceil`, with active positions bounded
    below by one day. A €2,000,000 position against €500,000 ADV at 10%
    participation needs :math:`\lceil2{,}000{,}000/50{,}000\rceil=40` days.
    Zero positions need zero days; infinite ADV yields a one-day horizon for
    any active position. Complexity: :math:`O(N)` time and space.
    """
    positions = _as_vector(position_sizes, "position_sizes")
    adv = _as_vector(advs, "advs", allow_positive_infinity=True)
    if positions.shape != adv.shape:
        raise ValueError("position_sizes and advs must have matching shapes")
    if np.any(adv <= 0.0):
        raise ValueError("advs must be strictly positive")
    participation = _check_participation(max_participation_rate)
    sizes = np.abs(positions)
    horizon = np.zeros_like(sizes, dtype=int)
    active = sizes > 0.0
    finite_adv = np.isfinite(adv)
    liquid_days = np.ceil(sizes[active & finite_adv] / (participation * adv[active & finite_adv]))
    horizon[active & finite_adv] = np.maximum(liquid_days.astype(int), 1)
    horizon[active & ~finite_adv] = 1
    return horizon


def _var_return_fraction(returns: np.ndarray, confidence: float, method: VaRMethod) -> float:
    """Return VaR as a positive decimal loss fraction; O(N log N) historical."""
    if method == "parametric":
        if returns.size < 2:
            raise ValueError("parametric VaR requires at least two observations")
        lower_return_quantile = float(np.mean(returns) + np.std(returns, ddof=1) * norm.ppf(1.0 - confidence))
    elif method == "historical":
        lower_return_quantile = float(np.quantile(returns, 1.0 - confidence, method="linear"))
    else:
        raise ValueError("method must be 'parametric' or 'historical'")
    return max(0.0, -lower_return_quantile)


def _market_impact_costs(
    positions: np.ndarray,
    adv: np.ndarray,
    volatilities: np.ndarray,
    participation: float,
    gamma: float,
) -> np.ndarray:
    """Sum square-root impact over capped daily tranches with vectorized O(N) work."""
    costs = np.zeros_like(positions, dtype=float)
    sizes = np.abs(positions)
    finite_active = (sizes > 0.0) & np.isfinite(adv)
    if not finite_active.any():
        return costs
    size = sizes[finite_active]
    liquidity = adv[finite_active]
    sigma = volatilities[finite_active]
    capacity = liquidity * participation
    full_days = np.floor(size / capacity)
    remainder = size - full_days * capacity
    full_tranche_cost = capacity * gamma * sigma * np.sqrt(capacity / liquidity)
    partial_tranche_cost = remainder * gamma * sigma * np.sqrt(
        np.divide(remainder, liquidity, out=np.zeros_like(remainder), where=liquidity > 0.0)
    )
    costs[finite_active] = full_days * full_tranche_cost + partial_tranche_cost
    return costs


def calculate_liquidity_adjusted_var(
    portfolio_returns: pd.Series,
    position_values: np.ndarray,
    advs: np.ndarray,
    volatilities: np.ndarray,
    spreads: np.ndarray,
    confidence_level: float = 0.99,
    max_participation_rate: float = 0.10,
    *,
    method: VaRMethod = "parametric",
    impact_gamma: float = 0.5,
) -> dict[str, Any]:
    r"""Calculate liquidity-adjusted VaR in currency units.

    The signed-return VaR is converted to a positive loss fraction :math:`v`.
    With maximum liquidation horizon :math:`H=\max_i H_i`, square-root time
    scaling gives :math:`VaR_H=V_{gross}v\sqrt{H}`. Add half-spread and
    constrained square-root impact costs:
    :math:`LVaR=VaR_H+\sum_i |Q_i|s_i/2+\sum_{i,d}q_{i,d}\gamma\sigma_i
    \sqrt{q_{i,d}/ADV_i}`, where each daily tranche obeys
    :math:`q_{i,d}\le p\,ADV_i`. For a €2,000,000 position with a 2% standard
    one-day VaR, base VaR is €40,000; a 40-day liquidation horizon raises the
    risk term to about €252,982 before spread and impact costs. ``method`` is
    ``"parametric"`` (Gaussian) by default or ``"historical"`` (empirical).
    Complexity: :math:`O(T\log T+N)` time and :math:`O(T+N)` space.
    """
    started = perf_counter()
    if not isinstance(portfolio_returns, pd.Series):
        raise TypeError("portfolio_returns must be a pandas Series")
    try:
        returns = portfolio_returns.to_numpy(dtype=float, na_value=np.nan, copy=True)
    except (TypeError, ValueError) as exc:
        raise TypeError("portfolio_returns must contain numeric values") from exc
    if returns.ndim != 1 or np.isinf(returns).any():
        raise ValueError("portfolio_returns must be one-dimensional and finite wherever present")
    returns = returns[np.isfinite(returns)]
    if returns.size == 0:
        raise ValueError("portfolio_returns must contain at least one finite observation")
    try:
        confidence = float(confidence_level)
    except (TypeError, ValueError) as exc:
        raise ValueError("confidence_level must be strictly between 0 and 1") from exc
    if not math.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("confidence_level must be strictly between 0 and 1")
    participation = _check_participation(max_participation_rate)
    gamma = float(impact_gamma)
    if not math.isfinite(gamma) or gamma < 0.0:
        raise ValueError("impact_gamma must be finite and non-negative")

    positions = _as_vector(position_values, "position_values")
    adv = _as_vector(advs, "advs", allow_positive_infinity=True)
    volatility = _as_vector(volatilities, "volatilities")
    spread = _as_vector(spreads, "spreads")
    if not (positions.shape == adv.shape == volatility.shape == spread.shape):
        raise ValueError("position_values, advs, volatilities, and spreads must have matching shapes")
    if np.any(adv <= 0.0):
        raise ValueError("advs must be strictly positive")
    if np.any(volatility < 0.0) or np.any(spread < 0.0):
        raise ValueError("volatilities and spreads must be non-negative")

    horizon_by_asset = calculate_liquidation_horizon(positions, adv, participation)
    max_horizon = int(horizon_by_asset.max()) if horizon_by_asset.size else 0
    horizon_scale = math.sqrt(max(1, max_horizon))
    gross_position_value = float(np.abs(positions).sum())
    daily_var_fraction = _var_return_fraction(returns, confidence, method)
    standard_var_currency = gross_position_value * daily_var_fraction
    horizon_scaled_var = standard_var_currency * horizon_scale
    spread_costs = np.abs(positions) * spread / 2.0
    impact_costs = _market_impact_costs(positions, adv, volatility, participation, gamma)
    total_spread_cost = float(spread_costs.sum())
    total_impact_cost = float(impact_costs.sum())
    total_liquidation_cost = total_spread_cost + total_impact_cost
    liquidity_adjusted_var = horizon_scaled_var + total_liquidation_cost
    return {
        "liquidity_adjusted_var": float(liquidity_adjusted_var),
        "standard_var": float(standard_var_currency),
        "daily_var_fraction": float(daily_var_fraction),
        "horizon_scaled_var": float(horizon_scaled_var),
        "max_liquidation_days": max_horizon,
        "liquidation_days_by_position": horizon_by_asset,
        "spread_liquidation_cost": total_spread_cost,
        "market_impact_liquidation_cost": total_impact_cost,
        "total_liquidation_cost": float(total_liquidation_cost),
        "method": method,
        "confidence_level": confidence,
        "total_position_value": gross_position_value,
        "input_shape": {
            "portfolio_returns": [int(returns.size)],
            "positions": [int(positions.size)],
        },
        "latency_ms": float((perf_counter() - started) * 1000.0),
    }


def evaluate_portfolio_liquidity_profile(
    portfolio_weights: np.ndarray,
    total_capital: float,
    advs: np.ndarray,
    max_participation_rate: float = 0.10,
    illiquid_days_threshold: int = 5,
) -> dict[str, Any]:
    r"""Summarize weighted liquidation horizon and illiquid asset concentration.

    Position values are :math:`Q_i=|w_i|C`, and each horizon uses
    :math:`H_i=\lceil Q_i/(p\,ADV_i)\rceil`. Weighted average days are
    :math:`\bar H=\sum_i|w_i|H_i/\sum_i|w_i|`; illiquid concentration is the
    absolute-weight share whose liquidation horizon exceeds the configured
    threshold (five days by default). Liquidity score is
    :math:`100/\max(1,\bar H)`, bounded in :math:`[0,100]`. If half of a
    €2,000,000 portfolio takes 40 days to liquidate and half takes one day,
    weighted average is 20.5 days and illiquid concentration is 50%.
    Complexity: :math:`O(N)` time and space.
    """
    started = perf_counter()
    weights = _as_vector(portfolio_weights, "portfolio_weights")
    capital = float(total_capital)
    if not math.isfinite(capital) or capital < 0.0:
        raise ValueError("total_capital must be finite and non-negative")
    adv = _as_vector(advs, "advs", allow_positive_infinity=True)
    if weights.shape != adv.shape:
        raise ValueError("portfolio_weights and advs must have matching shapes")
    if np.any(adv <= 0.0):
        raise ValueError("advs must be strictly positive")
    if (
        isinstance(illiquid_days_threshold, bool)
        or not isinstance(illiquid_days_threshold, int)
        or illiquid_days_threshold < 1
    ):
        raise ValueError("illiquid_days_threshold must be a positive integer")
    position_values = np.abs(weights) * capital
    horizons = calculate_liquidation_horizon(position_values, adv, max_participation_rate)
    gross_weight = float(np.abs(weights).sum())
    weighted_days = (
        float(np.average(horizons, weights=np.abs(weights))) if gross_weight > 0.0 else 0.0
    )
    illiquid_ratio = (
        float(np.abs(weights)[horizons > illiquid_days_threshold].sum() / gross_weight)
        if gross_weight > 0.0
        else 0.0
    )
    return {
        "weighted_average_liquidation_days": weighted_days,
        "illiquid_asset_concentration_ratio": illiquid_ratio,
        "liquidity_score": float(min(100.0, 100.0 / max(1.0, weighted_days))),
        "liquidation_days_by_asset": horizons,
        "illiquid_days_threshold": illiquid_days_threshold,
        "input_shape": [int(weights.size)],
        "latency_ms": float((perf_counter() - started) * 1000.0),
    }


def liquidity_var_telemetry(result: dict[str, Any]) -> dict[str, Any]:
    r"""Create a compact L-VaR telemetry view from a computed risk result.

    The summary exposes calculation latency, maximum liquidation horizon,
    total currency L-VaR, and input shapes without recomputing the model. For
    instance, a 40-day maximum horizon is visible alongside spread and impact
    add-ons. Complexity: :math:`O(1)` time and space.
    """
    if not isinstance(result, dict) or "liquidity_adjusted_var" not in result:
        raise ValueError("result must be returned by calculate_liquidity_adjusted_var")
    return {
        "latency_ms": result["latency_ms"],
        "max_liquidation_days": result["max_liquidation_days"],
        "total_lvar_currency": result["liquidity_adjusted_var"],
        "input_shape": result["input_shape"],
        "spread_liquidation_cost": result["spread_liquidation_cost"],
        "market_impact_liquidation_cost": result["market_impact_liquidation_cost"],
    }
