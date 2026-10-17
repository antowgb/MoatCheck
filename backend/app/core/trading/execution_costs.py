"""Pure market-impact, spread, fee, and execution-drag calculations."""

from __future__ import annotations

import math
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd


def _finite_nonnegative(value: float, name: str) -> float:
    """Convert and validate a finite non-negative scalar in O(1)."""
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and non-negative") from exc
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _order_vector(
    values: np.ndarray | pd.Series, name: str, *, allow_positive_infinity: bool = False
) -> np.ndarray:
    """Convert an order input to a finite 1D vector; O(N) time and space."""
    try:
        raw = values.to_numpy(dtype=float, copy=True) if isinstance(values, pd.Series) else values
        result = np.asarray(raw, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain numeric values") from exc
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    valid = (
        np.isfinite(result) | (np.isposinf(result) if allow_positive_infinity else False)
    )
    if not valid.all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def estimate_half_spread_cost(trade_value: float, bid_ask_spread_pct: float) -> float:
    r"""Estimate the deterministic cost of crossing half the quoted spread.

    ``bid_ask_spread_pct`` is a decimal fraction, so the crossing cost is
    :math:`C_{spread}=|V|s/2`. A €50,000 order with a 10 bp full spread
    (:math:`s=0.001`) pays an estimated €25 spread cost. A market buy pays
    half-spread above mid and a sell half-spread below mid. Complexity:
    :math:`O(1)` time and space.
    """
    value = _finite_nonnegative(trade_value, "trade_value")
    spread = _finite_nonnegative(bid_ask_spread_pct, "bid_ask_spread_pct")
    return float(value * spread / 2.0)


def calculate_square_root_slippage(
    trade_value: float,
    adv: float,
    daily_volatility: float,
    impact_gamma: float = 0.5,
) -> float:
    r"""Estimate fractional market impact under the square-root law.

    The Almgren-Chriss-style participation approximation is
    :math:`I=\gamma\sigma_d\sqrt{|V|/ADV}`, where trade value :math:`V` and
    average daily value traded :math:`ADV` must use the same currency units.
    This function returns the impact fraction :math:`I` (not currency cost or
    bps); multiply it by :math:`|V|` for the slippage amount. A €50,000 order,
    €1,000,000 ADV, 2% daily volatility, and :math:`\gamma=0.5` imply about
    0.224% impact. Four times the order value doubles :math:`I`, while its
    currency cost rises eightfold. Complexity: :math:`O(1)` time and space.
    """
    value = _finite_nonnegative(trade_value, "trade_value")
    volatility = _finite_nonnegative(daily_volatility, "daily_volatility")
    gamma = _finite_nonnegative(impact_gamma, "impact_gamma")
    try:
        liquidity = float(adv)
    except (TypeError, ValueError) as exc:
        raise ValueError("adv must be strictly positive and not NaN") from exc
    if np.isnan(liquidity) or liquidity <= 0.0:
        raise ValueError("adv must be strictly positive and not NaN")
    participation = 0.0 if math.isinf(liquidity) else value / liquidity
    return float(gamma * volatility * math.sqrt(participation))


def calculate_total_execution_drag(
    trade_sizes: np.ndarray,
    prices: np.ndarray,
    advs: np.ndarray,
    volatilities: np.ndarray,
    spreads: np.ndarray,
    fixed_fee_per_trade: float = 1.0,
    pct_broker_fee: float = 0.0005,
) -> dict[str, Any]:
    r"""Aggregate order spread, impact, fixed fees, and percentage fees.

    Order quantity :math:`q_i` and ADV :math:`ADV_i` use the same units;
    mid-price :math:`P_i` converts quantity to notional
    :math:`V_i=|q_i|P_i`. The order impact fraction is
    :math:`I_i=\gamma\sigma_i\sqrt{|q_i|/ADV_i}` and the net execution price
    (including all costs allocated per unit) is approximately
    :math:`P_{net,i}=P_i+\operatorname{sgn}(q_i)C_i/|q_i|`. For a €50,000
    order with a 10 bp full spread, 0.224% impact, €1 fixed commission, and
    5 bp broker fee, the breakdown is €25 spread, about €112 impact, €1 fixed,
    and €25 percentage fee, or roughly €163 total. Zero-size orders incur no
    costs. Complexity: :math:`O(N)` time and :math:`O(N)` space for ``N``
    orders.
    """
    started = perf_counter()
    arrays = {
        "trade_sizes": _order_vector(trade_sizes, "trade_sizes"),
        "prices": _order_vector(prices, "prices"),
        "advs": _order_vector(advs, "advs", allow_positive_infinity=True),
        "volatilities": _order_vector(volatilities, "volatilities"),
        "spreads": _order_vector(spreads, "spreads"),
    }
    lengths = {len(value) for value in arrays.values()}
    if len(lengths) != 1:
        raise ValueError("all order input arrays must have equal length")
    sizes = arrays["trade_sizes"]
    price = arrays["prices"]
    adv = arrays["advs"]
    volatility = arrays["volatilities"]
    spread = arrays["spreads"]
    if np.any(price <= 0.0):
        raise ValueError("prices must be strictly positive")
    if np.any(adv <= 0.0):
        raise ValueError("advs must be strictly positive (infinite ADV is allowed)")
    if np.any(volatility < 0.0) or np.any(spread < 0.0):
        raise ValueError("volatilities and spreads must be non-negative")
    fixed_fee = _finite_nonnegative(fixed_fee_per_trade, "fixed_fee_per_trade")
    pct_fee = _finite_nonnegative(pct_broker_fee, "pct_broker_fee")

    quantities = np.abs(sizes)
    notionals = quantities * price
    active = quantities > 0.0
    impact_fractions = np.zeros_like(notionals)
    positive_adv = np.isfinite(adv)
    impact_fractions[active & positive_adv] = (
        0.5
        * volatility[active & positive_adv]
        * np.sqrt(quantities[active & positive_adv] / adv[active & positive_adv])
    )
    spread_costs = notionals * spread / 2.0
    impact_costs = notionals * impact_fractions
    fixed_fees = active.astype(float) * fixed_fee
    percentage_fees = notionals * pct_fee
    # Zero-sized orders are economically inactive and incur no fee components.
    spread_costs[~active] = 0.0
    impact_costs[~active] = 0.0
    percentage_fees[~active] = 0.0
    total_order_costs = spread_costs + impact_costs + fixed_fees + percentage_fees
    total_notional = float(notionals.sum())
    total_cost = float(total_order_costs.sum())
    cost_breakdown = {
        "spread_cost": float(spread_costs.sum()),
        "market_impact_cost": float(impact_costs.sum()),
        "fixed_broker_fees": float(fixed_fees.sum()),
        "percentage_broker_fees": float(percentage_fees.sum()),
    }
    net_prices = np.full_like(price, np.nan)
    net_prices[active] = price[active] + np.sign(sizes[active]) * (
        total_order_costs[active] / quantities[active]
    )
    return {
        "total_execution_drag": total_cost,
        "total_traded_notional": total_notional,
        "total_friction_bps": (total_cost / total_notional * 10_000.0)
        if total_notional > 0.0
        else 0.0,
        "cost_breakdown": cost_breakdown,
        "per_order_costs": total_order_costs,
        "impact_fraction_by_order": impact_fractions,
        "effective_execution_prices": net_prices,
        "n_orders": int(sizes.size),
        "n_active_orders": int(active.sum()),
        "input_shape": [int(sizes.size)],
        "latency_ms": float((perf_counter() - started) * 1000.0),
    }


def execution_cost_telemetry(
    trade_sizes: np.ndarray,
    prices: np.ndarray,
    advs: np.ndarray,
    volatilities: np.ndarray,
    spreads: np.ndarray,
    fixed_fee_per_trade: float = 1.0,
    pct_broker_fee: float = 0.0005,
) -> dict[str, Any]:
    r"""Expose execution-cost latency, shapes, bps, and source-level breakdown.

    The returned diagnostics distinguish spread, nonlinear impact, fixed
    commission, and percentage commission so execution friction is
    auditable. Total friction is :math:`10{,}000\,C_{total}/V_{traded}` bps.
    For a €50,000 trade with costs around €163, friction is approximately
    32.6 bps. Complexity: :math:`O(N)` time and space.
    """
    return calculate_total_execution_drag(
        trade_sizes,
        prices,
        advs,
        volatilities,
        spreads,
        fixed_fee_per_trade=fixed_fee_per_trade,
        pct_broker_fee=pct_broker_fee,
    )
