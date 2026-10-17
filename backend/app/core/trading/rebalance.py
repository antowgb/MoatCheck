"""Discrete target rebalance orders and transparent cost estimates."""

from __future__ import annotations

import math
from time import perf_counter
from typing import Any, Callable

import numpy as np
import pandas as pd


def build_rebalance_orders(
    current_weights: dict[str, float], target_weights: dict[str, float], prices: dict[str, float],
    *, portfolio_value: float, average_daily_volume: dict[str, float] | None = None,
    volatility: dict[str, float] | None = None, transaction_cost_bps: float = 5,
    integer_shares: bool = True,
) -> dict[str, object]:
    r"""Build target-weight orders and estimate commission plus market impact.

    For asset :math:`i`, signed shares are
    :math:`q_i=V(w_i^*-w_i)/P_i`; estimated impact is
    :math:`10{,}000\,\sigma_i\sqrt{|q_i|P_i/(ADV_iP_i)}` basis points,
    and total cost adds commission to impact cost. Every traded asset requires
    a finite positive ADV and finite non-negative volatility; missing or zero
    ADV raises instead of silently assigning zero market impact. For example, rebalancing a
    €1,000,000 account from 40% to 50% in a €100 stock requests about 1,000
    shares before integer rounding. Orders are calculations only: this pure
    function does not send them to a broker. Complexity: :math:`O(A\log A)`
    time due to deterministic asset sorting and :math:`O(A)` output space.
    """
    if portfolio_value <= 0 or not math.isfinite(portfolio_value):
        raise ValueError("portfolio_value must be finite and positive")
    if not target_weights or not math.isclose(sum(target_weights.values()), 1.0, abs_tol=1e-5):
        raise ValueError("target weights must sum to 1")
    if current_weights and not math.isclose(sum(current_weights.values()), 1.0, abs_tol=1e-5):
        raise ValueError("current weights must sum to 1")
    if any(not math.isfinite(x) or x < 0 for data in (current_weights, target_weights) for x in data.values()):
        raise ValueError("weights must be finite and non-negative")
    try:
        transaction_cost_bps = float(transaction_cost_bps)
    except (TypeError, ValueError) as exc:
        raise ValueError("transaction_cost_bps must be finite and between 0 and 10000") from exc
    if not math.isfinite(transaction_cost_bps) or not 0 <= transaction_cost_bps <= 10_000:
        raise ValueError("transaction_cost_bps must be between 0 and 10000")
    adv, vols = average_daily_volume or {}, volatility or {}
    orders, total_cost = [], 0.0
    for asset in sorted(set(current_weights) | set(target_weights)):
        price = prices.get(asset)
        if price is None or not math.isfinite(price) or price <= 0:
            raise ValueError(f"a finite positive price is required for {asset}")
        daily_volume = adv.get(asset)
        daily_vol = vols.get(asset)
        delta = portfolio_value * (target_weights.get(asset, 0) - current_weights.get(asset, 0)) / price
        shares = float(math.trunc(delta)) if integer_shares else delta
        notional = abs(shares * price)
        if notional > 0.0:
            try:
                daily_volume = float(daily_volume)
                daily_vol = float(daily_vol)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"finite positive average_daily_volume and finite non-negative volatility are required for traded asset {asset}"
                ) from exc
            if not math.isfinite(daily_volume) or daily_volume <= 0:
                raise ValueError(f"a finite positive average_daily_volume is required for traded asset {asset}")
            if not math.isfinite(daily_vol) or daily_vol < 0:
                raise ValueError(f"a finite non-negative volatility is required for traded asset {asset}")
            participation = notional / (daily_volume * price)
        else:
            participation = 0.0
            try:
                daily_vol = 0.0 if daily_vol is None else float(daily_vol)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"volatility must be finite and non-negative for {asset}") from exc
            if not math.isfinite(daily_vol) or daily_vol < 0:
                raise ValueError(f"volatility must be finite and non-negative for {asset}")
        impact_bps = 10_000 * daily_vol * math.sqrt(participation)
        commission = notional * transaction_cost_bps / 10_000
        impact_cost = notional * impact_bps / 10_000
        cost = commission + impact_cost
        total_cost += cost
        orders.append({"asset": asset, "side": "BUY" if shares > 0 else "SELL" if shares < 0 else "HOLD",
                       "shares": abs(shares), "signed_shares": shares, "reference_price": price,
                       "notional": notional, "participation_rate": participation,
                       "commission_bps": transaction_cost_bps, "estimated_impact_bps": impact_bps,
                       "estimated_cost": cost})
    return {"orders": orders, "estimated_total_cost": total_cost,
            "estimated_cost_bps_of_portfolio": total_cost / portfolio_value * 10_000,
            "model": "linear commission plus square-root (daily volatility × sqrt(participation)) impact proxy"}


def rebalance_telemetry(
    current_weights: dict[str, float],
    target_weights: dict[str, float],
    prices: dict[str, float],
    *,
    portfolio_value: float,
    average_daily_volume: dict[str, float] | None = None,
    volatility: dict[str, float] | None = None,
    transaction_cost_bps: float = 5,
    integer_shares: bool = True,
) -> dict[str, Any]:
    r"""Report rebalance runtime, dimensions, turnover, and estimated costs.

    One-way turnover is :math:`\frac12\sum_i|w_i^*-w_i|`. The telemetry
    includes execution latency in milliseconds, asset/order counts, turnover,
    and the estimated commission/impact total. For example, moving one
    holding from 40% to 50% contributes five percentage points of one-way
    turnover. Complexity: :math:`O(A\log A)` time and :math:`O(A)` space.
    """
    started = perf_counter()
    result = build_rebalance_orders(
        current_weights,
        target_weights,
        prices,
        portfolio_value=portfolio_value,
        average_daily_volume=average_daily_volume,
        volatility=volatility,
        transaction_cost_bps=transaction_cost_bps,
        integer_shares=integer_shares,
    )
    assets = sorted(set(current_weights) | set(target_weights))
    return {
        "latency_ms": float((perf_counter() - started) * 1000.0),
        "input_shape": [len(assets)],
        "n_assets": len(assets),
        "n_orders": len(result["orders"]),
        "one_way_turnover": float(
            0.5
            * sum(
                abs(target_weights.get(asset, 0.0) - current_weights.get(asset, 0.0))
                for asset in assets
            )
        ),
        "estimated_total_cost": float(result["estimated_total_cost"]),
        "estimated_cost_bps_of_portfolio": float(
            result["estimated_cost_bps_of_portfolio"]
        ),
    }


def _weights(values: np.ndarray | pd.Series, name: str) -> np.ndarray:
    """Validate a finite, non-empty vector summing to one; O(A) time/space."""
    raw = values.to_numpy(dtype=float, copy=True) if isinstance(values, pd.Series) else values
    result = np.asarray(raw, dtype=float)
    if result.ndim != 1 or result.size == 0 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a non-empty finite one-dimensional vector")
    if not np.isclose(result.sum(), 1.0, rtol=0.0, atol=1e-8):
        raise ValueError(f"{name} must sum to 1")
    return result


def calculate_portfolio_turnover(
    previous_weights: np.ndarray, target_weights: np.ndarray
) -> float:
    r"""Calculate one-period gross portfolio turnover from weight changes.

    For previous weights :math:`w_{t-1}` and target weights :math:`w_t^*`,
    turnover is :math:`TO_t=\sum_i|w_{i,t}^*-w_{i,t-1}|`. It is the sum of
    buys and sells, so a complete flip from ``[1, 0]`` to ``[0, 1]`` has
    turnover 2.0; one-way turnover conventions would report half of this.
    In a €1,000,000 portfolio, a 0.20 change in gross turnover represents
    €200,000 traded notional. Complexity: :math:`O(A)` time and :math:`O(A)`
    space for :math:`A` assets.
    """
    previous = np.asarray(previous_weights, dtype=float)
    target = np.asarray(target_weights, dtype=float)
    if previous.ndim != 1 or target.ndim != 1 or previous.shape != target.shape:
        raise ValueError("previous_weights and target_weights must be same-length vectors")
    if not np.isfinite(previous).all() or not np.isfinite(target).all():
        raise ValueError("weight vectors must contain only finite values")
    return float(np.abs(target - previous).sum())


def calculate_rebalancing_drag(
    turnover_series: pd.Series,
    execution_cost_bps: float,
    periods_per_year: int = 12,
) -> float:
    r"""Annualize compounded performance drag from recurring turnover costs.

    Per-period friction is :math:`c_t=TO_t\,b/10{,}000` for gross turnover
    :math:`TO_t` and execution cost :math:`b` bps per traded notional. Its
    annualized drag across :math:`n` observed periods is
    :math:`D_{ann}=1-[\prod_{t=1}^n(1-c_t)]^{m/n}`, where :math:`m` is
    ``periods_per_year`` (12 for monthly rebalancing). For a €1,000,000
    portfolio with monthly turnover 0.20 and 10 bps costs, each rebalance
    costs €200 and the equivalent compounded annual drag is about 0.24%.
    Complexity: :math:`O(n)` time and :math:`O(n)` temporary space.
    """
    if not isinstance(turnover_series, pd.Series):
        raise TypeError("turnover_series must be a pandas Series")
    try:
        turnover = turnover_series.to_numpy(dtype=float, na_value=np.nan, copy=True)
        cost_bps = float(execution_cost_bps)
    except (TypeError, ValueError) as exc:
        raise ValueError("turnover and execution cost must be numeric") from exc
    if not np.isfinite(turnover).all() or np.any(turnover < 0.0):
        raise ValueError("turnover_series must contain finite non-negative values")
    if not math.isfinite(cost_bps) or cost_bps < 0.0:
        raise ValueError("execution_cost_bps must be finite and non-negative")
    if isinstance(periods_per_year, bool) or not isinstance(periods_per_year, int) or periods_per_year < 1:
        raise ValueError("periods_per_year must be a positive integer")
    if turnover.size == 0 or cost_bps == 0.0:
        return 0.0
    period_costs = turnover * cost_bps / 10_000.0
    if np.any(period_costs >= 1.0):
        raise ValueError("per-period transaction drag must be less than 100%")
    cumulative_retention = float(np.prod(1.0 - period_costs))
    return float(1.0 - cumulative_retention ** (periods_per_year / turnover.size))


class RebalancingEngine:
    r"""Simulate multi-period portfolio rebalancing and transaction friction.

    Target row :math:`t` is known at the start of the interval from price row
    :math:`t` to :math:`t+1`; this timing prevents using future prices to set
    current holdings. Asset simple returns are :math:`r_{i,t}=P_{i,t+1}/P_{i,t}-1`,
    portfolio wealth evolves as :math:`W_{t+1}=W_t(1+w_t^Tr_t)-C_t`, and gross
    turnover is :math:`\sum_i|w_{i,t}^*-w_{i,t}^{drift}|`. A cap scales the
    rebalance delta toward the requested target while preserving full
    investment. For a €1,000,000 monthly portfolio, 20% turnover at 10 bps
    costs €200; over 12 periods the compounded drag is approximately 0.24%.
    A custom transaction model receives signed trade quantities and current
    prices and must return a non-negative currency cost; the default charges
    the configured bps on total traded notional. Complexity: :math:`O(TA)`
    time and :math:`O(TA)` memory for :math:`T` periods and :math:`A` assets.
    """

    def __init__(
        self,
        initial_weights: np.ndarray | pd.Series,
        target_weights: np.ndarray | pd.DataFrame,
        prices: np.ndarray | pd.DataFrame,
        *,
        initial_wealth: float = 1_000_000.0,
        execution_cost_bps: float = 5.0,
        max_turnover: float | None = None,
        transaction_costs_model: Callable[[np.ndarray, np.ndarray], float] | None = None,
        periods_per_year: int = 12,
    ) -> None:
        """Initialize validated targets, prices, cost model, and constraints; O(TA)."""
        self.initial_weights = _weights(initial_weights, "initial_weights")
        target_values = (
            target_weights.to_numpy(dtype=float, copy=True)
            if isinstance(target_weights, pd.DataFrame)
            else np.asarray(target_weights, dtype=float)
        )
        price_values = (
            prices.to_numpy(dtype=float, copy=True)
            if isinstance(prices, pd.DataFrame)
            else np.asarray(prices, dtype=float)
        )
        if (
            target_values.ndim != 2
            or target_values.shape[0] == 0
            or target_values.shape[1] != self.initial_weights.size
        ):
            raise ValueError("target_weights must be a periods-by-assets matrix")
        if not np.isfinite(target_values).all() or not np.allclose(
            target_values.sum(axis=1), 1.0, rtol=0.0, atol=1e-8
        ):
            raise ValueError("every target_weights row must be finite and sum to 1")
        if price_values.shape != (target_values.shape[0] + 1, target_values.shape[1]):
            raise ValueError("prices must have one more time row than target_weights")
        if not np.isfinite(price_values).all() or np.any(price_values <= 0.0):
            raise ValueError("prices must contain finite positive values")
        try:
            wealth = float(initial_wealth)
            fee_bps = float(execution_cost_bps)
        except (TypeError, ValueError) as exc:
            raise ValueError("initial_wealth and execution_cost_bps must be numeric") from exc
        if not math.isfinite(wealth) or wealth <= 0.0:
            raise ValueError("initial_wealth must be finite and positive")
        if not math.isfinite(fee_bps) or fee_bps < 0.0:
            raise ValueError("execution_cost_bps must be finite and non-negative")
        if max_turnover is not None:
            try:
                turnover_cap = float(max_turnover)
            except (TypeError, ValueError) as exc:
                raise ValueError("max_turnover must be finite and non-negative") from exc
            if not math.isfinite(turnover_cap) or turnover_cap < 0.0:
                raise ValueError("max_turnover must be finite and non-negative")
        else:
            turnover_cap = None
        if isinstance(periods_per_year, bool) or not isinstance(periods_per_year, int) or periods_per_year < 1:
            raise ValueError("periods_per_year must be a positive integer")
        if transaction_costs_model is not None and not callable(transaction_costs_model):
            raise TypeError("transaction_costs_model must be callable")

        self.target_weights = target_values
        self.prices = price_values
        self.initial_wealth = wealth
        self.execution_cost_bps = fee_bps
        self.max_turnover = turnover_cap
        self.transaction_costs_model = transaction_costs_model
        self.periods_per_year = periods_per_year

    def run(self) -> dict[str, Any]:
        r"""Run the path-dependent simulation and return wealth and telemetry.

        Each target is applied before its matching price interval, transaction
        costs are deducted from ending wealth, and post-return weights drift
        with asset performance before the next rebalance. Turnover constraints
        clip the one-period :math:`L_1` weight change. Returned cash flows
        record costs as negative cash outflows, while turnover statistics use
        the gross buy-plus-sell convention. Complexity: :math:`O(TA)` time and
        :math:`O(TA)` space for paths, weights, and per-period records.
        """
        started = perf_counter()
        asset_returns = self.prices[1:] / self.prices[:-1] - 1.0
        wealth = self.initial_wealth
        current_weights = self.initial_weights.copy()
        wealth_values = [wealth]
        wealth_index: list[int] = [-1]
        cashflow_rows: list[dict[str, float | int]] = []
        turnover_values: list[float] = []
        transaction_cost_values: list[float] = []
        executed_targets: list[np.ndarray] = []

        for period, requested in enumerate(self.target_weights):
            target = requested.copy()
            difference = target - current_weights
            turnover = float(np.abs(difference).sum())
            if self.max_turnover is not None and turnover > self.max_turnover:
                if turnover == 0.0:
                    executed = current_weights.copy()
                else:
                    executed = current_weights + difference * (self.max_turnover / turnover)
                turnover = float(np.abs(executed - current_weights).sum())
            else:
                executed = target

            signed_trade_values = (executed - current_weights) * wealth
            trade_quantities = signed_trade_values / self.prices[period]
            if self.transaction_costs_model is None:
                cost = float(np.abs(signed_trade_values).sum() * self.execution_cost_bps / 10_000.0)
            else:
                cost = float(self.transaction_costs_model(trade_quantities.copy(), self.prices[period].copy()))
            if not math.isfinite(cost) or cost < 0.0:
                raise ValueError("transaction_costs_model must return a finite non-negative currency cost")

            period_return = float(executed @ asset_returns[period])
            if not math.isfinite(period_return) or period_return < -1.0:
                raise ValueError("portfolio simple return must be finite and at least -100%")
            gross_ending_wealth = wealth * (1.0 + period_return)
            net_ending_wealth = gross_ending_wealth - cost
            if net_ending_wealth < 0.0:
                raise ValueError("transaction costs exceed portfolio wealth")

            cashflow_rows.append({
                "period": period,
                "starting_wealth": wealth,
                "turnover": turnover,
                "gross_portfolio_return": period_return,
                "transaction_cost": cost,
                "transaction_cost_cash_flow": -cost,
                "ending_wealth": net_ending_wealth,
            })
            turnover_values.append(turnover)
            transaction_cost_values.append(cost)
            executed_targets.append(executed.copy())
            wealth = net_ending_wealth
            wealth_values.append(wealth)
            wealth_index.append(period)

            gross_growth = 1.0 + period_return
            if gross_growth > 0.0:
                current_weights = executed * (1.0 + asset_returns[period]) / gross_growth
                # Floating point drift can perturb fully invested weights slightly.
                current_weights /= current_weights.sum()
            else:
                current_weights = executed.copy()

        turnovers = pd.Series(turnover_values, name="turnover", dtype=float)
        cash_flows = pd.DataFrame(cashflow_rows).set_index("period")
        total_cost = float(sum(transaction_cost_values))
        total_traded_notional = float(
            sum(row["starting_wealth"] * row["turnover"] for row in cashflow_rows)
        )
        average_annual_turnover = (
            float(turnovers.mean() * self.periods_per_year) if not turnovers.empty else 0.0
        )
        return {
            "wealth_path": pd.Series(wealth_values, index=wealth_index, name="net_wealth"),
            "cash_flows": cash_flows,
            "turnover_series": turnovers,
            "executed_target_weights": np.asarray(executed_targets, dtype=float),
            "total_turnover": float(turnovers.sum()),
            "average_annual_turnover": average_annual_turnover,
            "total_transaction_cost": total_cost,
            "total_transaction_drag_bps": float(total_cost / self.initial_wealth * 10_000.0),
            "execution_cost_bps_of_traded_notional": (
                float(total_cost / total_traded_notional * 10_000.0)
                if total_traded_notional > 0.0
                else 0.0
            ),
            "input_shape": {
                "initial_weights": [int(self.initial_weights.size)],
                "target_weights": [int(self.target_weights.shape[0]), int(self.target_weights.shape[1])],
                "prices": [int(self.prices.shape[0]), int(self.prices.shape[1])],
            },
            "latency_ms": float((perf_counter() - started) * 1000.0),
        }
