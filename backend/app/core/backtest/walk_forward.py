"""Point-in-time top-N walk-forward backtest over aligned return arrays."""

from __future__ import annotations

import numpy as np
import pandas as pd
from time import perf_counter


def run_walk_forward(
    returns: pd.DataFrame | np.ndarray,
    scores: pd.DataFrame | np.ndarray,
    *,
    assets: list[str] | None = None,
    lookback_periods: int = 52,
    rebalance_every: int = 13,
    top_n: int = 5,
    transaction_cost_bps: float = 10,
) -> dict[str, object]:
    r"""Use prior-observation scores to select and compound an equal-weight basket.

    At each rebalance, the holdings are selected using score row :math:`t-1`
    and applied to return row :math:`t`. Portfolio return is
    :math:`r_{p,t}=w_{t-1}^Tr_t`; the equity curve compounds returns after
    transaction costs. This explicit one-period lag prevents using a score
    that would not yet have been known when earning the period's return. For
    example, a top-five ranking observed at Friday's close applies to the
    following return period, not Friday's already-realized return. Complexity:
    :math:`O(TA\log A)` time for per-rebalance ranking and :math:`O(T+A)`
    output space, where :math:`T` is periods and :math:`A` is assets.
    """
    r = returns.to_numpy(dtype=float, copy=True) if isinstance(returns, pd.DataFrame) else np.asarray(returns, dtype=float)
    s = scores.to_numpy(dtype=float, copy=True) if isinstance(scores, pd.DataFrame) else np.asarray(scores, dtype=float)
    labels = assets or (list(map(str, returns.columns)) if isinstance(returns, pd.DataFrame) else [f"asset_{i}" for i in range(r.shape[1])])
    if r.ndim != 2 or s.shape != r.shape or r.shape[1] != len(labels) or not np.isfinite(r).all():
        raise ValueError("returns and scores must be finite, same-shaped period-by-asset matrices")
    if lookback_periods < 1 or rebalance_every < 1 or top_n < 1 or top_n > r.shape[1]:
        raise ValueError("lookback/rebalance/top_n values are outside valid bounds")
    if r.shape[0] <= lookback_periods:
        raise ValueError("returns must extend beyond lookback_periods")
    if transaction_cost_bps < 0 or transaction_cost_bps > 10_000:
        raise ValueError("transaction_cost_bps must be between 0 and 10000")
    equity, prior_weights, turnover = 1.0, np.zeros(r.shape[1]), 0.0
    cost_rate, curve, holdings = transaction_cost_bps / 10_000, [], []
    first_period = lookback_periods
    for period in range(first_period, len(r)):
        if period == first_period or (period - first_period) % rebalance_every == 0:
            row = s[period - 1]
            candidates = np.flatnonzero(np.isfinite(row))
            if candidates.size == 0:
                selected = np.flatnonzero(prior_weights)
                new_weights = prior_weights.copy()
            else:
                selected = candidates[np.argsort(row[candidates])[-top_n:]]
                new_weights = np.zeros(r.shape[1])
                new_weights[selected] = 1 / len(selected)
            traded = float(np.abs(new_weights - prior_weights).sum())
            if period > first_period:
                equity *= max(0.0, 1 - cost_rate * traded)
                turnover += traded
            prior_weights = new_weights
            holdings.append({"period": period, "assets": [labels[i] for i in selected]})
        period_return = float(prior_weights @ r[period])
        equity *= 1 + period_return
        curve.append({"period": period, "return": period_return, "equity": equity})
    return {"equity_curve": curve, "holdings": holdings, "total_return": equity - 1,
            "turnover": turnover, "transaction_cost_bps": transaction_cost_bps,
            "lookahead_policy": "selection uses scores from the preceding observation"}


def walk_forward_telemetry(
    returns: pd.DataFrame | np.ndarray,
    scores: pd.DataFrame | np.ndarray,
    *,
    assets: list[str] | None = None,
    lookback_periods: int = 52,
    rebalance_every: int = 13,
    top_n: int = 5,
    transaction_cost_bps: float = 10,
) -> dict[str, object]:
    r"""Report walk-forward execution latency, shapes, turnover, and returns.

    The helper runs the same causal backtest and summarizes its main outputs.
    Its dimensions are the supplied period-by-asset matrix shape and the
    number of realized test periods; it also reports total return and turnover
    after costs. The one-period score lag remains enforced by
    :func:`run_walk_forward`. Complexity: :math:`O(TA\log A)` time and
    :math:`O(T+A)` output space.
    """
    started = perf_counter()
    result = run_walk_forward(
        returns,
        scores,
        assets=assets,
        lookback_periods=lookback_periods,
        rebalance_every=rebalance_every,
        top_n=top_n,
        transaction_cost_bps=transaction_cost_bps,
    )
    if isinstance(returns, pd.DataFrame):
        input_shape = [int(returns.shape[0]), int(returns.shape[1])]
    else:
        values = np.asarray(returns)
        input_shape = [int(values.shape[0]), int(values.shape[1])]
    return {
        "latency_ms": float((perf_counter() - started) * 1000.0),
        "input_shape": input_shape,
        "n_test_periods": len(result["equity_curve"]),
        "n_rebalances": len(result["holdings"]),
        "total_return": result["total_return"],
        "turnover": result["turnover"],
        "lookahead_policy": result["lookahead_policy"],
    }
