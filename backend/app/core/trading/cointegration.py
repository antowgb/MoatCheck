"""Pure cointegration diagnostics and event-driven pairs strategy simulation."""

from __future__ import annotations

import math
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.stattools import adfuller, coint


def _aligned_prices(price_a: pd.Series, price_b: pd.Series) -> pd.DataFrame:
    """Inner-align and validate price levels in chronological order; O(T)."""
    if not isinstance(price_a, pd.Series) or not isinstance(price_b, pd.Series):
        raise TypeError("price_a and price_b must be pandas Series")
    if not price_a.index.is_unique or not price_b.index.is_unique:
        raise ValueError("price indexes must be unique")
    if not price_a.index.is_monotonic_increasing or not price_b.index.is_monotonic_increasing:
        raise ValueError("price indexes must be in chronological order")
    try:
        joined = pd.concat(
            [price_a.rename("price_a"), price_b.rename("price_b")], axis=1, join="inner"
        ).dropna()
        values = joined.to_numpy(dtype=float, copy=True)
    except (TypeError, ValueError) as exc:
        raise TypeError("prices must contain numeric values") from exc
    if values.ndim != 2 or values.shape[0] < 3:
        raise ValueError("at least 3 aligned non-missing price pairs are required")
    if not np.isfinite(values).all() or np.any(values <= 0.0):
        raise ValueError("prices must be finite and strictly positive")
    return joined


def test_engle_granger_cointegration(price_a: pd.Series, price_b: pd.Series) -> dict[str, Any]:
    r"""Test whether two nonstationary price series share a stationary spread.

    Step one estimates :math:`P_{A,t}=\alpha+\beta P_{B,t}+\varepsilon_t`
    by OLS. Step two tests residual stationarity using the Engle-Granger
    cointegration test. A pair of banking stocks whose prices wander together
    can have a stable residual even though neither individual price is
    stationary; the p-value below 0.05 flags evidence of such a long-run link.
    The output also reports an ADF statistic computed on the OLS residuals.
    Exact linear dependence is treated as deterministic cointegration and
    handled without invoking degenerate ADF calculations. Complexity:
    :math:`O(T)` for OLS plus approximately :math:`O(T\log T)` for the
    automatic-lag residual test; working space is :math:`O(T)`.
    """
    started = perf_counter()
    aligned = _aligned_prices(price_a, price_b)
    y = aligned["price_a"].to_numpy(dtype=float)
    x = aligned["price_b"].to_numpy(dtype=float)
    if np.var(x) <= np.finfo(float).eps or np.var(y) <= np.finfo(float).eps:
        raise ValueError("both price series must have non-zero variance")
    design = sm.add_constant(x, has_constant="add")
    regression = OLS(y, design, missing="raise").fit()
    intercept = float(regression.params[0])
    beta = float(regression.params[1])
    residuals = np.asarray(regression.resid, dtype=float)
    residual_variance = float(np.var(residuals))
    scale = max(float(np.var(y)), float(np.var(x)), 1.0)
    deterministic = residual_variance <= np.finfo(float).eps * scale

    if deterministic:
        eg_statistic, p_value, critical_values = float("-inf"), 0.0, np.array([np.nan] * 3)
        residual_adf_statistic = float("nan")
    else:
        eg_statistic, p_value, critical_values = coint(y, x, trend="c", autolag="aic")
        adf_result = adfuller(
            residuals,
            regression="n",
            autolag="AIC",
            result_object=True,
        )
        residual_adf_statistic = float(adf_result.statistic)
    return {
        "alpha": intercept,
        "hedge_ratio": beta,
        "beta": beta,
        "p_value": float(p_value),
        "test_statistic": float(eg_statistic),
        "residual_adf_statistic": residual_adf_statistic,
        "critical_values": np.asarray(critical_values, dtype=float),
        "is_cointegrated": bool(p_value < 0.05),
        "n_observations": int(len(aligned)),
        "input_shape": [int(len(aligned)), 2],
        "latency_ms": float((perf_counter() - started) * 1000.0),
    }


def calculate_spread_zscore(
    price_a: pd.Series,
    price_b: pd.Series,
    hedge_ratio: float | None = None,
    window: int = 60,
    *,
    calibration_window: int | None = None,
) -> pd.Series:
    r"""Calculate a trailing rolling Z-score for the OLS hedge spread.

    The spread is :math:`S_t=P_{A,t}-\beta_t P_{B,t}` and the trailing score is
    :math:`Z_t=(S_t-\bar S_{t,w})/s_{t,w}`, using the latest ``window``
    observations including the current close. In a bank pair with :math:`Z=2.2`,
    short the rich bank/long the cheap bank; exit when :math:`|Z|<0.5`, or stop
    if the divergence reaches 3.5 standard deviations. A zero rolling spread
    deviation produces NaN rather than an infinite signal. Pass a fixed beta
    only when it was estimated using data available before the evaluated
    window. If ``hedge_ratio=None``, beta is estimated at every date from
    strictly prior observations: expanding history by default, or the trailing
    ``calibration_window`` observations when specified. For example, beta used
    for today's bank-pair score is estimated from past prices only, never from
    tomorrow's prices. Early scores remain NaN until there are at least three
    calibration observations and a complete Z-score window. Complexity is
    :math:`O(TW_c+T)` time and :math:`O(T)` space for a direct rolling/expanding
    regression implementation, where :math:`W_c` is calibration history size.
    """
    if isinstance(window, bool) or not isinstance(window, int) or window < 2:
        raise ValueError("window must be an integer of at least 2")
    aligned = _aligned_prices(price_a, price_b)
    if window > len(aligned):
        raise ValueError("window cannot exceed the number of aligned prices")
    if hedge_ratio is not None and calibration_window is not None:
        raise ValueError("provide either a fixed hedge_ratio or calibration_window, not both")
    if hedge_ratio is None:
        betas = estimate_point_in_time_hedge_ratio(
            price_a, price_b, calibration_window=calibration_window
        ).reindex(aligned.index)
        spread = aligned["price_a"] - betas * aligned["price_b"]
    else:
        try:
            beta = float(hedge_ratio)
        except (TypeError, ValueError) as exc:
            raise ValueError("hedge_ratio must be finite") from exc
        if not math.isfinite(beta):
            raise ValueError("hedge_ratio must be finite")
        spread = aligned["price_a"] - beta * aligned["price_b"]
    rolling = spread.rolling(window=window, min_periods=window)
    mean = rolling.mean()
    std = rolling.std(ddof=1)
    zscore = (spread - mean) / std.replace(0.0, np.nan)
    zscore.name = "spread_zscore"
    if hedge_ratio is None:
        zscore.attrs["hedge_ratios"] = betas.copy()
        zscore.attrs["calibration_window"] = calibration_window
    return zscore


def estimate_point_in_time_hedge_ratio(
    price_a: pd.Series,
    price_b: pd.Series,
    calibration_window: int | None = None,
) -> pd.Series:
    r"""Estimate a causal OLS hedge ratio at every date using prior prices only.

    At time :math:`t`, the slope
    :math:`\hat\beta_t=\operatorname{Cov}(P_{A,<t},P_{B,<t})/
    \operatorname{Var}(P_{B,<t})` is fitted on observations strictly before
    :math:`t`; it never uses the current or future price. ``None`` selects an
    expanding calibration window, while an integer selects a rolling window.
    For example, the beta applied to today's two bank prices is calibrated only
    on prior dates, so changing tomorrow's prices cannot alter today's beta.
    Values are NaN until three prior aligned observations are available or if
    the calibration asset has zero variance. Complexity: :math:`O(TW_c)` time
    with repeated OLS fits and :math:`O(T+W_c)` space, where :math:`W_c` is the
    calibration-window length (at most :math:`T` for expanding estimation).
    """
    aligned = _aligned_prices(price_a, price_b)
    if calibration_window is not None and (
        isinstance(calibration_window, bool)
        or not isinstance(calibration_window, int)
        or calibration_window < 3
    ):
        raise ValueError("calibration_window must be an integer of at least 3")
    x = aligned["price_b"].to_numpy(dtype=float)
    y = aligned["price_a"].to_numpy(dtype=float)
    betas = np.full(len(aligned), np.nan, dtype=float)
    for t in range(3, len(aligned)):
        start = 0 if calibration_window is None else max(0, t - calibration_window)
        x_history = x[start:t]
        y_history = y[start:t]
        if np.var(x_history) <= np.finfo(float).eps:
            continue
        design = sm.add_constant(x_history, has_constant="add")
        betas[t] = float(OLS(y_history, design, missing="raise").fit().params[1])
    return pd.Series(betas, index=aligned.index, name="hedge_ratio")


def simulate_stat_arb_strategy(
    price_a: pd.Series,
    price_b: pd.Series,
    zscore: pd.Series,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.5,
    stop_loss_threshold: float = 3.5,
    *,
    hedge_ratio: float | pd.Series | None = None,
) -> dict[str, Any]:
    r"""Simulate a causal mean-reverting pairs strategy from spread Z-scores.

    Signals at close :math:`t` change only the position earning the next
    interval's PnL. For position :math:`q\in\{-1,0,1\}`, the held-beta
    calculation is :math:`\Pi_t=q_{t-1}(\Delta P_{A,t}-\beta_{t-1}
    \Delta P_{B,t})`. A beta series attached to a Z-score from
    :func:`calculate_spread_zscore` is used by default; absent that metadata,
    beta defaults to one for backward compatibility. A dynamic beta known at
    close :math:`t-1` is held over the next interval. For example, a short
    bank-pair spread entered at :math:`Z=2.2` with :math:`\beta=1.4` shorts one
    A share and holds 1.4 B shares. PnL is in spread-price units per share of A
    and excludes costs from changing hedge ratios. Complexity: :math:`O(T)`
    time and space.
    """
    started = perf_counter()
    aligned = _aligned_prices(price_a, price_b)
    if not isinstance(zscore, pd.Series):
        raise TypeError("zscore must be a pandas Series")
    if not zscore.index.is_unique or not zscore.index.is_monotonic_increasing:
        raise ValueError("zscore index must be unique and chronological")
    aligned_strategy = aligned.join(zscore.rename("zscore"), how="inner")
    if len(aligned_strategy) < 2:
        raise ValueError("at least two aligned prices and z-scores are required")
    z = aligned_strategy["zscore"].to_numpy(dtype=float, na_value=np.nan, copy=True)
    if np.isinf(z).any():
        raise ValueError("zscore must be finite wherever present")
    try:
        entry = float(entry_threshold)
        exit_level = float(exit_threshold)
        stop = float(stop_loss_threshold)
    except (TypeError, ValueError) as exc:
        raise ValueError("thresholds must be finite numbers") from exc
    if not all(math.isfinite(value) for value in (entry, exit_level, stop)):
        raise ValueError("thresholds must be finite numbers")
    if entry <= 0.0 or exit_level < 0.0 or exit_level >= entry:
        raise ValueError("thresholds must satisfy entry > exit >= 0")
    if stop <= entry:
        raise ValueError("stop_loss_threshold must exceed entry_threshold")

    a = aligned_strategy["price_a"].to_numpy(dtype=float)
    b = aligned_strategy["price_b"].to_numpy(dtype=float)
    if hedge_ratio is None:
        beta_input = zscore.attrs.get("hedge_ratios")
        if beta_input is None:
            beta_values = np.ones(len(aligned_strategy), dtype=float)
        else:
            beta_input = beta_input if isinstance(beta_input, pd.Series) else pd.Series(beta_input)
            beta_values = beta_input.reindex(aligned_strategy.index).to_numpy(dtype=float)
    elif isinstance(hedge_ratio, pd.Series):
        if not hedge_ratio.index.is_unique or not hedge_ratio.index.is_monotonic_increasing:
            raise ValueError("hedge_ratio index must be unique and chronological")
        beta_values = hedge_ratio.reindex(aligned_strategy.index).to_numpy(dtype=float)
    else:
        try:
            beta = float(hedge_ratio)
        except (TypeError, ValueError) as exc:
            raise ValueError("hedge_ratio must be finite") from exc
        if not math.isfinite(beta):
            raise ValueError("hedge_ratio must be finite")
        beta_values = np.full(len(aligned_strategy), beta, dtype=float)
    if np.isinf(beta_values).any():
        raise ValueError("hedge_ratio values must be finite wherever present")
    position = 0
    entry_index: Any = None
    trade_pnl = 0.0
    period_pnl = np.zeros(len(aligned_strategy), dtype=float)
    position_path = np.zeros(len(aligned_strategy), dtype=int)
    signal_path = np.full(len(aligned_strategy), "flat", dtype=object)
    trade_records: list[dict[str, Any]] = []
    wins = 0
    completed_trades = 0

    for t in range(len(aligned_strategy)):
        # The position established at t-1 earns the price move ending at t.
        if t > 0 and position != 0:
            held_beta = beta_values[t - 1]
            if not math.isfinite(held_beta):
                raise ValueError("a finite prior hedge_ratio is required while a trade is open")
            increment = float(
                position * ((a[t] - a[t - 1]) - held_beta * (b[t] - b[t - 1]))
            )
            period_pnl[t] = increment
            trade_pnl += increment

        score = z[t]
        closed_this_bar = False
        if position == 1:
            if math.isfinite(score) and score <= -stop:
                signal_path[t] = "exit_stop_loss"
            elif math.isfinite(score) and score >= -exit_level:
                signal_path[t] = "exit_convergence"
            else:
                signal_path[t] = "hold_long_spread"
        elif position == -1:
            if math.isfinite(score) and score >= stop:
                signal_path[t] = "exit_stop_loss"
            elif math.isfinite(score) and score <= exit_level:
                signal_path[t] = "exit_convergence"
            else:
                signal_path[t] = "hold_short_spread"

        if position != 0 and signal_path[t] in {"exit_stop_loss", "exit_convergence"}:
            completed_trades += 1
            wins += int(trade_pnl > 0.0)
            trade_records.append({
                "entry_index": entry_index,
                "exit_index": aligned_strategy.index[t],
                "direction": "long_spread" if position == 1 else "short_spread",
                "pnl": float(trade_pnl),
                "status": "closed",
                "exit_reason": str(signal_path[t]),
            })
            position = 0
            entry_index = None
            trade_pnl = 0.0
            closed_this_bar = True

        if (
            position == 0
            and not closed_this_bar
            and math.isfinite(score)
            and math.isfinite(beta_values[t])
        ):
            if score <= -entry:
                position = 1
                entry_index = aligned_strategy.index[t]
                trade_pnl = 0.0
                signal_path[t] = "enter_long_spread"
            elif score >= entry:
                position = -1
                entry_index = aligned_strategy.index[t]
                trade_pnl = 0.0
                signal_path[t] = "enter_short_spread"
        position_path[t] = position

    if position != 0:
        trade_records.append({
            "entry_index": entry_index,
            "exit_index": None,
            "direction": "long_spread" if position == 1 else "short_spread",
            "pnl": float(trade_pnl),
            "status": "open",
            "exit_reason": None,
        })
    index = aligned_strategy.index
    pnl_path = pd.Series(period_pnl, index=index, name="pnl")
    cumulative_pnl = pnl_path.cumsum().rename("cumulative_pnl")
    return {
        "positions": pd.Series(position_path, index=index, name="spread_position"),
        "signals": pd.Series(signal_path, index=index, name="trade_signal"),
        "pnl_path": pnl_path,
        "cumulative_pnl": cumulative_pnl,
        "total_pnl": float(period_pnl.sum()),
        "trade_log": pd.DataFrame(trade_records),
        "trade_count": int(completed_trades + int(position != 0)),
        "completed_trade_count": completed_trades,
        "active_trade_count": int(position != 0),
        "win_rate": float(wins / completed_trades) if completed_trades else 0.0,
        "input_shape": [int(len(aligned_strategy)), 3],
        "latency_ms": float((perf_counter() - started) * 1000.0),
    }


def cointegration_telemetry(
    price_a: pd.Series,
    price_b: pd.Series,
) -> dict[str, Any]:
    r"""Expose Engle-Granger latency, p-value, sample shape, and status flag.

    A low p-value supports stationary residuals from
    :math:`P_A=\alpha+\beta P_B+\varepsilon`; for instance, p-value below
    0.05 marks a statistically linked pair for further strategy analysis.
    This helper runs the same diagnostic and returns its glass-box metrics.
    Complexity: :math:`O(T\log T)` time and :math:`O(T)` space.
    """
    result = test_engle_granger_cointegration(price_a, price_b)
    return {
        "latency_ms": result["latency_ms"],
        "input_shape": result["input_shape"],
        "cointegration_p_value": result["p_value"],
        "is_cointegrated": result["is_cointegrated"],
        "hedge_ratio": result["hedge_ratio"],
        "n_observations": result["n_observations"],
    }
