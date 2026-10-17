"""Medium-horizon advisory orchestration over pure quantitative modules.

This application service accepts an already validated point-in-time snapshot,
calls the pure math core, and assembles an explainable recommendation. It has
no database, HTTP, filesystem, or network responsibilities.
""" 

from __future__ import annotations

from datetime import date, datetime, timezone
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from app.core.portfolio.optimizer import optimize_maximum_sharpe, optimize_minimum_variance
from app.core.portfolio.risk_parity import optimize_risk_parity
from app.core.profiling.stochastic_processes import calculate_stationary_distribution, test_martingale_property
from app.core.trading.execution_costs import calculate_total_execution_drag
from app.core.trading.l_var import calculate_liquidity_adjusted_var
from app.core.trading.position_sizing import calculate_multivariate_kelly
from app.core.trading.rebalance import calculate_portfolio_turnover, calculate_rebalancing_drag


REGIME_LABELS = ("low_volatility", "normal_volatility", "high_volatility")
MIN_NET_SHARPE_IMPROVEMENT = 0.15


def calculate_tolerance_bands(
    current_weights: dict[str, float],
    target_weights: dict[str, float],
    annualized_volatility: dict[str, float],
) -> list[dict[str, Any]]:
    r"""Classify portfolio-weight drift using volatility-adjusted bands.

    For asset :math:`i`, band half-width is
    :math:`b_i=\min(0.20w_i^*,2\sigma_i)` and its interval is
    :math:`[\max(0,w_i^*-b_i),w_i^*+b_i]`. A holding above the upper bound
    is ``DRIFT_UP``; below the lower bound it is ``DRIFT_DOWN``; otherwise it
    is ``BALANCED``. For a 30% target and 20% annual volatility, the band is
    6 percentage points wide, so 34% is balanced and 38% is overweight.
    Complexity is :math:`O(A)` time and space for :math:`A` assets.
    """
    if set(current_weights) != set(target_weights) or set(target_weights) != set(annualized_volatility):
        raise ValueError("current weights, target weights, and volatility keys must match")
    rows: list[dict[str, Any]] = []
    for ticker, target in target_weights.items():
        current = float(current_weights[ticker])
        target = float(target)
        volatility = float(annualized_volatility[ticker])
        if not np.isfinite([current, target, volatility]).all() or current < 0 or target < 0 or volatility < 0:
            raise ValueError("weights and annualized volatility must be finite and non-negative")
        width = min(0.20 * target, 2.0 * volatility)
        lower, upper = max(0.0, target - width), target + width
        status = "DRIFT_UP" if current > upper + 1e-12 else "DRIFT_DOWN" if current < lower - 1e-12 else "BALANCED"
        rows.append({
            "ticker": ticker, "current_weight": current, "target_weight": target,
            "annualized_volatility": volatility, "band_width": width,
            "lower_bound": lower, "upper_bound": upper, "drift": current - target,
            "status": status,
        })
    return rows


def evaluate_trade_signal(
    tolerance_bands: list[dict[str, Any]],
    net_sharpe_improvement: float,
    holding_periods_satisfied: bool,
    *,
    minimum_sharpe_improvement: float = MIN_NET_SHARPE_IMPROVEMENT,
) -> dict[str, Any]:
    r"""Gate discretionary trade signals on drift, net Sharpe, and holding age.

    A formal signal is emitted iff at least one asset is outside its band,
    net portfolio Sharpe improvement after modeled execution costs is at least
    :math:`0.15`, and every affected position satisfies its minimum holding
    period. The three checks are conjunctive: a 4-point Sharpe gain cannot
    override a holding lock, and an overweight inside its band creates no
    trade. Complexity is :math:`O(A)` time and space.
    """
    if not np.isfinite([net_sharpe_improvement, minimum_sharpe_improvement]).all():
        raise ValueError("Sharpe thresholds must be finite")
    drifted = [row for row in tolerance_bands if row["status"] != "BALANCED"]
    drift_gate = bool(drifted)
    sharpe_gate = float(net_sharpe_improvement) > float(minimum_sharpe_improvement)
    signal = drift_gate and sharpe_gate and bool(holding_periods_satisfied)
    return {
        "triggered": signal,
        "net_sharpe_improvement": float(net_sharpe_improvement),
        "minimum_net_sharpe_improvement": float(minimum_sharpe_improvement),
        "drift_boundary_exceeded": drift_gate,
        "sharpe_threshold_met": sharpe_gate,
        "holding_periods_satisfied": bool(holding_periods_satisfied),
        "affected_assets": [
            {"ticker": row["ticker"], "action": "SELL" if row["status"] == "DRIFT_UP" else "BUY",
             "drift_status": row["status"]}
            for row in drifted
        ] if signal else [],
        "reason": "all_signal_conditions_met" if signal else (
            "no_asset_outside_tolerance_band" if not drift_gate else
            "net_sharpe_improvement_below_threshold" if not sharpe_gate else
            "minimum_holding_period_not_satisfied"
        ),
    }


def allocate_smart_dca(
    monthly_amount: float,
    target_weights: dict[str, float],
    current_values: dict[str, float],
    portfolio_value: float,
    tolerance_bands: list[dict[str, Any]],
) -> dict[str, float]:
    r"""Allocate monthly cash to underweight/balanced assets without sales.

    The notional deficit is :math:`d_i=\max(0,w_i^*(V+D)-V_i)`. New cash is
    distributed across ``DRIFT_DOWN`` assets first, then balanced assets;
    ``DRIFT_UP`` assets receive zero while another eligible asset exists.
    If every asset is overweight, the unallocated amount stays in cash. For
    example, with €500 DCA and A overweight/B underweight, all €500 goes to
    B instead of selling A. Complexity: :math:`O(A)` time and space.
    """
    if not np.isfinite([monthly_amount, portfolio_value]).all() or monthly_amount < 0 or portfolio_value < 0:
        raise ValueError("monthly amount and portfolio value must be finite and non-negative")
    if set(target_weights) != set(current_values):
        raise ValueError("target weights and current values must contain identical assets")
    for ticker, weight in target_weights.items():
        if not np.isfinite([weight, current_values[ticker]]).all() or weight < 0 or current_values[ticker] < 0:
            raise ValueError("weights and current values must be finite and non-negative")
    status = {row["ticker"]: row["status"] for row in tolerance_bands}
    if set(status) != set(target_weights):
        raise ValueError("tolerance bands must cover every target asset")
    underweight = {ticker for ticker in status if status[ticker] == "DRIFT_DOWN"}
    balanced = {ticker for ticker in status if status[ticker] == "BALANCED"}
    eligible = underweight or balanced
    deficits = {
        ticker: max(0.0, float(target_weights[ticker]) * (portfolio_value + monthly_amount) - float(current_values[ticker]))
        if ticker in eligible else 0.0
        for ticker in target_weights
    }
    deficit_total = sum(deficits.values())
    if monthly_amount == 0 or not eligible:
        return {ticker: 0.0 for ticker in target_weights}
    if deficit_total > 0:
        allocation = {ticker: monthly_amount * deficit / deficit_total for ticker, deficit in deficits.items()}
    else:
        eligible_weight = sum(float(target_weights[ticker]) for ticker in eligible)
        if eligible_weight <= 0:
            return {ticker: 0.0 for ticker in target_weights}
        allocation = {
            ticker: monthly_amount * float(target_weights[ticker]) / eligible_weight if ticker in eligible else 0.0
            for ticker in target_weights
        }
    return allocation


def _periods_per_year(observation_dates: list[date]) -> int:
    """Infer trading observations/year from median date spacing."""
    if len(observation_dates) < 2:
        raise ValueError("at least two dated return observations are required")
    gaps = np.asarray(
        [(later - earlier).days for earlier, later in zip(observation_dates, observation_dates[1:])],
        dtype=float,
    )
    median_gap = float(np.median(gaps))
    if median_gap <= 4.0:
        return 252
    return max(1, int(round(365.25 / median_gap)))


def _frequency_label(periods_per_year: int) -> str:
    """Map inferred periods/year to a display-friendly cadence."""
    if periods_per_year >= 200:
        return "daily"
    if periods_per_year >= 40:
        return "weekly"
    if periods_per_year >= 9:
        return "monthly"
    return "irregular"


def _one_way_turnover(current: np.ndarray, target: np.ndarray) -> float:
    """Return the conventional one-way traded fraction, ``0.5 * L1``."""
    return float(0.5 * calculate_portfolio_turnover(current, target))


def _limit_one_way_turnover(
    current: np.ndarray, target: np.ndarray, maximum: float
) -> tuple[np.ndarray, float]:
    """Scale a target move proportionally to a one-way turnover cap."""
    one_way = _one_way_turnover(current, target)
    if one_way <= maximum or one_way == 0.0:
        return target, one_way
    constrained = current + (target - current) * (maximum / one_way)
    constrained = np.maximum(constrained, 0.0)
    constrained /= constrained.sum()
    return constrained, _one_way_turnover(current, constrained)


def _estimate_regime_transition(
    portfolio_returns: np.ndarray, lookback_months: int, periods_per_year: int
) -> dict[str, Any]:
    """Estimate smoothed volatility-regime transitions over 3–12 months."""
    periods_per_month = max(2, int(round(periods_per_year / 12.0)))
    min_periods = max(2, int(np.ceil(periods_per_month * 0.7)))
    volatility = pd.Series(portfolio_returns).rolling(
        window=periods_per_month, min_periods=min_periods
    ).std(ddof=1).dropna()
    window = min(len(volatility), lookback_months * periods_per_month)
    selected = volatility.iloc[-window:].to_numpy(dtype=float)
    if selected.size < 3:
        raise ValueError("at least three realized-volatility observations are required for regime analysis")
    thresholds = np.quantile(selected, [1.0 / 3.0, 2.0 / 3.0])
    states = np.digitize(selected, thresholds, right=True)
    counts = np.full((3, 3), 0.5, dtype=float)  # symmetric Dirichlet smoothing
    for previous, following in zip(states, states[1:]):
        counts[previous, following] += 1.0
    transition = counts / counts.sum(axis=1, keepdims=True)
    stationary = calculate_stationary_distribution(transition)
    current_state = int(states[-1])
    next_probabilities = transition[current_state]
    return {
        "labels": list(REGIME_LABELS),
        "current_state": REGIME_LABELS[current_state],
        "probability_next_period": {
            label: float(next_probabilities[i]) for i, label in enumerate(REGIME_LABELS)
        },
        "stationary_distribution": {
            label: float(stationary[i]) for i, label in enumerate(REGIME_LABELS)
        },
        "transition_matrix": transition.tolist(),
        "lookback_months": int(lookback_months),
        "observations": int(selected.size),
        "high_volatility_transition_probability": float(next_probabilities[2]),
    }


def _holding_period_returns(
    period_returns: np.ndarray, horizon_periods: int, frequency: str
) -> tuple[np.ndarray, str]:
    """Compound overlapping returns at their observed frequency or label fallback."""
    # Require at least two overlapping horizon observations; a single
    # compounded sample cannot support a meaningful tail quantile.
    if period_returns.size > horizon_periods:
        windows = np.lib.stride_tricks.sliding_window_view(period_returns, horizon_periods)
        return np.prod(1.0 + windows, axis=1) - 1.0, f"overlapping_compounded_{frequency}_observations"
    return period_returns * np.sqrt(horizon_periods), "sqrt_time_scaled_short_history_fallback"


def calculate_portfolio_projections(
    initial_capital: float,
    monthly_dca_amount: float,
    annualized_return: float,
    annualized_volatility: float,
    target_horizon_years: int = 5,
    *,
    n_simulations: int = 10_000,
    seed: int = 42,
) -> list[dict[str, float | int]]:
    r"""Project wealth with monthly log-normal returns and end-month deposits.

    For each month, :math:`R_m = \exp((\ln(1+r_p)-\frac12\sigma_p^2)/12 +
    \sigma_p Z/\sqrt{12}) - 1`, then :math:`V_m=V_{m-1}(1+R_m)+D`.
    The reported 10th/50th/90th percentiles are Monte Carlo quantiles, and
    estimated net gains subtract initial capital plus every contribution.
    For example, a €10,000 account adding €200/month for 5 years deposits
    €22,000 total before returns. This is a scenario distribution, not a
    guaranteed return forecast. Time and memory complexity are both
    :math:`O(S \cdot 12Y)` for ``S`` simulations and ``Y`` years.
    """
    values = np.asarray(
        [initial_capital, monthly_dca_amount, annualized_return, annualized_volatility], dtype=float
    )
    if not np.isfinite(values).all() or initial_capital < 0 or monthly_dca_amount < 0:
        raise ValueError("capital and DCA must be finite and non-negative; moments must be finite")
    if annualized_volatility < 0 or annualized_return <= -1:
        raise ValueError("annualized return must exceed -100% and volatility must be non-negative")
    if isinstance(target_horizon_years, bool) or not isinstance(target_horizon_years, (int, np.integer)) or not 1 <= target_horizon_years <= 20:
        raise ValueError("target_horizon_years must be between 1 and 20")
    if isinstance(n_simulations, bool) or not isinstance(n_simulations, (int, np.integer)) or n_simulations < 100:
        raise ValueError("n_simulations must be at least 100")

    if annualized_volatility > np.sqrt(np.finfo(float).max):
        raise ValueError("projection inputs exceed the supported numerical range")
    months = target_horizon_years * 12
    monthly_log_drift = (np.log1p(annualized_return) - 0.5 * annualized_volatility**2) / 12.0
    monthly_sigma = annualized_volatility / np.sqrt(12.0)
    rng = np.random.default_rng(seed)
    wealth = np.full(n_simulations, float(initial_capital), dtype=float)
    projections: list[dict[str, float | int]] = []
    for month in range(1, months + 1):
        shocks = rng.standard_normal(n_simulations)
        with np.errstate(over="raise", invalid="raise"):
            try:
                growth = np.exp(monthly_log_drift + monthly_sigma * shocks)
                wealth = wealth * growth + monthly_dca_amount
            except FloatingPointError as exc:
                raise ValueError("projection inputs exceed the supported numerical range") from exc
        if month % 12 == 0:
            year = month // 12
            deposited = float(initial_capital + monthly_dca_amount * month)
            p10, p50, p90 = np.quantile(wealth, [0.10, 0.50, 0.90])
            projections.append({
                "year": year,
                "total_capital_deposited": deposited,
                "pessimistic_value": float(p10),
                "median_value": float(p50),
                "optimistic_value": float(p90),
                "pessimistic_net_gain": float(p10 - deposited),
                "median_net_gain": float(p50 - deposited),
                "optimistic_net_gain": float(p90 - deposited),
            })
    return projections


def _portfolio_tracking_outputs(
    *, tickers: list[str], prices: list[float], price_currencies: list[str] | None,
    holdings: list[dict[str, Any]] | None, current_weights: dict[str, float],
    target_weights: dict[str, float], capital: float, monthly_dca_amount: float,
    target_horizon_years: int, expected_returns: pd.Series, covariance: pd.DataFrame,
    risk_free_rate: float, tolerance_bands: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, float]]:
    """Build position valuation, buy-only DCA orders, and gain scenarios."""
    price_map = dict(zip(tickers, map(float, prices)))
    currency_map = dict(zip(tickers, [item.upper() for item in (price_currencies or ["EUR"] * len(tickers))]))
    holding_map = {str(item["ticker"]).upper(): item for item in (holdings or [])}
    unknown = set(holding_map) - set(tickers)
    if unknown:
        raise ValueError(f"holdings contain assets outside the saved portfolio: {sorted(unknown)}")
    currencies = {str(item["currency"]).upper() for item in holding_map.values()}
    currencies.update(currency_map[ticker] for ticker in holding_map if ticker in currency_map)
    if monthly_dca_amount > 0:
        quoted_currencies = set(currency_map.values())
        if len(quoted_currencies) > 1:
            raise ValueError("monthly DCA across multiple quote currencies requires FX conversion, which is not configured")
        currencies.update(quoted_currencies)
    if len(currencies) > 1:
        raise ValueError("portfolio valuation requires one currency; FX conversion is not configured")
    currency = next(iter(currencies), next(iter(currency_map.values()), "EUR"))

    values_by_ticker: dict[str, float] = {}
    cost_by_ticker: dict[str, float] = {}
    rows: list[dict[str, Any]] = []
    for ticker in tickers:
        item = holding_map.get(ticker)
        if item is None:
            continue
        item_currency = str(item["currency"]).upper()
        if currency_map[ticker] != item_currency:
            raise ValueError(f"holding currency for {ticker} does not match its current price currency")
        quantity = float(item["quantity"])
        current_price = price_map[ticker]
        pru = float(item["average_buy_price"])
        market_value = quantity * current_price
        cost_basis = quantity * pru
        values_by_ticker[ticker] = market_value
        cost_by_ticker[ticker] = cost_basis
        rows.append({
            "ticker": ticker, "quantity": quantity, "average_buy_price": pru,
            "current_price": current_price, "market_value": market_value,
            "cost_basis": cost_basis, "unrealized_pnl": market_value - cost_basis,
            "unrealized_pnl_fraction": (current_price / pru - 1.0) if pru else 0.0,
            "currency": item_currency,
        })

    total_market_value = float(sum(values_by_ticker.values()))
    total_cost_basis = float(sum(cost_by_ticker.values()))
    valuation_available = bool(rows)
    cash_only_start = not valuation_available and np.allclose(
        list(current_weights.values()), 0.0, atol=1e-12
    )
    if valuation_available and total_market_value <= 0:
        raise ValueError("portfolio holdings must have positive total market value")
    if valuation_available:
        actual_weights = {ticker: values_by_ticker.get(ticker, 0.0) / total_market_value for ticker in tickers}
        pnl = total_market_value - total_cost_basis
        portfolio_weights = np.asarray([target_weights.get(ticker, 0.0) for ticker in tickers], dtype=float)
        risky_expected_return = float(portfolio_weights @ expected_returns.reindex(tickers).to_numpy(dtype=float))
        risky_sigma = float(np.sqrt(max(0.0, portfolio_weights @ covariance.loc[tickers, tickers].to_numpy(dtype=float) @ portfolio_weights)))
        projection_capital = total_market_value
    else:
        actual_weights = {ticker: float(current_weights.get(ticker, 0.0)) for ticker in tickers}
        pnl = 0.0
        total_market_value = 0.0
        total_cost_basis = 0.0
        projection_capital = 0.0 if cash_only_start else float(capital)
        portfolio_weights = np.asarray([target_weights.get(ticker, 0.0) for ticker in tickers], dtype=float)
        risky_expected_return = float(portfolio_weights @ expected_returns.reindex(tickers).to_numpy(dtype=float))
        risky_sigma = float(np.sqrt(max(0.0, portfolio_weights @ covariance.loc[tickers, tickers].to_numpy(dtype=float) @ portfolio_weights)))

    valuation = {
        "available": valuation_available,
        "currency": currency,
        "total_market_value": total_market_value if valuation_available else None,
        "total_cost_basis": total_cost_basis if valuation_available else None,
        "unrealized_pnl": pnl if valuation_available else None,
        "unrealized_pnl_fraction": pnl / total_cost_basis if valuation_available and total_cost_basis else None,
        "holdings": rows,
        "valuation_basis": "quantity × latest aligned close; unrealized PnL excludes fees and taxes",
    }

    if valuation_available:
        current_values = {ticker: values_by_ticker.get(ticker, 0.0) for ticker in tickers}
        dca_basis = total_market_value
    else:
        current_values = {ticker: float(current_weights.get(ticker, 0.0)) * capital for ticker in tickers}
        dca_basis = 0.0 if cash_only_start else float(capital)
    if cash_only_start:
        allocation = {
            ticker: float(monthly_dca_amount) * float(target_weights.get(ticker, 0.0))
            for ticker in tickers
        }
    else:
        allocation = allocate_smart_dca(
            float(monthly_dca_amount),
            {ticker: float(target_weights.get(ticker, 0.0)) for ticker in tickers},
            current_values,
            dca_basis,
            tolerance_bands,
        )
    dca = {
        "monthly_amount": float(monthly_dca_amount), "currency": currency,
        "orders": [{
            "ticker": ticker, "side": "buy", "amount": float(allocation[ticker]),
            "share_of_monthly_dca": float(allocation[ticker] / monthly_dca_amount) if monthly_dca_amount else 0.0,
            "estimated_quantity": float(allocation[ticker] / price_map[ticker]) if price_map[ticker] > 0 else 0.0,
        } for ticker in tickers if allocation[ticker] > 0],
        "buy_only": True,
        "unallocated_amount": max(0.0, float(monthly_dca_amount) - sum(float(value) for value in allocation.values())),
        "basis": (
            "starting from cash, the monthly contribution follows suggested target weights and the remainder stays in cash"
            if cash_only_start else
            "underweight assets receive new cash first; balanced assets may receive residual cash; overweight assets are excluded; no asset is sold"
        ),
        "tolerance_bands": tolerance_bands,
    }

    risky_weight = float(portfolio_weights.sum())
    annual_return = max(-0.999999, risky_expected_return + max(0.0, 1.0 - risky_weight) * risk_free_rate)
    projections = calculate_portfolio_projections(
        projection_capital, monthly_dca_amount, annual_return, risky_sigma,
        target_horizon_years,
    )
    projection_summary = {
        "annualized_expected_return": annual_return,
        "annualized_volatility": risky_sigma,
        "initial_capital": projection_capital,
        "monthly_dca_amount": float(monthly_dca_amount),
        "target_horizon_years": target_horizon_years,
        "scenario_labels": {"p10": "pessimistic", "p50": "median", "p90": "optimistic"},
        "method": "10,000-path monthly log-normal Monte Carlo; contributions enter at month end",
        "currency": currency,
        "years": projections,
    }
    return valuation, dca, projection_summary, actual_weights


def build_advisory_recommendation(
    *,
    portfolio_name: str,
    tickers: list[str],
    current_weights: dict[str, float],
    market: Any,
    policy: Any,
    last_decision_at: datetime | None = None,
    holdings: list[dict[str, Any]] | None = None,
    monthly_dca_amount: float = 0.0,
    target_horizon_years: int = 5,
) -> dict[str, Any]:
    r"""Create a medium/long-term allocation, risk, and trading-cost preview.

    The service infers return cadence from observation dates, annualizes
    arithmetic simple-return sample moments, and optimizes a long-only
    strategic allocation, applies a documented turnover-damping overlay and
    cap, and estimates fractional-Kelly exposure, transaction friction,
    holding-period VaR/CVaR, liquidity-adjusted VaR, and Markov volatility
    regime transitions. Weekly provider observations are annualized at 52
    periods/year rather than being treated as daily. Calendar holding days are
    converted to matching return periods. A formal signal also requires an
    entered acquisition date for every affected tracked holding.
    A tracked $1,000,000 account with 20% Kelly sizing and a 10% one-way
    turnover cap gets a restrained, reviewable target rather than a
    day-trading signal. A saved portfolio with no tracked holdings starts
    fully in cash; only its configured monthly contribution is used for
    contribution sizing and the projection starts with zero capital.
    Complexity is approximately :math:`O(TA^2 + A^3)` for ``T`` return
    observations and ``A`` assets, plus the optimizer's iterative work.
    """
    started = perf_counter()
    assets = list(tickers)
    if assets != list(market.tickers):
        raise ValueError("portfolio tickers and market snapshot tickers must have identical ordering")
    if set(current_weights) != set(assets):
        raise ValueError("current_weights must match the portfolio tickers exactly")

    raw_returns = np.asarray(market.returns, dtype=float)
    periods_per_year = _periods_per_year(market.observation_dates)
    frequency = _frequency_label(periods_per_year)
    returns_frame = pd.DataFrame(raw_returns, columns=assets)
    # Markowitz/Kelly use arithmetic moments; a weighted sum of asset log
    # returns is not an exact portfolio return. Growth is estimated separately.
    expected_returns = returns_frame.mean(axis=0) * periods_per_year
    annual_covariance = returns_frame.cov(ddof=1) * periods_per_year

    if policy.allocation_method == "risk_parity":
        allocation = optimize_risk_parity(annual_covariance)
    elif policy.allocation_method == "max_sharpe":
        allocation = optimize_maximum_sharpe(expected_returns, annual_covariance, policy.risk_free_rate)
    else:
        allocation = optimize_minimum_variance(annual_covariance)
    strategic_weights = np.asarray([allocation["weights"][ticker] for ticker in assets], dtype=float)

    quoted_currencies = getattr(market, "price_currencies", None)
    if quoted_currencies is not None and len({code.upper() for code in quoted_currencies}) > 1:
        currency_by_ticker = dict(zip(assets, (code.upper() for code in quoted_currencies)))
        quote_summary = ", ".join(f"{ticker}={currency_by_ticker[ticker]}" for ticker in assets)
        raise ValueError(
            "portfolio valuation requires one currency; quote currencies in the submitted market data are "
            f"{quote_summary}. FX conversion is not configured"
        )
    capital = float(policy.capital)
    holding_map = {str(item["ticker"]).upper(): item for item in (holdings or [])}
    unknown_holdings = set(holding_map) - set(assets)
    if unknown_holdings:
        raise ValueError(f"holdings contain assets outside the portfolio: {sorted(unknown_holdings)}")
    if holding_map:
        price_currencies = dict(zip(assets, [code.upper() for code in (getattr(market, "price_currencies", None) or ["EUR"] * len(assets))]))
        holding_currencies = {str(item["currency"]).upper() for item in holding_map.values()}
        holding_currencies.update(price_currencies[ticker] for ticker in holding_map)
        if len(holding_currencies) > 1:
            currency_summary = ", ".join(
                f"{ticker}=holding:{str(holding_map[ticker]['currency']).upper()}/quote:{price_currencies[ticker]}"
                for ticker in holding_map
            )
            raise ValueError(
                "portfolio valuation requires one currency; holding and quote currencies are "
                f"{currency_summary}. FX conversion is not configured"
            )
        if any(price_currencies[ticker] != str(item["currency"]).upper() for ticker, item in holding_map.items()):
            raise ValueError("holding currency must match the latest price currency")
        prices_by_ticker = dict(zip(assets, map(float, market.prices)))
        holding_values = {ticker: float(item["quantity"]) * prices_by_ticker[ticker] for ticker, item in holding_map.items()}
        capital = float(sum(holding_values.values()))
        if capital <= 0:
            raise ValueError("tracked holdings must have positive total market value")
        current_asset_weights = np.asarray([holding_values.get(ticker, 0.0) / capital for ticker in assets], dtype=float)
    else:
        cash_only_start = np.allclose(list(current_weights.values()), 0.0, atol=1e-12)
        current_asset_weights = (
            np.zeros(len(assets), dtype=float)
            if cash_only_start
            else np.asarray([current_weights[ticker] for ticker in assets], dtype=float)
        )
        if cash_only_start:
            capital = float(monthly_dca_amount)
    current_all_weights = np.append(current_asset_weights, max(0.0, 1.0 - current_asset_weights.sum()))
    # The portfolio model requires a fully invested allocation. A cash residual
    # is explicit in recommendations when the conservative Kelly overlay sizes
    # the risky sleeve below 100%.
    if current_all_weights.sum() <= 0 or not np.isclose(current_all_weights.sum(), 1.0, atol=1e-6):
        raise ValueError("current portfolio weights must sum to 1")

    kelly_weights = calculate_multivariate_kelly(
        expected_returns.to_numpy(dtype=float),
        annual_covariance.to_numpy(dtype=float),
        risk_free_rate=policy.risk_free_rate,
        fraction=policy.fractional_kelly,
        max_leverage=1.0,
    )
    kelly_risky_fraction = float(np.clip(np.maximum(kelly_weights, 0.0).sum(), 0.0, 1.0))
    kelly_long_only_composition = np.maximum(kelly_weights, 0.0)
    if kelly_long_only_composition.sum() > 0:
        kelly_long_only_composition /= kelly_long_only_composition.sum()
    else:
        kelly_long_only_composition = strategic_weights.copy()
    kelly_sleeve = (
        kelly_long_only_composition * kelly_risky_fraction
        if policy.kelly_sizing_enabled
        else strategic_weights.copy()
    )
    cash_weight = 1.0 - float(kelly_sleeve.sum())
    proposed = np.append(kelly_sleeve, cash_weight)

    # This convex movement overlay is intentionally explicit: a factor of 3
    # advances one quarter of the full target change per recommendation.
    movement_fraction = 1.0 / (1.0 + float(policy.turnover_penalty_factor))
    damped = current_all_weights + movement_fraction * (proposed - current_all_weights)
    max_one_way_turnover = min(float(policy.max_one_way_turnover), 0.5)
    target_all_weights, one_way_turnover = _limit_one_way_turnover(
        current_all_weights, damped, max_one_way_turnover
    )

    holding_interval_satisfied = True
    elapsed_since_last = None
    if last_decision_at is not None:
        if last_decision_at.tzinfo is None:
            last_decision_at = last_decision_at.replace(tzinfo=timezone.utc)
        elapsed_since_last = max(0, (market.as_of - last_decision_at).days)
        holding_interval_satisfied = elapsed_since_last >= policy.minimum_holding_period_days
        if not holding_interval_satisfied:
            target_all_weights = current_all_weights.copy()
            one_way_turnover = 0.0

    target_map = {ticker: float(target_all_weights[i]) for i, ticker in enumerate(assets)}
    target_map["CASH"] = float(target_all_weights[-1])
    current_map = {ticker: float(current_all_weights[i]) for i, ticker in enumerate(assets)}
    current_map["CASH"] = float(current_all_weights[-1])
    asset_volatility = {
        ticker: float(np.sqrt(max(0.0, annual_covariance.loc[ticker, ticker])))
        for ticker in assets
    }
    tolerance_bands = calculate_tolerance_bands(
        {ticker: current_map[ticker] for ticker in assets},
        {ticker: target_map[ticker] for ticker in assets},
        asset_volatility,
    )
    weight_deltas = {ticker: target_map[ticker] - current_map[ticker] for ticker in target_map}
    gross_turnover = float(sum(abs(value) for value in weight_deltas.values()))
    prices = np.asarray(market.prices, dtype=float)
    adv_values = np.asarray(market.adv_values, dtype=float)
    spreads = np.asarray(market.spreads, dtype=float)
    period_asset_volatility = returns_frame.std(axis=0, ddof=1).to_numpy(dtype=float)
    daily_asset_volatility = period_asset_volatility * np.sqrt(periods_per_year / 252.0)
    trade_notionals = np.asarray([weight_deltas[ticker] * capital for ticker in assets], dtype=float)
    execution_costs = calculate_total_execution_drag(
        trade_notionals,
        np.ones(len(assets), dtype=float),
        adv_values,
        daily_asset_volatility,
        spreads,
    )
    estimated_annualized_turnover_drag = calculate_rebalancing_drag(
        pd.Series([gross_turnover], dtype=float),
        float(execution_costs.get("total_friction_bps", 0.0)),
        periods_per_year=max(1, int(round(365.25 / policy.minimum_holding_period_days))),
    )

    cash_period_return = np.expm1(np.log1p(float(policy.risk_free_rate)) / periods_per_year)
    current_returns = raw_returns @ current_all_weights[:-1] + current_all_weights[-1] * cash_period_return
    proposed_returns = raw_returns @ target_all_weights[:-1] + target_all_weights[-1] * cash_period_return

    def annualized_sharpe(series: np.ndarray) -> float:
        annual_mean = float(np.mean(series) * periods_per_year)
        annual_volatility = float(np.std(series, ddof=1) * np.sqrt(periods_per_year))
        if not np.isfinite(annual_volatility) or annual_volatility <= 1e-12:
            return 0.0
        return (annual_mean - float(policy.risk_free_rate)) / annual_volatility

    execution_drag_annualized = (
        float(execution_costs["total_execution_drag"]) / max(capital, 1e-12)
        * 365.25 / max(int(policy.holding_period_days), 1)
    )
    net_sharpe_improvement = annualized_sharpe(proposed_returns) - execution_drag_annualized / max(
        float(np.std(proposed_returns, ddof=1) * np.sqrt(periods_per_year)), 1e-12
    ) - annualized_sharpe(current_returns)
    age_by_ticker: dict[str, int | None] = {}
    holding_age_ok = True
    for row in tolerance_bands:
        if row["status"] == "BALANCED":
            continue
        lot = holding_map.get(row["ticker"])
        acquired = lot.get("acquired_at") if lot else None
        if acquired is None:
            age_by_ticker[row["ticker"]] = None
            holding_age_ok = False
            continue
        try:
            timestamp = pd.Timestamp(acquired)
            if timestamp.tzinfo is None:
                timestamp = timestamp.tz_localize("UTC")
            age_days = max(0, int((pd.Timestamp(market.as_of) - timestamp).days))
        except (TypeError, ValueError):
            age_days = None
        age_by_ticker[row["ticker"]] = age_days
        if age_days is None or age_days < int(policy.minimum_holding_period_days):
            holding_age_ok = False
    holding_periods_satisfied = holding_interval_satisfied and holding_age_ok
    trade_signal = evaluate_trade_signal(
        tolerance_bands,
        net_sharpe_improvement,
        holding_periods_satisfied,
    )
    trade_signal["holding_age_days_by_asset"] = age_by_ticker
    trade_signal["holding_age_basis"] = "user-entered acquisition date; missing dates block a formal signal"

    period_portfolio_returns = raw_returns @ target_all_weights[:-1]
    horizon_periods = max(1, int(round(policy.holding_period_days * periods_per_year / 365.25)))
    horizon_returns, holding_method = _holding_period_returns(
        period_portfolio_returns, horizon_periods, frequency
    )
    confidence = float(policy.confidence_level)
    return_quantile = float(np.quantile(horizon_returns, 1.0 - confidence))
    var_fraction = max(0.0, -return_quantile)
    tail_losses = -horizon_returns[horizon_returns <= return_quantile]
    cvar_fraction = max(var_fraction, float(np.mean(tail_losses)) if tail_losses.size else var_fraction)
    position_values = np.abs(target_all_weights[:-1]) * capital
    liquidity_var = calculate_liquidity_adjusted_var(
        pd.Series(period_portfolio_returns * np.sqrt(periods_per_year / 252.0)),
        position_values,
        adv_values,
        daily_asset_volatility,
        spreads,
        confidence_level=confidence,
        max_participation_rate=policy.max_adv_participation,
        method="historical",
    )
    portfolio_growth_returns = period_portfolio_returns + target_all_weights[-1] * cash_period_return
    annualized_portfolio_log_growth = float(np.log1p(portfolio_growth_returns).mean() * periods_per_year)
    expected_horizon_return = float(np.expm1(annualized_portfolio_log_growth * policy.holding_period_days / 365.25))

    regimes = _estimate_regime_transition(period_portfolio_returns, policy.regime_lookback_months, periods_per_year)
    current = pd.Series(100.0 * np.cumprod(1.0 + period_portfolio_returns))
    try:
        martingale_check = test_martingale_property(current, max_lags=min(5, max(1, len(current) - 2)))
    except (ValueError, TypeError):
        martingale_check = {"available": False, "reason": "insufficient return variation"}

    rationale: list[dict[str, Any]] = [
        {
            "code": "horizon_policy",
            "title": "Medium-term holding policy",
            "explanation": (
                f"This recommendation is calibrated for a {policy.holding_period_days}-day holding horizon "
                f"with a {policy.minimum_holding_period_days}-day minimum decision interval."
            ),
            "evidence": {
                "holding_period_days": policy.holding_period_days,
                "minimum_holding_period_days": policy.minimum_holding_period_days,
                "elapsed_since_last_decision_days": elapsed_since_last,
            },
        },
        {
            "code": "allocation_method",
            "title": "Structural asset allocation",
            "explanation": (
                "Structural asset risk parity rebalance based on annualized historical covariance."
                if policy.allocation_method == "risk_parity"
                else f"Long-only {policy.allocation_method.replace('_', ' ')} allocation from the same dated snapshot."
            ),
            "evidence": {"method": policy.allocation_method, "strategic_weights": dict(zip(assets, strategic_weights.tolist()))},
        },
        {
            "code": "turnover_damping",
            "title": "Turnover control",
            "explanation": (
                f"Only {movement_fraction:.1%} of the unconstrained target move is advanced per decision; "
                f"one-way turnover is then capped at {max_one_way_turnover:.1%}."
            ),
            "evidence": {
                "turnover_penalty_factor": policy.turnover_penalty_factor,
                "movement_fraction": movement_fraction,
                "one_way_turnover": one_way_turnover,
                "gross_turnover": gross_turnover,
                "max_one_way_turnover": max_one_way_turnover,
            },
        },
        {
            "code": "fractional_kelly",
            "title": "Conservative capital sizing",
            "explanation": (
                f"Fractional Kelly uses {policy.fractional_kelly:.0%} of the continuous estimate and caps gross "
                "risky exposure at 100%; any residual is shown explicitly as cash."
                if policy.kelly_sizing_enabled
                else "Kelly is reported as a diagnostic only; the selected allocation remains fully invested."
            ),
            "evidence": {
                "enabled": policy.kelly_sizing_enabled,
                "fraction": policy.fractional_kelly,
                "long_only_risky_fraction": kelly_risky_fraction,
                "raw_fractional_weights": dict(zip(assets, kelly_weights.tolist())),
            },
        },
        {
            "code": "realized_volatility_regime",
            "title": "Multi-month regime transition",
            "explanation": (
                f"The latest realized-volatility bucket is {regimes['current_state']}; "
                f"the estimated next-period high-volatility transition probability is "
                f"{regimes['high_volatility_transition_probability']:.1%}."
            ),
            "evidence": regimes,
        },
        {
            "code": "tolerance_band_signal_check",
            "title": "Tolerance bands and trade-signal checks",
            "explanation": (
                "A formal trade signal passed all drift, net Sharpe, and holding-age checks."
                if trade_signal["triggered"] else
                f"No formal trade signal was issued: {trade_signal['reason'].replace('_', ' ')}. "
                "The DCA plan remains buy-only and directs new contributions toward underweight assets."
            ),
            "evidence": trade_signal,
        },
    ]
    if not holding_interval_satisfied:
        rationale.append({
            "code": "minimum_holding_interval_not_met",
            "title": "Maintain current allocation",
            "explanation": (
                "The minimum holding interval has not elapsed since the last saved recommendation; "
                "the target is held at current weights for this preview."
            ),
            "evidence": {"elapsed_days": elapsed_since_last, "required_days": policy.minimum_holding_period_days},
        })
    if trade_signal["triggered"]:
        rationale.append({
            "code": "tolerance_band_trade_signal",
            "title": "Rebalance signal passed long-term safeguards",
            "explanation": (
                f"Drift crossed an asset tolerance band and modeled net Sharpe improvement is "
                f"{net_sharpe_improvement:+.2f}, above the +{MIN_NET_SHARPE_IMPROVEMENT:.2f} threshold. "
                "Every affected tracked position also meets the minimum holding period."
            ),
            "evidence": trade_signal,
        })
    if regimes["high_volatility_transition_probability"] >= 0.25:
        rationale.append({
            "code": "risk_regime_transition_alert",
            "title": "Risk regime transition alert",
            "explanation": "Macro regime shift detected: the multi-month Markov estimate assigns elevated probability to a high-volatility regime next period.",
            "evidence": {"probability": regimes["high_volatility_transition_probability"], "lookback_months": policy.regime_lookback_months},
        })

    valuation, dca_plan, projection, valuation_weights = _portfolio_tracking_outputs(
        tickers=assets,
        prices=list(map(float, market.prices)),
        price_currencies=getattr(market, "price_currencies", None),
        holdings=holdings,
        current_weights={ticker: float(current_asset_weights[i]) for i, ticker in enumerate(assets)},
        target_weights=target_map,
        capital=capital,
        monthly_dca_amount=float(monthly_dca_amount),
        target_horizon_years=int(target_horizon_years),
        expected_returns=expected_returns,
        covariance=annual_covariance,
        risk_free_rate=float(policy.risk_free_rate),
        tolerance_bands=tolerance_bands,
    )
    # Expose tracked market-value weights as the recommendation's actual starting
    # allocation whenever cost-basis holdings are available.
    if holdings:
        current_map = dict(valuation_weights)
        current_map["CASH"] = 0.0
        weight_deltas = {ticker: target_map[ticker] - current_map.get(ticker, 0.0) for ticker in target_map}
        gross_turnover = float(sum(abs(value) for value in weight_deltas.values()))
        one_way_turnover = 0.5 * gross_turnover

    result = {
        "model_version": "advisor-medium-horizon-1.3",
        "portfolio_name": portfolio_name,
        "tickers": assets,
        "current_weights": current_map,
        "target_weights": target_map,
        "weight_deltas": weight_deltas,
        "allocation_method": policy.allocation_method,
        "annualized_expected_returns": {ticker: float(expected_returns[ticker]) for ticker in assets},
        "annualized_covariance_matrix": annual_covariance.to_numpy(dtype=float).tolist(),
        "expected_holding_period_return_fraction": expected_horizon_return,
        "var_loss_fraction": var_fraction,
        "cvar_loss_fraction": cvar_fraction,
        "confidence_level": confidence,
        "holding_period_days": policy.holding_period_days,
        "observation_frequency": frequency,
        "periods_per_year": periods_per_year,
        "confidence_score": float(min(100.0, raw_returns.shape[0] / periods_per_year * 100.0)),
        "confidence_interpretation": "heuristic historical evidence coverage; not a calibrated probability of success",
        "holding_period_return_method": holding_method,
        "liquidity_adjusted_var_currency": float(liquidity_var["liquidity_adjusted_var"]),
        "liquidity_adjusted_var_loss_fraction": (
            float(liquidity_var["liquidity_adjusted_var"] / capital) if capital > 0 else 0.0
        ),
        "liquidation_horizon_days": int(liquidity_var["max_liquidation_days"]),
        "execution_cost": execution_costs,
        "gross_turnover": gross_turnover,
        "one_way_turnover": one_way_turnover,
        "estimated_annualized_turnover_drag_fraction": float(estimated_annualized_turnover_drag),
        "regime": regimes,
        "martingale_diagnostics": martingale_check,
        "valuation": valuation,
        "dca_plan": dca_plan,
        "tolerance_bands": tolerance_bands,
        "trade_signal": trade_signal,
        "projections": projection,
        "rationale": rationale,
        "snapshot_timestamp": market.as_of.isoformat(),
        "snapshot_start_date": market.observation_dates[0].isoformat(),
        "snapshot_end_date": market.observation_dates[-1].isoformat(),
        "latency_ms": float((perf_counter() - started) * 1000.0),
        "input_shape": {
            "returns": [int(raw_returns.shape[0]), int(raw_returns.shape[1])],
            "covariance_matrix": [len(assets), len(assets)],
            "regime_transition_matrix": [3, 3],
        },
        "model_conventions": {
            "returns": f"{frequency} simple returns; arithmetic mean and sample covariance annualized using {periods_per_year} periods/year; portfolio log growth for holding-period compounding",
            "var_cvar": "positive loss fractions at the stated confidence level",
            "liquidity_adjusted_var": "positive currency loss estimate including liquidation horizon and costs",
            "gross_turnover": "sum(abs(target_weight - current_weight))",
            "one_way_turnover": "0.5 * sum(abs(target_weight - current_weight))",
            "adv_values": "currency value per trading day",
            "cash_ticker": "CASH residual; no return/covariance series required",
        },
    }
    return result
