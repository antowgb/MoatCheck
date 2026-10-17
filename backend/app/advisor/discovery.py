"""Pure portfolio-discovery math for long-horizon replacement analysis.

The module accepts a dated return panel supplied by the application layer and
contains no database, network, filesystem, or web-framework dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from app.core.portfolio.optimizer import optimize_maximum_sharpe, optimize_minimum_variance


@dataclass(frozen=True)
class DiscoveryConfig:
    """Controls a point-in-time discovery run over a three-year return window."""

    lookback_years: int = 3
    periods_per_year: int = 252
    risk_free_rate: float = 0.03
    minimum_observations: int = 252
    model_universe_limit: int = 30
    candidate_limit: int = 10


def _moments(returns: pd.DataFrame, periods_per_year: int, risk_free_rate: float) -> tuple[pd.Series, pd.Series, pd.Series]:
    means = returns.mean() * periods_per_year
    volatility = returns.std(ddof=1) * np.sqrt(periods_per_year)
    sharpe = (means - risk_free_rate) / volatility.replace(0.0, np.nan)
    return means, volatility, sharpe


def _long_term_projection(expected_return: float, volatility: float) -> list[dict[str, float | int]]:
    """Return parametric log-normal value quantiles per €1 invested."""
    from scipy.stats import norm

    rows = []
    for years in (1, 3, 5, 10, 20):
        log_mean = (expected_return - 0.5 * volatility**2) * years
        log_std = volatility * np.sqrt(years)
        rows.append({
            "years": years,
            "p10_value_per_unit": float(np.exp(log_mean + norm.ppf(0.10) * log_std)),
            "p50_value_per_unit": float(np.exp(log_mean)),
            "p90_value_per_unit": float(np.exp(log_mean + norm.ppf(0.90) * log_std)),
        })
    return rows


def find_replacement_candidates(
    returns: pd.DataFrame,
    current_weights: dict[str, float],
    *,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.03,
    minimum_observations: int = 60,
    limit: int = 10,
) -> list[dict[str, Any]]:
    r"""Find candidates that beat a held asset on Sharpe and portfolio effect.

    Returns columns are assets and rows are chronologically ordered simple
    returns. For candidate :math:`j` replacing held asset :math:`i`, the
    candidate must have a higher standalone Sharpe and the hypothetical
    portfolio must have both higher Sharpe and lower correlation to the
    remaining holdings than the current portfolio. Correlations and moments
    use pairwise complete observations. A practical example is replacing a
    high-volatility holding only when the alternative improves its own
    risk-adjusted return and improves the portfolio as a whole.

    Complexity is :math:`O(TAH + TCA)` for :math:`T` dates, :math:`A`
    assets, :math:`H` holdings, and :math:`C` candidate replacements.
    """
    if not isinstance(returns, pd.DataFrame) or returns.empty:
        return []
    if periods_per_year < 1 or minimum_observations < 2 or limit < 1:
        raise ValueError("periods_per_year, minimum_observations, and limit must be positive")
    clean = returns.copy()
    clean.columns = [str(column).upper() for column in clean.columns]
    clean = clean.loc[:, ~clean.columns.duplicated()]
    clean = clean.replace([np.inf, -np.inf], np.nan)
    clean = clean.loc[:, clean.notna().sum() >= minimum_observations]
    if clean.empty or not current_weights:
        return []
    _, _, sharpes = _moments(clean, periods_per_year, risk_free_rate)
    held = {str(ticker).upper(): float(weight) for ticker, weight in current_weights.items() if float(weight) > 0}
    held = {ticker: weight for ticker, weight in held.items() if ticker in clean.columns}
    if not held:
        return []
    total = float(sum(held.values()))
    if total <= 0:
        return []
    held = {ticker: weight / total for ticker, weight in held.items()}
    base_panel = clean[list(held)].dropna(how="any")
    if len(base_panel) < minimum_observations:
        return []
    portfolio_returns = base_panel.mul(pd.Series(held), axis=1).sum(axis=1)
    base_vol = float(portfolio_returns.std(ddof=1) * np.sqrt(periods_per_year)) if len(portfolio_returns) > 1 else 0.0
    base_mean = float(portfolio_returns.mean() * periods_per_year) if len(portfolio_returns) else 0.0
    base_sharpe = (base_mean - risk_free_rate) / base_vol if base_vol > 0 else float("nan")
    if not np.isfinite(base_sharpe):
        return []
    candidates: list[dict[str, Any]] = []
    for source, source_weight in held.items():
        if source_weight <= 0 or not np.isfinite(sharpes.get(source, np.nan)):
            continue
        others = [ticker for ticker in held if ticker != source]
        for replacement in clean.columns:
            if replacement in held or not np.isfinite(sharpes.get(replacement, np.nan)):
                continue
            # Compare the old and proposed portfolios on exactly the same
            # dates. IPO gaps or suspended histories must not bias one side.
            comparison = clean[[*held, replacement]].dropna(how="any")
            if len(comparison) < minimum_observations:
                continue
            source_sharpe = float((comparison[source].mean() * periods_per_year - risk_free_rate)
                                  / (comparison[source].std(ddof=1) * np.sqrt(periods_per_year)))
            replacement_sharpe = float((comparison[replacement].mean() * periods_per_year - risk_free_rate)
                                       / (comparison[replacement].std(ddof=1) * np.sqrt(periods_per_year)))
            if not np.isfinite([source_sharpe, replacement_sharpe]).all() or replacement_sharpe <= source_sharpe:
                continue
            source_correlations = [comparison[source].corr(comparison[ticker]) for ticker in others]
            replacement_correlations = [comparison[replacement].corr(comparison[ticker]) for ticker in others]
            source_corr = float(np.mean(np.abs(source_correlations))) if others and np.isfinite(source_correlations).all() else 0.0
            replacement_corr = float(np.mean(np.abs(replacement_correlations))) if others and np.isfinite(replacement_correlations).all() else 0.0
            if others and (not np.isfinite(replacement_corr) or replacement_corr >= source_corr):
                continue
            proposed = {ticker: weight for ticker, weight in held.items() if ticker != source}
            proposed[replacement] = proposed.get(replacement, 0.0) + source_weight
            normalization = sum(proposed.values())
            proposed = {ticker: weight / normalization for ticker, weight in proposed.items()}
            old_portfolio_returns = comparison[list(held)].mul(pd.Series(held), axis=1).sum(axis=1)
            hypothetical = comparison[list(others)].mul(pd.Series({ticker: held[ticker] for ticker in others}), axis=1).sum(axis=1) if others else pd.Series(0.0, index=comparison.index)
            hypothetical = hypothetical + source_weight * comparison[replacement]
            comparison_base_vol = float(old_portfolio_returns.std(ddof=1) * np.sqrt(periods_per_year))
            comparison_base_mean = float(old_portfolio_returns.mean() * periods_per_year)
            comparison_base_sharpe = (comparison_base_mean - risk_free_rate) / comparison_base_vol if comparison_base_vol > 0 else float("nan")
            proposed_vol = float(hypothetical.std(ddof=1) * np.sqrt(periods_per_year))
            proposed_mean = float(hypothetical.mean() * periods_per_year)
            proposed_sharpe = (proposed_mean - risk_free_rate) / proposed_vol if proposed_vol > 0 else float("nan")
            if not np.isfinite(proposed_sharpe) or proposed_sharpe <= comparison_base_sharpe or proposed_vol >= comparison_base_vol:
                continue
            candidates.append({
                "replace_ticker": source,
                "with_ticker": str(replacement),
                "weight_to_transfer": source_weight,
                "asset_sharpe_before": source_sharpe,
                "asset_sharpe_after": replacement_sharpe,
                "portfolio_sharpe_before": float(comparison_base_sharpe),
                "portfolio_sharpe_after": float(proposed_sharpe),
                "portfolio_volatility_before": comparison_base_vol,
                "portfolio_volatility_after": proposed_vol,
                "absolute_correlation_to_other_holdings_before": source_corr,
                "absolute_correlation_to_other_holdings_after": replacement_corr,
                "target_weights": {ticker: float(value) for ticker, value in proposed.items() if value > 1e-12},
                "observations": int(min(clean[source].notna().sum(), clean[replacement].notna().sum())),
            })
    candidates.sort(key=lambda row: (row["portfolio_sharpe_after"] - row["portfolio_sharpe_before"], row["portfolio_volatility_before"] - row["portfolio_volatility_after"]), reverse=True)
    return candidates[:limit]


def build_portfolio_discovery_proposals(
    returns: pd.DataFrame,
    current_weights: dict[str, float],
    *,
    config: DiscoveryConfig | None = None,
) -> dict[str, Any]:
    r"""Build replacement ideas plus maximum-Sharpe and minimum-risk models.

    The supplied panel should contain the most recent three years of simple
    returns, with no future observations relative to the run timestamp. The
    implementation limits the optimizer universe to the highest historical
    Sharpe assets plus existing holdings, which keeps a 3,000-symbol screen
    from creating an impractical cubic optimizer problem. For example, the
    service can screen 3,000 assets and solve the two model portfolios on the
    top 30 eligible names. Model weights are research comparisons, not orders.

    Complexity is :math:`O(TAH + I K^2)` where :math:`K` is the capped model
    universe and :math:`I` the SLSQP iteration count; memory is :math:`O(TA)`.
    """
    started = perf_counter()
    settings = config or DiscoveryConfig()
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame")
    if settings.lookback_years != 3:
        raise ValueError("automated discovery currently requires a three-year lookback")
    if settings.periods_per_year < 1 or settings.minimum_observations < 2 or settings.model_universe_limit < 2:
        raise ValueError("discovery configuration is invalid")
    data = returns.copy()
    data.columns = [str(column).upper() for column in data.columns]
    data = data.loc[:, ~data.columns.duplicated()].replace([np.inf, -np.inf], np.nan)
    if not data.index.is_monotonic_increasing:
        data = data.sort_index()
    data = data.tail(settings.lookback_years * settings.periods_per_year)
    valid = [column for column in data if data[column].notna().sum() >= settings.minimum_observations]
    data = data[valid]
    if len(data) < settings.minimum_observations or data.shape[1] < 2:
        raise ValueError("three-year price coverage is insufficient for portfolio discovery")
    means, _, sharpes = _moments(data, settings.periods_per_year, settings.risk_free_rate)
    ranked = [str(ticker) for ticker in sharpes.dropna().sort_values(ascending=False).index]
    existing = [str(ticker).upper() for ticker in current_weights if str(ticker).upper() in data.columns]
    selected = list(dict.fromkeys([*existing, *ranked[: settings.model_universe_limit]]))[: settings.model_universe_limit]
    # Require complete common rows for covariance and both constrained solvers.
    model_returns = data[selected].dropna(how="any")
    if len(model_returns) < settings.minimum_observations:
        # Per-asset missingness is common in a large universe. Keep the best
        # covered model names before enforcing a common, stable sample.
        coverage = data[selected].notna().sum().sort_values(ascending=False)
        selected = list(coverage.head(min(len(selected), max(2, settings.model_universe_limit // 2))).index)
        model_returns = data[selected].dropna(how="any")
    if len(model_returns) < settings.minimum_observations or len(selected) < 2:
        raise ValueError("not enough aligned observations to optimize model portfolios")
    annual_mu = model_returns.mean() * settings.periods_per_year
    annual_cov = model_returns.cov() * settings.periods_per_year
    max_sharpe = optimize_maximum_sharpe(annual_mu, annual_cov, settings.risk_free_rate)
    minimum_risk = optimize_minimum_variance(annual_cov)
    for model in (max_sharpe, minimum_risk):
        if model.get("expected_return") is not None and model.get("volatility") is not None:
            model["long_term_projection_per_unit"] = _long_term_projection(
                float(model["expected_return"]), float(model["volatility"])
            )
    replacement_candidates = find_replacement_candidates(
        data, current_weights, periods_per_year=settings.periods_per_year,
        risk_free_rate=settings.risk_free_rate, minimum_observations=settings.minimum_observations,
        limit=settings.candidate_limit,
    )
    current = {str(key).upper(): float(value) for key, value in current_weights.items()
               if str(key).upper() in data.columns and float(value) > 0}
    current_total = sum(current.values())
    if current_total > 0:
        current = {ticker: weight / current_total for ticker, weight in current.items()}
    current_ret = data[list(ticker for ticker in current if ticker in data)].mul(
        pd.Series({ticker: weight for ticker, weight in current.items() if ticker in data}), axis=1
    ).sum(axis=1, min_count=1).dropna() if any(ticker in data for ticker in current) else pd.Series(dtype=float)
    current_vol = float(current_ret.std(ddof=1) * np.sqrt(settings.periods_per_year)) if len(current_ret) > 1 else None
    current_mean = float(current_ret.mean() * settings.periods_per_year) if len(current_ret) else None
    current_sharpe = ((current_mean - settings.risk_free_rate) / current_vol) if current_mean is not None and current_vol and current_vol > 0 else None
    swap_proposals: list[dict[str, Any]] = []
    for candidate in replacement_candidates:
        swap_proposals.append({"proposal_id": f"swap:{candidate['replace_ticker']}:{candidate['with_ticker']}",
                               "swap_count": 1, "swaps": [candidate], "target_weights": candidate["target_weights"],
                               "expected_return": candidate["portfolio_sharpe_after"] * candidate["portfolio_volatility_after"] + settings.risk_free_rate,
                               "volatility": candidate["portfolio_volatility_after"],
                               "sharpe_ratio": candidate["portfolio_sharpe_after"]})
    # Test non-overlapping pairs on the same dated return panel, not by adding
    # individual Sharpe deltas. This captures covariance interaction directly.
    for first_index, first in enumerate(replacement_candidates):
        for second in replacement_candidates[first_index + 1:]:
            if first["replace_ticker"] == second["replace_ticker"] or first["with_ticker"] == second["with_ticker"]:
                continue
            pair_weights = dict(current)
            for candidate in (first, second):
                source, destination = candidate["replace_ticker"], candidate["with_ticker"]
                transfer = pair_weights.pop(source, 0.0)
                pair_weights[destination] = pair_weights.get(destination, 0.0) + transfer
            pair_weights = {ticker: value for ticker, value in pair_weights.items() if value > 1e-12}
            pair_data = data[list(pair_weights)].dropna(how="any")
            if len(pair_data) < settings.minimum_observations:
                continue
            vector = pd.Series(pair_weights, dtype=float)
            vector /= vector.sum()
            pair_returns = pair_data.mul(vector, axis=1).sum(axis=1)
            pair_vol = float(pair_returns.std(ddof=1) * np.sqrt(settings.periods_per_year))
            pair_mean = float(pair_returns.mean() * settings.periods_per_year)
            pair_sharpe = (pair_mean - settings.risk_free_rate) / pair_vol if pair_vol > 0 else float("nan")
            if current_sharpe is not None and pair_sharpe > current_sharpe and current_vol is not None and pair_vol < current_vol:
                swap_proposals.append({"proposal_id": f"swap:{first['replace_ticker']}:{first['with_ticker']}+{second['replace_ticker']}:{second['with_ticker']}",
                                       "swap_count": 2, "swaps": [first, second],
                                       "target_weights": {ticker: float(weight) for ticker, weight in vector.items()},
                                       "expected_return": pair_mean, "volatility": pair_vol, "sharpe_ratio": pair_sharpe})
    swap_proposals.sort(key=lambda proposal: proposal["sharpe_ratio"], reverse=True)
    return {
        "lookback_years": settings.lookback_years,
        "periods_per_year": settings.periods_per_year,
        "observations": int(len(data)),
        "universe_assets_evaluated": int(data.shape[1]),
        "model_universe": selected,
        "current_portfolio": {"expected_return": current_mean, "volatility": current_vol, "sharpe_ratio": current_sharpe},
        "replacement_candidates": replacement_candidates,
        "swap_proposals": swap_proposals[: settings.candidate_limit],
        "model_portfolios": {
            "maximum_sharpe": max_sharpe,
            "minimum_risk": minimum_risk,
        },
        "methodology": {
            "return_type": "simple_periodic_returns",
            "annualization": "sample mean and covariance scaled by periods_per_year",
            "replacement_rule": "higher asset Sharpe, lower correlation to remaining holdings, higher portfolio Sharpe and lower portfolio volatility",
            "model_universe_limit": settings.model_universe_limit,
            "disclaimer": "Historical estimates are not forecasts. Proposals update saved target weights only and do not place trades.",
        },
        "telemetry": {
            "execution_time_ms": (perf_counter() - started) * 1000.0,
            "input_shape": {"return_panel": list(returns.shape), "evaluated_panel": list(data.shape)},
        },
    }
