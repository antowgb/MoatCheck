from datetime import timedelta

import pytest

from app.advisor.service import build_advisory_recommendation


def test_recommendation_uses_medium_horizon_policy_and_positive_loss_conventions(market_snapshot, advisor_policy):
    result = build_advisory_recommendation(
        portfolio_name="Long Horizon",
        tickers=["AAA", "BBB"],
        current_weights={"AAA": 0.55, "BBB": 0.45},
        market=market_snapshot,
        policy=advisor_policy,
    )
    assert result["holding_period_days"] == 60
    assert result["var_loss_fraction"] >= 0
    assert result["cvar_loss_fraction"] >= result["var_loss_fraction"]
    assert result["liquidity_adjusted_var_currency"] >= 0
    assert result["gross_turnover"] == pytest.approx(2 * result["one_way_turnover"])
    assert result["one_way_turnover"] <= advisor_policy.max_one_way_turnover + 1e-10
    assert result["input_shape"]["returns"] == [160, 2]
    assert result["snapshot_timestamp"] == market_snapshot.as_of.isoformat()
    assert result["latency_ms"] >= 0
    assert any("turnover" in step["code"] for step in result["rationale"])


def test_minimum_holding_interval_blocks_recommendation(market_snapshot, advisor_policy):
    result = build_advisory_recommendation(
        portfolio_name="Long Horizon",
        tickers=["AAA", "BBB"],
        current_weights={"AAA": 0.55, "BBB": 0.45},
        market=market_snapshot,
        policy=advisor_policy,
        last_decision_at=market_snapshot.as_of - timedelta(days=10),
    )
    assert result["one_way_turnover"] == 0
    assert result["target_weights"] == result["current_weights"]
    assert any(step["code"] == "minimum_holding_interval_not_met" for step in result["rationale"])


def test_tickers_must_match_snapshot(market_snapshot, advisor_policy):
    with pytest.raises(ValueError, match="identical ordering"):
        build_advisory_recommendation(
            portfolio_name="Mismatch",
            tickers=["BBB", "AAA"],
            current_weights={"BBB": 0.5, "AAA": 0.5},
            market=market_snapshot,
            policy=advisor_policy,
        )


def test_allocation_inputs_are_arithmetic_moments_of_simple_returns(market_snapshot, advisor_policy):
    import numpy as np

    result = build_advisory_recommendation(
        portfolio_name="Arithmetic contract", tickers=["AAA", "BBB"],
        current_weights={"AAA": .5, "BBB": .5}, market=market_snapshot, policy=advisor_policy,
    )
    returns = np.asarray(market_snapshot.returns)
    periods = result["periods_per_year"]
    np.testing.assert_allclose(list(result["annualized_expected_returns"].values()), returns.mean(axis=0) * periods)
    np.testing.assert_allclose(result["annualized_covariance_matrix"], np.cov(returns, rowvar=False, ddof=1) * periods)
    weights = np.array([result["target_weights"][ticker] for ticker in market_snapshot.tickers])
    cash = result["target_weights"]["CASH"]
    portfolio_returns = returns @ weights + cash * np.expm1(np.log1p(advisor_policy.risk_free_rate) / periods)
    expected_growth = np.expm1(np.log1p(portfolio_returns).mean() * periods * advisor_policy.holding_period_days / 365.25)
    assert result["expected_holding_period_return_fraction"] == pytest.approx(expected_growth)
