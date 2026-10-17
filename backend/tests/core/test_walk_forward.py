import numpy as np

from app.core.backtest.walk_forward import run_walk_forward, walk_forward_telemetry


def test_walk_forward_uses_prior_scores_and_reports_telemetry():
    returns = np.array(
        [[0.0, 0.0], [0.0, 0.0], [0.1, -0.1], [-0.1, 0.1], [0.02, 0.03], [0.0, 0.0]]
    )
    scores = np.array(
        [[10.0, 0.0], [0.0, 10.0], [10.0, 0.0], [0.0, 10.0], [10.0, 0.0], [0.0, 10.0]]
    )

    result = run_walk_forward(
        returns, scores, assets=["A", "B"], lookback_periods=2,
        rebalance_every=1, top_n=1, transaction_cost_bps=0,
    )
    telemetry = walk_forward_telemetry(
        returns, scores, assets=["A", "B"], lookback_periods=2,
        rebalance_every=1, top_n=1, transaction_cost_bps=0,
    )

    assert result["holdings"][0]["assets"] == ["B"]
    assert result["equity_curve"][0]["return"] == -0.1
    assert telemetry["input_shape"] == [6, 2]
    assert telemetry["n_test_periods"] == 4
    assert telemetry["n_rebalances"] == 4
    assert telemetry["lookahead_policy"] == result["lookahead_policy"]
    assert telemetry["latency_ms"] >= 0.0
