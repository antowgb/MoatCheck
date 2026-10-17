from datetime import date, datetime, timedelta, timezone

import numpy as np
import pytest
from pydantic import ValidationError

from app.schemas.advisor import AdvisorMarketSnapshot, AdvisorPolicy, AdvisorPreviewRequest


def test_policy_rejects_day_trading_horizon():
    with pytest.raises(ValidationError):
        AdvisorPolicy(holding_period_days=5)


def test_policy_rejects_kelly_outside_conservative_band():
    with pytest.raises(ValidationError):
        AdvisorPolicy(fractional_kelly=0.5)


def test_market_snapshot_rejects_non_finite_returns():
    n = 63
    dates = [date(2025, 1, 1) + timedelta(days=i) for i in range(n)]
    # Move to business-day-like unique dates; calendar gaps are permitted.
    matrix = np.zeros((n, 1)).tolist()
    matrix[10][0] = float("nan")
    with pytest.raises((ValidationError, ValueError)):
        AdvisorMarketSnapshot(
            as_of=datetime(2025, 5, 1, tzinfo=timezone.utc),
            tickers=["AAA"], observation_dates=dates, daily_returns=matrix,
            prices=[10], adv_values=[1000], spreads=[0.01],
        )


def test_preview_requires_saved_or_complete_inline_portfolio(market_snapshot):
    with pytest.raises(ValidationError):
        AdvisorPreviewRequest(market=market_snapshot)
    req = AdvisorPreviewRequest(
        market=market_snapshot, tickers=["AAA", "BBB"],
        current_weights={"AAA": 0.5, "BBB": 0.5},
    )
    assert req.policy.holding_period_days == 90
