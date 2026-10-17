from datetime import date, datetime, timedelta, timezone

import numpy as np
import pytest

from app.schemas.advisor import AdvisorMarketSnapshot, AdvisorPolicy


@pytest.fixture
def market_snapshot() -> AdvisorMarketSnapshot:
    rng = np.random.default_rng(2026)
    count = 160
    observations = rng.normal(loc=[0.00035, 0.00025], scale=[0.012, 0.009], size=(count, 2))
    dates = []
    day = date(2025, 1, 2)
    while len(dates) < count:
        if day.weekday() < 5:
            dates.append(day)
        day += timedelta(days=1)
    return AdvisorMarketSnapshot(
        as_of=datetime(2025, 8, 15, tzinfo=timezone.utc),
        tickers=["AAA", "BBB"],
        observation_dates=dates,
        daily_returns=observations.tolist(),
        prices=[100.0, 50.0],
        adv_values=[50_000_000.0, 40_000_000.0],
        spreads=[0.001, 0.0015],
    )


@pytest.fixture
def advisor_policy() -> AdvisorPolicy:
    return AdvisorPolicy(
        holding_period_days=60,
        minimum_holding_period_days=30,
        allocation_method="risk_parity",
        fractional_kelly=0.20,
        max_one_way_turnover=0.08,
        turnover_penalty_factor=2.0,
        regime_lookback_months=6,
        capital=1_000_000,
    )
