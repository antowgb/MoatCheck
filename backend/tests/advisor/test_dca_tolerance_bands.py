from datetime import date

import pytest

from app.advisor.service import (
    allocate_smart_dca,
    calculate_tolerance_bands,
    evaluate_trade_signal,
)
from app.services.email_service import build_monthly_dca_email


def test_volatility_adjusted_tolerance_band_and_drift_classification():
    rows = calculate_tolerance_bands(
        {"AAA": 0.38, "BBB": 0.20, "CCC": 0.42},
        {"AAA": 0.30, "BBB": 0.30, "CCC": 0.40},
        {"AAA": 0.20, "BBB": 0.01, "CCC": 0.30},
    )
    by_ticker = {row["ticker"]: row for row in rows}
    assert by_ticker["AAA"]["band_width"] == pytest.approx(0.06)
    assert by_ticker["AAA"]["status"] == "DRIFT_UP"
    assert by_ticker["BBB"]["band_width"] == pytest.approx(0.02)
    assert by_ticker["BBB"]["status"] == "DRIFT_DOWN"
    assert by_ticker["CCC"]["status"] == "BALANCED"


def test_trade_signal_requires_all_three_gates():
    drift = [{"ticker": "AAA", "status": "DRIFT_UP"}]
    assert evaluate_trade_signal(drift, 0.20, True)["triggered"] is True
    assert evaluate_trade_signal(drift, 0.15, True)["triggered"] is False
    assert evaluate_trade_signal(drift, 0.149, True)["triggered"] is False
    assert evaluate_trade_signal(drift, 0.20, False)["reason"] == "minimum_holding_period_not_satisfied"
    assert evaluate_trade_signal([{"ticker": "AAA", "status": "BALANCED"}], 1.0, True)["reason"] == "no_asset_outside_tolerance_band"


def test_smart_dca_never_sells_or_buys_overweight_asset_while_underweight_exists():
    bands = [
        {"ticker": "OVER", "status": "DRIFT_UP"},
        {"ticker": "UNDER", "status": "DRIFT_DOWN"},
    ]
    allocation = allocate_smart_dca(
        500,
        {"OVER": 0.5, "UNDER": 0.5},
        {"OVER": 8000, "UNDER": 2000},
        10000,
        bands,
    )
    assert allocation["OVER"] == 0
    assert allocation["UNDER"] == pytest.approx(500)
    assert all(amount >= 0 for amount in allocation.values())


def test_monthly_dca_email_shows_buy_amounts_health_pnl_and_long_term_scenarios():
    recommendation = {
        "snapshot_timestamp": "2026-10-16T22:30:00+00:00",
        "trade_signal": {"triggered": True},
        "tolerance_bands": [{"ticker": "AAA", "status": "DRIFT_DOWN"}],
        "valuation": {"unrealized_pnl_fraction": 0.125},
        "dca_plan": {
            "monthly_amount": 500,
            "currency": "EUR",
            "orders": [{"ticker": "AAA", "side": "buy", "amount": 300}, {"ticker": "BBB", "side": "buy", "amount": 200}],
        },
        "projections": {
            "initial_capital": 10000,
            "annualized_expected_return": 0.06,
            "annualized_volatility": 0.15,
            "years": [
                {"year": year, "pessimistic_value": 1, "median_value": 2, "optimistic_value": 3}
                for year in (5, 10, 20)
            ],
        },
    }
    payload = build_monthly_dca_email(
        "person@example.com", "Core portfolio <main>", recommendation,
        cycle_date=date(2026, 11, 1),
    )
    assert payload["to"] == ["person@example.com"]
    assert "300.00 EUR" in payload["html"]
    assert "200.00 EUR" in payload["html"]
    assert "Review suggested allocation" in payload["html"]
    assert "+12.50%" in payload["html"]
    assert "20 years" in payload["html"]
    assert "Core portfolio &lt;main&gt;" in payload["html"]
    assert "No trades are placed" in payload["text"]
