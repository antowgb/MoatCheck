"""Tests for point-in-time and survivorship backtest guards."""

import pandas as pd
import pytest

from app.core.research.backtest_guards import (
    apply_survivorship_filter,
    apply_survivorship_filter_with_telemetry,
    filter_point_in_time_data,
    filter_point_in_time_data_with_telemetry,
    validate_point_in_time_integrity,
    validate_point_in_time_integrity_with_telemetry,
)


def test_point_in_time_filter_observes_reporting_lag_and_inclusive_boundary() -> None:
    frame = pd.DataFrame(
        {
            "ticker": ["OLD", "Q4", "LATER"],
            "period_end": ["2024-09-30", "2024-12-31", "2024-12-31"],
            "know_date": ["2024-10-25", "2025-02-15", "2025-02-16"],
        },
        index=[10, 20, 30],
    )

    before_filing = filter_point_in_time_data(frame, pd.Timestamp("2025-02-14"))
    on_filing_date = filter_point_in_time_data(frame, pd.Timestamp("2025-02-15"))

    assert before_filing["ticker"].tolist() == ["OLD"]
    assert on_filing_date["ticker"].tolist() == ["OLD", "Q4"]
    assert on_filing_date.index.tolist() == [10, 20]


def test_missing_knowledge_dates_are_excluded_and_input_is_unchanged() -> None:
    frame = pd.DataFrame(
        {"ticker": ["KNOWN", "UNKNOWN"], "know_date": ["2025-01-01", None]}
    )
    original = frame.copy(deep=True)

    result = filter_point_in_time_data(frame, pd.Timestamp("2025-01-01"))

    assert result["ticker"].tolist() == ["KNOWN"]
    pd.testing.assert_frame_equal(frame, original)


def test_validate_pit_integrity_allows_equal_boundary_and_reporting_lag() -> None:
    valid = pd.DataFrame(
        {
            "period_end": ["2024-12-31", "2025-01-31"],
            "know_date": ["2025-02-15", "2025-01-31"],
        }
    )

    assert validate_point_in_time_integrity(valid) is True
    valid.loc[1, "know_date"] = "2025-02-01"
    assert validate_point_in_time_integrity(valid) is True


def test_validate_pit_integrity_detects_lookahead_leakage() -> None:
    leaked = pd.DataFrame(
        {
            "ticker": ["BAD", "GOOD"],
            "period_end": ["2024-12-31", "2024-12-31"],
            "know_date": ["2024-12-30", "2025-02-15"],
        }
    )

    with pytest.raises(ValueError, match="1 lookahead violation"):
        validate_point_in_time_integrity(leaked)
    valid, telemetry = validate_point_in_time_integrity_with_telemetry(leaked)
    assert valid is False
    assert telemetry["detected_lookahead_violations"] == 1
    assert telemetry["rows_checked"] == 2
    assert telemetry["latency_ms"] >= 0.0


def test_survivorship_filter_keeps_historically_alive_and_active_assets() -> None:
    universe = pd.DataFrame(
        {
            "ticker": ["ACTIVE", "DELISTED_TODAY", "DELISTED_BEFORE", "NOT_LISTED", "UNKNOWN_LISTING"],
            "first_listed": [
                "2020-01-01",
                "2010-01-01",
                "2010-01-01",
                "2025-02-16",
                None,
            ],
            "delisted_date": [None, "2025-02-15", "2025-02-14", None, None],
        },
        index=list("abcde"),
    )

    result = apply_survivorship_filter(universe, pd.Timestamp("2025-02-15"))

    assert result["ticker"].tolist() == ["ACTIVE", "DELISTED_TODAY"]
    assert result.index.tolist() == ["a", "b"]


def test_survivorship_filter_handles_nat_delisting_as_still_active() -> None:
    universe = pd.DataFrame(
        {"ticker": ["A", "B"], "first_listed": ["2020-01-01", "2026-01-01"], "delisted_date": [pd.NaT, pd.NaT]}
    )

    result = apply_survivorship_filter(universe, pd.Timestamp("2025-01-01"))

    assert result["ticker"].tolist() == ["A"]


def test_filter_telemetry_reports_rows_latency_and_shapes() -> None:
    frame = pd.DataFrame(
        {"ticker": ["A", "B", "C"], "know_date": ["2025-01-01", "2025-02-01", None]}
    )

    result, telemetry = filter_point_in_time_data_with_telemetry(
        frame, pd.Timestamp("2025-01-15")
    )

    assert result["ticker"].tolist() == ["A"]
    assert telemetry["latency_ms"] >= 0.0
    assert telemetry["input_rows"] == 3
    assert telemetry["filtered_rows"] == 1
    assert telemetry["excluded_rows"] == 2
    assert telemetry["input_shape"] == [3, 2]
    assert telemetry["output_shape"] == [1, 2]


def test_survivorship_telemetry_reports_included_and_excluded_assets() -> None:
    universe = pd.DataFrame(
        {
            "ticker": ["ACTIVE", "NOT_YET_LISTED"],
            "first_listed": ["2020-01-01", "2026-01-01"],
            "delisted_date": [pd.NaT, pd.NaT],
        }
    )

    result, telemetry = apply_survivorship_filter_with_telemetry(
        universe, pd.Timestamp("2025-01-01")
    )

    assert result["ticker"].tolist() == ["ACTIVE"]
    assert telemetry["latency_ms"] >= 0.0
    assert telemetry["input_rows"] == 2
    assert telemetry["filtered_rows"] == 1
    assert telemetry["excluded_rows"] == 1
    assert telemetry["input_shape"] == [2, 3]
    assert telemetry["output_shape"] == [1, 3]


def test_empty_frames_are_safe_and_empty_pit_integrity_is_valid() -> None:
    factors = pd.DataFrame(columns=["know_date", "period_end"])
    universe = pd.DataFrame(columns=["first_listed", "delisted_date"])

    assert filter_point_in_time_data(factors, pd.Timestamp("2025-01-01")).empty
    assert apply_survivorship_filter(universe, pd.Timestamp("2025-01-01")).empty
    assert validate_point_in_time_integrity(factors) is True


@pytest.mark.parametrize(
    ("frame", "when", "message"),
    [
        (pd.DataFrame({"know_date": ["not a date"]}), pd.Timestamp("2025-01-01"), "invalid timestamps"),
        (pd.DataFrame({"other": ["2025-01-01"]}), pd.Timestamp("2025-01-01"), "required date column"),
    ],
)
def test_point_in_time_filter_rejects_invalid_dates_and_missing_columns(
    frame: pd.DataFrame, when: pd.Timestamp, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        filter_point_in_time_data(frame, when)


def test_pit_validation_rejects_missing_integrity_dates() -> None:
    frame = pd.DataFrame({"know_date": [None], "period_end": ["2024-12-31"]})

    with pytest.raises(ValueError, match="must be present"):
        validate_point_in_time_integrity(frame)
