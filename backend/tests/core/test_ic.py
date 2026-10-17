"""Tests for information coefficients and factor predictability summaries."""

import numpy as np
import pandas as pd
import pytest

from app.core.research.ic import (
    calculate_ic_time_series,
    calculate_ic_time_series_with_telemetry,
    calculate_information_ratio,
    calculate_normal_ic,
    calculate_rank_ic,
    information_ratio_with_telemetry,
)


def test_rank_and_normal_ic_perfect_positive_and_inverse() -> None:
    factor = pd.Series([1.0, 2.0, 3.0, 4.0], index=list("ABCD"))
    positive = pd.Series([0.01, 0.02, 0.03, 0.04], index=list("ABCD"))
    inverse = pd.Series([0.04, 0.03, 0.02, 0.01], index=list("ABCD"))

    assert calculate_rank_ic(factor, positive) == pytest.approx(1.0)
    assert calculate_normal_ic(factor, positive) == pytest.approx(1.0)
    assert calculate_rank_ic(factor, inverse) == pytest.approx(-1.0)
    assert calculate_normal_ic(factor, inverse) == pytest.approx(-1.0)


def test_uncorrelated_four_stock_rankings_have_zero_ic() -> None:
    # Ranks [2, 4, 1, 3] have zero rank covariance with [1, 2, 3, 4].
    factor = pd.Series([1.0, 2.0, 3.0, 4.0])
    forward = pd.Series([2.0, 4.0, 1.0, 3.0])

    assert calculate_rank_ic(factor, forward) == pytest.approx(0.0, abs=1e-14)
    assert calculate_normal_ic(factor, forward) == pytest.approx(0.0, abs=1e-14)


def test_series_alignment_and_nan_pairs_are_handled_pairwise() -> None:
    factor = pd.Series([1.0, 2.0, np.nan, 4.0, 99.0], index=list("ABCDE"))
    forward = pd.Series([0.01, 0.02, 0.03, 0.04, 0.99], index=list("ABCDZ"))

    assert calculate_rank_ic(factor, forward) == pytest.approx(1.0)
    assert calculate_normal_ic(factor, forward) == pytest.approx(1.0)


def test_ic_time_series_is_cross_sectional_for_each_date() -> None:
    assets = list("ABCD")
    factors = pd.DataFrame(
        [[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]],
        index=["t1", "t2"],
        columns=assets,
    )
    forward = pd.DataFrame(
        [[0.01, 0.02, 0.03, 0.04], [0.04, 0.03, 0.02, 0.01]],
        index=["t1", "t2"],
        columns=assets,
    )

    result = calculate_ic_time_series(factors, forward)

    pd.testing.assert_series_equal(
        result,
        pd.Series([1.0, -1.0], index=["t1", "t2"], name="rank_ic"),
    )


def test_panel_alignment_missing_values_and_coverage_telemetry() -> None:
    factors = pd.DataFrame(
        [[1.0, 2.0, np.nan, 4.0], [1.0, 2.0, 3.0, 4.0]],
        index=["t1", "t2"],
        columns=list("ABCD"),
    )
    forward = pd.DataFrame(
        [[0.01, 0.02, 0.03, 0.04], [0.04, 0.03, 0.02, 0.01]],
        index=["t1", "t2"],
        columns=list("ABCD"),
    )

    result, telemetry = calculate_ic_time_series_with_telemetry(factors, forward)

    assert result.loc["t1"] == pytest.approx(1.0)
    assert result.loc["t2"] == pytest.approx(-1.0)
    assert telemetry["latency_ms"] >= 0.0
    assert telemetry["time_periods_evaluated"] == 2
    assert telemetry["valid_pairs"] == 7
    assert telemetry["possible_pairs"] == 8
    assert telemetry["coverage_ratio"] == pytest.approx(7 / 8)


def test_information_ratio_matches_manual_mean_and_sample_std() -> None:
    ic_series = pd.Series([0.1, 0.2, -0.1, np.nan, 0.3])
    observed = np.array([0.1, 0.2, -0.1, 0.3])

    result = calculate_information_ratio(ic_series)

    assert result["mean_ic"] == pytest.approx(observed.mean())
    assert result["ic_std"] == pytest.approx(observed.std(ddof=1))
    assert result["information_ratio"] == pytest.approx(observed.mean() / observed.std(ddof=1))
    assert result["observations"] == 4


def test_information_ratio_constant_ic_has_undefined_ir_and_telemetry() -> None:
    series = pd.Series([0.25, 0.25, 0.25])

    result = calculate_information_ratio(series)
    summary, telemetry = information_ratio_with_telemetry(series)

    assert result["mean_ic"] == pytest.approx(0.25)
    assert result["ic_std"] == pytest.approx(0.0)
    assert np.isnan(result["information_ratio"])
    assert summary["mean_ic"] == pytest.approx(result["mean_ic"])
    assert summary["ic_std"] == pytest.approx(result["ic_std"])
    assert np.isnan(summary["information_ratio"])
    assert summary["observations"] == result["observations"]
    assert telemetry["time_periods_evaluated"] == 3
    assert telemetry["coverage_ratio"] == pytest.approx(1.0)


def test_empty_series_and_insufficient_pairs_have_clear_behavior() -> None:
    empty = pd.Series([], dtype=float)
    with pytest.raises(ValueError, match="two aligned"):
        calculate_rank_ic(empty, empty)
    with pytest.raises(ValueError, match="two aligned"):
        calculate_normal_ic(empty, empty)
    with pytest.raises(ValueError, match="at least one non-missing"):
        calculate_information_ratio(empty)

    empty_panel = pd.DataFrame(dtype=float)
    assert calculate_ic_time_series(empty_panel, empty_panel).empty


def test_degenerate_cross_section_returns_nan_and_preserves_dates() -> None:
    factors = pd.DataFrame([[1.0, 1.0, np.nan]], index=["t1"], columns=list("ABC"))
    forward = pd.DataFrame([[0.01, 0.02, 0.03]], index=["t1"], columns=list("ABC"))

    result = calculate_ic_time_series(factors, forward)

    assert result.index.tolist() == ["t1"]
    assert np.isnan(result.iloc[0])
