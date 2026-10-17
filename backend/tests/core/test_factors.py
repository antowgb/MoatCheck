"""Tests for pure factor winsorization and standardization."""

import numpy as np
import pandas as pd
import pytest

from app.core.research.factors import (
    combine_factor_scores,
    factor_stats,
    preprocess_factor,
    preprocess_factor_with_telemetry,
    winsorize_factor,
    z_score_standardize,
)


def test_winsorize_preserves_nan_index_and_series_name() -> None:
    index = pd.Index(["A", "B", "C", "D"], name="ticker")
    factor = pd.Series([1.0, np.nan, 3.0, 100.0], index=index, name="growth")

    result = winsorize_factor(factor, limits=(0.34, 0.34))

    assert result.index.equals(index)
    assert result.name == "growth"
    assert result.iloc[1] != result.iloc[1]  # NaN remains NaN.
    assert result.iloc[0] == pytest.approx(3.0)
    assert result.iloc[2] == pytest.approx(3.0)
    assert result.iloc[3] == pytest.approx(3.0)


def test_winsorize_caps_extreme_upper_outlier() -> None:
    ordinary = np.arange(1.0, 101.0)
    factor = pd.Series(np.append(ordinary, 10_000.0), index=[f"T{i}" for i in range(101)])

    result = winsorize_factor(factor)

    assert result.index.equals(factor.index)
    assert result.max() == pytest.approx(100.0)
    assert result.iloc[-1] == pytest.approx(100.0)


def test_winsorize_handles_empty_and_all_nan_series() -> None:
    empty = pd.Series([], index=pd.Index([], name="ticker"), dtype=float, name="factor")
    all_missing = pd.Series([np.nan, np.nan], index=["A", "B"], name="factor")

    assert winsorize_factor(empty).equals(empty)
    assert winsorize_factor(all_missing).isna().all()
    assert winsorize_factor(all_missing).index.equals(all_missing.index)


@pytest.mark.parametrize("limits", [(-0.1, 0.01), (0.01, 1.0), (0.5, 0.5), (float("nan"), 0.0)])
def test_winsorize_rejects_invalid_limits(limits: tuple[float, float]) -> None:
    with pytest.raises(ValueError, match="limits"):
        winsorize_factor(pd.Series([1.0, 2.0, 3.0]), limits=limits)


def test_z_score_uses_population_standard_deviation_and_preserves_missing_values() -> None:
    factor = pd.Series([1.0, 2.0, 3.0, np.nan], index=["A", "B", "C", "D"], name="quality")

    result = z_score_standardize(factor)

    assert result.index.equals(factor.index)
    assert result.name == factor.name
    np.testing.assert_allclose(result.iloc[:3], [-np.sqrt(1.5), 0.0, np.sqrt(1.5)])
    assert np.isnan(result.iloc[3])
    assert result.iloc[:3].mean() == pytest.approx(0.0)
    assert result.iloc[:3].std(ddof=0) == pytest.approx(1.0)


def test_identical_values_and_single_observation_standardize_to_zero() -> None:
    identical = pd.Series([7.5, 7.5, 7.5], index=["A", "B", "C"])
    singleton = pd.Series([4.0], index=["ONLY"])

    pd.testing.assert_series_equal(z_score_standardize(identical), pd.Series([0.0, 0.0, 0.0], index=identical.index))
    pd.testing.assert_series_equal(z_score_standardize(singleton), pd.Series([0.0], index=singleton.index))


def test_preprocess_winsorizes_before_standardizing_and_preserves_index() -> None:
    ordinary = np.arange(1.0, 101.0)
    factor = pd.Series(np.append(ordinary, 10_000.0), index=[f"T{i}" for i in range(101)])

    result = preprocess_factor(factor)

    assert result.index.equals(factor.index)
    assert result.mean() == pytest.approx(0.0, abs=1e-12)
    assert result.std(ddof=0) == pytest.approx(1.0)
    assert result.iloc[-1] == pytest.approx(result.iloc[-2])


def test_preprocess_preserves_nan_positions() -> None:
    factor = pd.Series([2.0, np.nan, 2.0], index=["A", "MISSING", "C"])

    result = preprocess_factor(factor)

    assert result.index.equals(factor.index)
    assert result.iloc[0] == 0.0
    assert np.isnan(result.iloc[1])
    assert result.iloc[2] == 0.0


def test_factor_stats_and_telemetry_report_before_after_values() -> None:
    factor = pd.Series([1.0, 2.0, 3.0, np.nan], index=list("ABCD"), name="value")

    assert factor_stats(factor) == {
        "count": 3,
        "mean": 2.0,
        "std": pytest.approx(np.sqrt(2.0 / 3.0)),
        "min": 1.0,
        "max": 3.0,
    }
    result, telemetry = preprocess_factor_with_telemetry(factor)

    assert result.index.equals(factor.index)
    assert telemetry["latency_ms"] >= 0.0
    assert telemetry["before"]["mean"] == pytest.approx(2.0)
    assert telemetry["after"]["mean"] == pytest.approx(0.0, abs=1e-12)
    assert telemetry["input_shape"] == [4]
    assert telemetry["output_shape"] == [4]


def test_factor_transformations_reject_non_series_and_infinite_values() -> None:
    with pytest.raises(TypeError, match="Series"):
        z_score_standardize([1.0, 2.0])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite"):
        winsorize_factor(pd.Series([1.0, np.inf]))


def test_existing_factor_score_aggregation_remains_available() -> None:
    result = combine_factor_scores(
        {"quality": {"A": 80.0, "B": 60.0}, "value": {"A": 50.0, "B": 100.0}},
        {"quality": 3.0, "value": 1.0},
    )

    assert result["scores"] == {"A": 72.5, "B": 70.0}
