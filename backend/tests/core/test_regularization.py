"""Tests for regularized alpha models and bias-variance diagnostics."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from app.core.research.regularization import (
    evaluate_bias_variance_tradeoff,
    extract_active_feature_mask,
    fit_regularized_alpha_model,
)


def _signal_data(n: int = 500, seed: int = 23):
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(n, 12))
    columns = [f"factor_{index}" for index in range(values.shape[1])]
    X = pd.DataFrame(values, columns=columns)
    y = pd.Series(4.0 * values[:, 0] - 2.5 * values[:, 1] + rng.normal(scale=0.2, size=n))
    return X, y


def test_lasso_zeros_irrelevant_noise_features_and_returns_telemetry():
    X, y = _signal_data()

    result = fit_regularized_alpha_model(X, y, model_type="lasso", alpha=0.08)
    mask = extract_active_feature_mask(result)

    assert mask["factor_0"]
    assert mask["factor_1"]
    assert int(mask.sum()) <= 3
    assert result["n_zeroed_features"] == len(X.columns) - int(mask.sum())
    assert result["train_mse"] >= 0
    assert result["test_mse"] >= 0
    assert result["latency_ms"] >= 0
    assert result["n_train"] + result["n_test"] == len(X)


def test_ridge_stabilizes_collinear_factor_weights():
    rng = np.random.default_rng(5)
    base = rng.normal(size=300)
    X = pd.DataFrame(
        {"factor_a": base, "factor_b": base + rng.normal(scale=1e-5, size=300)}
    )
    y = pd.Series(2.0 * base + rng.normal(scale=0.1, size=300))

    result = fit_regularized_alpha_model(X, y, model_type="ridge", alpha=1.0)
    coefficients = result["coefficients"]

    assert np.isfinite(coefficients.to_numpy()).all()
    assert np.linalg.norm(coefficients.to_numpy()) < 3.0
    assert coefficients["factor_a"] == pytest.approx(coefficients["factor_b"], abs=0.2)


def test_zero_alpha_matches_ordinary_least_squares_on_same_split():
    X, y = _signal_data(n=120, seed=12)
    result = fit_regularized_alpha_model(
        X, y, model_type="elasticnet", alpha=0.0, validation_scheme="iid"
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    ols = make_pipeline(StandardScaler(), LinearRegression()).fit(X_train, y_train)

    assert result["train_mse"] == pytest.approx(
        np.mean((y_train.to_numpy() - ols.predict(X_train)) ** 2), abs=1e-12
    )
    assert result["test_mse"] == pytest.approx(
        np.mean((y_test.to_numpy() - ols.predict(X_test)) ** 2), abs=1e-12
    )


def test_tradeoff_evaluation_reports_holdout_and_cross_validation_errors():
    X, y = _signal_data(n=150, seed=99)
    dates = pd.date_range("2020-01-01", periods=len(X), freq="B")
    X.index = y.index = dates

    results = evaluate_bias_variance_tradeoff(
        X, y, alphas=[0.0, 0.1, 10.0], gap_periods=2
    )

    assert list(results.columns) == [
        "alpha",
        "train_mse",
        "cv_mse",
        "cv_mse_std",
        "selected",
        "test_mse",
        "latency_ms",
    ]
    assert results["alpha"].tolist() == [0.0, 0.1, 10.0]
    selected_index = results["cv_mse"].idxmin()
    assert results.loc[selected_index, "selected"]
    assert results.attrs["selected_alpha"] == results.loc[selected_index, "alpha"]
    assert results.attrs["selection_metric"] == "cv_mse"
    assert results.attrs["validation_scheme"] == "time_series"
    assert results.attrs["train_end_index"] < results.attrs["test_start_index"]
    assert results.attrs["gap_periods"] == 2
    assert all(
        train_end < validation_start
        and (validation_start - train_end).days >= 3
        for train_end, validation_start in results.attrs["cv_split_boundaries"]
    )
    assert results["selected"].sum() == 1
    assert results["test_mse"].notna().sum() == 1
    assert results.loc[selected_index, "test_mse"] == pytest.approx(
        results.attrs["final_test_mse"]
    )
    assert (results[["train_mse", "cv_mse"]] >= 0).all().all()
    assert (results["latency_ms"] >= 0).all()


def test_time_series_cv_metrics_do_not_depend_on_future_holdout_targets():
    X, y = _signal_data(n=120, seed=102)
    dates = pd.date_range("2021-01-01", periods=len(X), freq="B")
    X.index = y.index = dates
    changed_y = y.copy()
    changed_y.iloc[96:] += 1_000_000.0

    first = evaluate_bias_variance_tradeoff(X, y, alphas=[0.01, 1.0, 100.0])
    second = evaluate_bias_variance_tradeoff(X, changed_y, alphas=[0.01, 1.0, 100.0])

    assert np.allclose(first["cv_mse"], second["cv_mse"])
    assert first.attrs["selected_alpha"] == second.attrs["selected_alpha"]
    assert first.attrs["train_end_index"] == second.attrs["train_end_index"]


def test_time_series_validation_requires_chronologically_ordered_index():
    X, y = _signal_data(n=40, seed=1)
    reversed_X, reversed_y = X.iloc[::-1], y.iloc[::-1]
    with pytest.raises(ValueError, match="monotonically increasing"):
        evaluate_bias_variance_tradeoff(reversed_X, reversed_y, alphas=[0.1, 1.0])


def test_iid_validation_scheme_is_explicitly_available():
    X, y = _signal_data(n=60, seed=42)
    results = evaluate_bias_variance_tradeoff(
        X, y, alphas=[0.1, 1.0], validation_scheme="iid"
    )
    assert results.attrs["validation_scheme"] == "iid"
    assert results.attrs["selection_metric"] == "cv_mse"


def test_features_and_target_align_by_index_and_drop_missing_rows():
    X = pd.DataFrame({"x": [1.0, 2.0, np.nan, 4.0]}, index=[10, 11, 12, 13])
    y = pd.Series([10.0, 20.0, 30.0, 40.0], index=[10, 11, 12, 13])
    X = pd.concat([X] * 3, ignore_index=True)
    y = pd.concat([y] * 3, ignore_index=True)
    X.loc[2, "x"] = np.nan

    result = fit_regularized_alpha_model(X, y, model_type="ridge", alpha=0.5)

    assert result["n_train"] + result["n_test"] == len(X.dropna())


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"model_type": "svm"}, "model_type"),
        ({"alpha": -1.0}, "alpha"),
        ({"alpha": float("inf")}, "alpha"),
        ({"l1_ratio": 1.1}, "l1_ratio"),
    ],
)
def test_fit_rejects_invalid_model_parameters(kwargs, message):
    X, y = _signal_data(n=50)
    with pytest.raises(ValueError, match=message):
        fit_regularized_alpha_model(X, y, **kwargs)


def test_fit_rejects_too_few_or_unaligned_rows():
    with pytest.raises(ValueError, match="at least 5"):
        fit_regularized_alpha_model(
            pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]}), pd.Series([1.0, 2.0, 3.0, 4.0])
        )
    with pytest.raises(ValueError, match="aligned"):
        fit_regularized_alpha_model(
            pd.DataFrame({"x": [1.0] * 5}, index=range(5)),
            pd.Series([1.0] * 5, index=range(5, 10)),
        )


def test_active_feature_mask_validates_result_shape():
    with pytest.raises(ValueError, match="coefficient"):
        extract_active_feature_mask({})
