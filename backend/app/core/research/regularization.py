"""Pure regularized linear models for factor-based alpha research."""

from __future__ import annotations

import math
from time import perf_counter
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.model_selection import KFold, TimeSeriesSplit, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


_MODEL_TYPES = {"ridge", "lasso", "elasticnet"}
ValidationScheme = Literal["time_series", "iid"]


def _validation_partition(
    features: pd.DataFrame,
    target: pd.Series,
    validation_scheme: ValidationScheme,
    gap_periods: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, dict[str, Any]]:
    """Create a chronological or explicitly IID train/holdout partition; O(N)."""
    if validation_scheme not in {"time_series", "iid"}:
        raise ValueError("validation_scheme must be 'time_series' or 'iid'")
    if isinstance(gap_periods, bool) or not isinstance(gap_periods, int) or gap_periods < 0:
        raise ValueError("gap_periods must be a non-negative integer")
    if validation_scheme == "iid":
        if gap_periods:
            raise ValueError("gap_periods is only supported for time_series validation")
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=42, shuffle=True
        )
        return X_train, X_test, y_train, y_test, {
            "validation_scheme": "iid",
            "gap_periods": 0,
            "train_end_index": None,
            "test_start_index": None,
        }

    if not features.index.is_monotonic_increasing:
        raise ValueError(
            "time_series validation requires X/y rows ordered by a monotonically increasing index"
        )
    n_rows = len(features)
    test_start = int(np.floor(n_rows * 0.8))
    train_stop = test_start - gap_periods
    if train_stop < 2 or test_start >= n_rows:
        raise ValueError("not enough rows for chronological train/test split and requested gap")
    return (
        features.iloc[:train_stop],
        features.iloc[test_start:],
        target.iloc[:train_stop],
        target.iloc[test_start:],
        {
            "validation_scheme": "time_series",
            "gap_periods": gap_periods,
            "train_end_index": features.index[train_stop - 1],
            "test_start_index": features.index[test_start],
        },
    )


def _prepare_xy(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    r"""Validate, align, and clean a factor matrix and forward-alpha target.

    Each row :math:`x_t` is paired with the target :math:`y_t` sharing the
    same index; rows with missing values are excluded and feature order is
    retained. This avoids fitting on misaligned dates or securities. Complexity:
    :math:`O(NP)` time and space for ``N`` observations and ``P`` factors.
    """
    if not isinstance(X, pd.DataFrame) or not isinstance(y, pd.Series):
        raise TypeError("X must be a pandas DataFrame and y a pandas Series")
    if X.shape[1] == 0:
        raise ValueError("X must contain at least one feature")
    if not X.index.is_unique or not y.index.is_unique:
        raise ValueError("X and y indexes must be unique")
    if not X.columns.is_unique:
        raise ValueError("X feature names must be unique")
    try:
        numeric_X = X.astype(float)
        numeric_y = y.astype(float).rename("target")
    except (TypeError, ValueError) as exc:
        raise TypeError("X and y must contain numeric values") from exc
    target_name = "__alpha_target__"
    while target_name in numeric_X.columns:
        target_name = f"_{target_name}"
    aligned = numeric_X.join(numeric_y.rename(target_name), how="inner").dropna()
    if aligned.empty:
        raise ValueError("X and y must have at least one aligned complete row")
    values = aligned.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("X and y must be finite wherever present")
    features = aligned.drop(columns=target_name)
    target = aligned[target_name]
    if len(features) < 5:
        raise ValueError("at least 5 aligned complete observations are required")
    return features, target


def _make_estimator(model_type: str, alpha: float, l1_ratio: float) -> Any:
    r"""Create the requested sklearn linear estimator or unpenalized OLS.

    Elastic Net minimizes squared error plus :math:`\alpha\rho\|w\|_1` and
    :math:`\frac{\alpha(1-\rho)}{2}\|w\|_2^2`; Ridge uses the L2 term and
    Lasso the L1 term. For :math:`\alpha=0`, ordinary least squares provides
    the same unpenalized objective without singular-penalty warnings. For a
    20-factor universe, L1 can zero irrelevant exposures while L2 shrinks
    correlated exposures smoothly. Complexity: estimator construction is
    :math:`O(1)` time and space.
    """
    if model_type not in _MODEL_TYPES:
        raise ValueError("model_type must be 'ridge', 'lasso', or 'elasticnet'")
    if alpha == 0.0:
        return LinearRegression()
    if model_type == "ridge":
        return Ridge(alpha=alpha)
    if model_type == "lasso":
        return Lasso(alpha=alpha, max_iter=20_000, tol=1e-7)
    return ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=20_000, tol=1e-7)


def _coefficient_series(pipeline: Any, feature_names: pd.Index) -> pd.Series:
    r"""Convert standardized pipeline coefficients back to input feature units.

    If :math:`z_j=(x_j-\mu_j)/s_j`, then the original-unit slope is
    :math:`\beta_j=\beta^{(z)}_j/s_j`. Mapping back makes the result useful for
    attribution while the active-feature mask remains based on fitted zeros.
    Complexity: :math:`O(P)` time and space for ``P`` factors.
    """
    scaler = pipeline.named_steps["standardscaler"]
    estimator = pipeline.steps[-1][1]
    coefficients = np.asarray(estimator.coef_, dtype=float) / scaler.scale_
    return pd.Series(coefficients, index=feature_names, name="coefficient")


def fit_regularized_alpha_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str = "elasticnet",
    alpha: float = 1.0,
    l1_ratio: float = 0.5,
    validation_scheme: ValidationScheme = "time_series",
    gap_periods: int = 0,
) -> dict[str, Any]:
    r"""Fit Ridge, Lasso, Elastic Net, or OLS to predict forward alpha.

    The Elastic Net objective is
    :math:`\min_{w,b}\frac{1}{2N}\|y-Xw-b\|_2^2+
    \alpha\rho\|w\|_1+\frac{\alpha(1-\rho)}{2}\|w\|_2^2`, where
    :math:`\rho` is ``l1_ratio``. Lasso can select a sparse subset, while Ridge
    stabilizes estimates when factors are correlated. For example, among 20
    candidate factors, Lasso may retain 3 nonzero signals and set 17 noisy
    coefficients to zero; Elastic Net can retain groups of related signals.
    The scaler is fit on training rows only. By default, an 80/20 chronological
    holdout is used and optional ``gap_periods`` rows are left between train
    and holdout (set the gap at least as wide as any overlapping forward-label
    horizon). ``validation_scheme="iid"`` selects a seeded shuffled split and
    assumes independent, identically distributed observations; it is not
    appropriate for chronological market data. Chronological time-series
    validation should use TimeSeriesSplit and, for overlapping labels, purging
    and embargo. The result reports train/test MSE, latency, and zeroed-feature
    count. Complexity is approximately :math:`O(I NP)` per fit (algorithm and
    convergence dependent), with :math:`O(NP)` data/model memory.
    """
    if isinstance(model_type, str):
        model_type = model_type.lower()
    if model_type not in _MODEL_TYPES:
        raise ValueError("model_type must be 'ridge', 'lasso', or 'elasticnet'")
    try:
        alpha = float(alpha)
        l1_ratio = float(l1_ratio)
    except (TypeError, ValueError) as exc:
        raise ValueError("alpha and l1_ratio must be finite numbers") from exc
    if not math.isfinite(alpha) or alpha < 0.0:
        raise ValueError("alpha must be finite and non-negative")
    if not math.isfinite(l1_ratio) or not 0.0 <= l1_ratio <= 1.0:
        raise ValueError("l1_ratio must be finite and between 0 and 1")

    features, target = _prepare_xy(X, y)
    X_train, X_test, y_train, y_test, split_info = _validation_partition(
        features, target, validation_scheme, gap_periods
    )
    pipeline = make_pipeline(
        StandardScaler(), _make_estimator(model_type, alpha, l1_ratio)
    )
    start = perf_counter()
    pipeline.fit(X_train, y_train)
    train_predictions = pipeline.predict(X_train)
    test_predictions = pipeline.predict(X_test)
    train_mse = float(np.mean((y_train.to_numpy() - train_predictions) ** 2))
    test_mse = float(np.mean((y_test.to_numpy() - test_predictions) ** 2))
    latency_ms = float((perf_counter() - start) * 1000.0)
    coefficients = _coefficient_series(pipeline, features.columns)
    zeroed_count = int(np.count_nonzero(np.isclose(coefficients.to_numpy(), 0.0, atol=1e-10)))
    return {
        "model": pipeline,
        "model_type": model_type,
        "alpha": alpha,
        "l1_ratio": l1_ratio,
        "feature_names": features.columns.copy(),
        "coefficients": coefficients,
        "intercept": float(pipeline.predict(features.iloc[[0]])[0] - coefficients.dot(features.iloc[0])),
        "train_mse": train_mse,
        "test_mse": test_mse,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_features": int(features.shape[1]),
        "n_zeroed_features": zeroed_count,
        "latency_ms": latency_ms,
        **split_info,
    }


def evaluate_bias_variance_tradeoff(
    X: pd.DataFrame,
    y: pd.Series,
    alphas: list[float] | None = None,
    *,
    validation_scheme: ValidationScheme = "time_series",
    gap_periods: int = 0,
) -> pd.DataFrame:
    r"""Compare fit and validation error across regularization strengths.

    For each :math:`\alpha`, CV on the training partition supplies
    ``cv_mse`` and determines the selected penalty. The holdout is evaluated
    exactly once after selection; its ``test_mse`` appears only on the selected
    row and is never used to choose :math:`\alpha`. Small penalties may overfit
    noisy factors, while large penalties may underfit. With 20 factors, choose
    the penalty minimizing ``cv_mse`` and treat the final holdout score as an
    unbiased estimate of that selected model's out-of-sample error when the
    data assumptions hold. The default ``validation_scheme="time_series"``
    uses an ordered 80/20 holdout and ``TimeSeriesSplit`` without shuffling;
    ``gap_periods`` adds a gap before both holdout and CV validation folds.
    For labels spanning multiple periods, choose a sufficient gap or use a
    purged/embargoed splitter. ``validation_scheme="iid"`` instead uses a
    randomized holdout and shuffled K-Fold and assumes IID observations; it
    leaks future information when used on ordered market time series. In all
    schemes, StandardScaler is inside the estimator pipeline and is fitted
    separately within each training fold. Complexity
    is approximately :math:`O(AK I NP)` for ``A`` penalties, ``K`` folds,
    ``N`` rows, ``P`` factors, and solver iterations ``I``; memory is
    :math:`O(NP)`.
    """
    features, target = _prepare_xy(X, y)
    if alphas is None:
        alphas = np.logspace(-4, 4, 17).tolist()
    if not isinstance(alphas, (list, tuple, np.ndarray)) or len(alphas) == 0:
        raise ValueError("alphas must be a non-empty sequence of finite non-negative values")
    try:
        alpha_values = [float(value) for value in alphas]
    except (TypeError, ValueError) as exc:
        raise ValueError("alphas must contain finite numeric values") from exc
    if any(not math.isfinite(value) or value < 0.0 for value in alpha_values):
        raise ValueError("alphas must contain finite non-negative values")

    X_train, X_test, y_train, y_test, split_info = _validation_partition(
        features, target, validation_scheme, gap_periods
    )
    folds = min(5, len(X_train))
    if folds < 2:
        raise ValueError("at least two training observations are required for cross-validation")
    if validation_scheme == "time_series":
        folds = min(5, len(X_train) - gap_periods - 1)
        if folds < 2:
            raise ValueError("not enough training rows for chronological cross-validation and requested gap")
        cv = TimeSeriesSplit(n_splits=folds, gap=gap_periods)
    else:
        cv = KFold(n_splits=folds, shuffle=True, random_state=42)
    cv_splits = list(cv.split(X_train))
    if validation_scheme == "time_series":
        split_info["cv_split_boundaries"] = [
            (X_train.index[train[-1]], X_train.index[validation[0]])
            for train, validation in cv_splits
        ]
    rows: list[dict[str, float | int | bool]] = []
    for alpha_value in alpha_values:
        estimator = _make_estimator("ridge", alpha_value, 0.5)
        pipeline = make_pipeline(StandardScaler(), estimator)
        start = perf_counter()
        cv_mse = -cross_val_score(
            pipeline, X_train, y_train, cv=cv_splits, scoring="neg_mean_squared_error", n_jobs=None
        )
        pipeline.fit(X_train, y_train)
        train_mse = float(np.mean((y_train.to_numpy() - pipeline.predict(X_train)) ** 2))
        rows.append(
            {
                "alpha": alpha_value,
                "train_mse": train_mse,
                "cv_mse": float(np.mean(cv_mse)),
                "cv_mse_std": float(np.std(cv_mse, ddof=1)) if len(cv_mse) > 1 else 0.0,
                "selected": False,
                "test_mse": float("nan"),
                "latency_ms": float((perf_counter() - start) * 1000.0),
            }
        )
    best_row = int(np.argmin([float(row["cv_mse"]) for row in rows]))
    best_alpha = float(rows[best_row]["alpha"])
    rows[best_row]["selected"] = True
    final_model = make_pipeline(
        StandardScaler(), _make_estimator("ridge", best_alpha, 0.5)
    )
    final_model.fit(X_train, y_train)
    final_test_mse = float(
        np.mean((y_test.to_numpy() - final_model.predict(X_test)) ** 2)
    )
    rows[best_row]["test_mse"] = final_test_mse
    results = pd.DataFrame(rows)
    results.attrs["selected_alpha"] = best_alpha
    results.attrs["selection_metric"] = "cv_mse"
    results.attrs["final_test_mse"] = final_test_mse
    results.attrs.update(split_info)
    return results


def extract_active_feature_mask(model_result: dict[str, Any]) -> pd.Series:
    r"""Return a Boolean mask identifying features with active coefficients.

    The mask is :math:`m_j=\mathbf{1}(|\beta_j|>\epsilon)` for the fitted
    feature coefficients. In a 20-factor example, three selected nonzero
    exposures produce three ``True`` entries and 17 ``False`` entries. The
    result is useful as a selection diagnostic; Ridge generally keeps all
    coefficients active. Complexity: :math:`O(P)` time and space for ``P``
    model features.
    """
    if not isinstance(model_result, dict):
        raise TypeError("model_result must be a dictionary returned by the fit function")
    coefficients = model_result.get("coefficients")
    if not isinstance(coefficients, pd.Series):
        raise ValueError("model_result must contain a coefficient Series")
    return coefficients.abs().gt(1e-10).rename("is_active")
