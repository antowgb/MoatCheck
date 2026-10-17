"""Pure PCA factor extraction and low-rank covariance denoising."""

from __future__ import annotations

import math
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def _prepare_returns(returns_df: pd.DataFrame) -> np.ndarray:
    r"""Validate and extract a complete observations-by-assets return matrix.

    PCA decomposes centered observations :math:`X\in\mathbb{R}^{T\times N}`
    into orthogonal factors; every sample must have finite values for every
    asset. In a 10-stock panel of six Tech and four Energy names, complete-date
    filtering keeps all ten assets on the same observations. Complexity:
    :math:`O(TN)` time and space for :math:`T` observations and :math:`N` assets.
    """
    if not isinstance(returns_df, pd.DataFrame):
        raise TypeError("returns_df must be a pandas DataFrame")
    if returns_df.empty or returns_df.shape[1] == 0:
        raise ValueError("returns_df must contain observations and at least one asset")
    if returns_df.shape[0] < 2:
        raise ValueError("returns_df must contain at least two observations")
    if not returns_df.index.is_unique or not returns_df.columns.is_unique:
        raise ValueError("returns_df must have unique observation and asset labels")
    try:
        values = returns_df.to_numpy(dtype=float, na_value=np.nan, copy=True)
    except (TypeError, ValueError) as exc:
        raise TypeError("returns_df must contain numeric values") from exc
    if not np.isfinite(values).all():
        raise ValueError("returns_df must contain only finite, non-missing values")
    if not np.isfinite(values).any() or float(np.var(values, axis=0, ddof=1).sum()) <= 0.0:
        raise ValueError("returns_df must contain positive total variance")
    return values


def _validate_component_count(
    n_components: int | float, n_observations: int, n_assets: int
) -> int | float:
    r"""Validate an integer component count or cumulative-variance threshold.

    An integer :math:`k` retains exactly that many directions; a float
    :math:`q\in(0,1]` retains the smallest number explaining at least :math:`q`
    of total variance. In a 10-stock Tech/Energy universe where PC1 explains
    70% and PC2 explains 20%, ``0.9`` retains exactly those first two modes.
    Complexity:
    :math:`O(1)` time and space.
    """
    limit = min(n_observations, n_assets)
    if isinstance(n_components, (bool, np.bool_)):
        raise ValueError("n_components must be a positive integer or a float in (0, 1]")
    if isinstance(n_components, (int, np.integer)):
        if n_components < 1 or n_components > limit:
            raise ValueError(f"integer n_components must be between 1 and {limit}")
        return int(n_components)
    if isinstance(n_components, (float, np.floating)):
        value = float(n_components)
        if not math.isfinite(value) or value <= 0.0 or value > 1.0:
            raise ValueError("float n_components must be in (0, 1]")
        return value
    raise TypeError("n_components must be an integer or float")


def _raw_covariance(values: np.ndarray) -> np.ndarray:
    r"""Calculate the sample covariance matrix from complete return rows.

    The matrix is :math:`\Sigma=\frac{1}{T-1}(X-\bar X)^T(X-\bar X)`.
    For 10 stocks, it captures each asset's variance and every pairwise
    covariance; for example, a Tech/Energy covariance of :math:`0.0002`
    records shared daily movement. Complexity: :math:`O(TN^2)` time and
    :math:`O(N^2)` output space.
    """
    return np.atleast_2d(np.cov(values, rowvar=False, ddof=1))


def _denoised_covariance(values: np.ndarray, n_components: int) -> np.ndarray:
    r"""Reconstruct low-rank common covariance and diagonal residual risk.

    Spectral decomposition gives :math:`\Sigma=V\Lambda V^T`; retaining the
    first :math:`k` modes yields :math:`\Sigma_{den}=V_k\Lambda_kV_k^T+
    \operatorname{diag}(D)`, where :math:`D_i=\max(\Sigma_{ii}-(V_k\Lambda_k
    V_k^T)_{ii},0)`. In a 10-stock example, shared Tech moves may load on one
    common component while each Energy stock keeps its idiosyncratic diagonal
    variance; if PC1 explains 70% of ten-stock variance, that common mode is
    retained while smaller modes are treated as noise. Complexity:
    :math:`O(TN^2+N^3)` time and :math:`O(N^2)` space.
    """
    raw = _raw_covariance(values)
    model = PCA(n_components=n_components, svd_solver="full")
    model.fit(values)
    vectors = model.components_.T
    common = (vectors * model.explained_variance_) @ vectors.T
    residual_variance = np.maximum(np.diag(raw - common), 0.0)
    denoised = common + np.diag(residual_variance)
    return (denoised + denoised.T) / 2.0


def fit_pca_factor_model(
    returns_df: pd.DataFrame, n_components: int | float = 0.95
) -> dict[str, Any]:
    r"""Fit PCA factors to a complete returns panel and report variance shares.

    The centered panel has spectral decomposition :math:`X^TX=V\Lambda V^T`;
    the explained-variance share is :math:`EVR_j=\lambda_j/\sum_l\lambda_l`.
    Integer ``n_components`` keeps exactly :math:`k` components. A float such
    as ``0.95`` selects the smallest prefix reaching 95% cumulative variance.
    For 10 stocks split between Tech and Energy, a dominant shared sector
    movement may explain most variance in PC1 (for example, 70% for six Tech
    and four Energy names), while subsequent components capture smaller
    cross-sector and stock-specific moves. Output includes
    loadings, factor scores, variance ratios, condition numbers, and latency.
    Complexity: :math:`O(TN\min(T,N))` time and :math:`O(TN+N^2)` space for
    :math:`T` dates and :math:`N` assets.
    """
    start = perf_counter()
    values = _prepare_returns(returns_df)
    component_spec = _validate_component_count(
        n_components, values.shape[0], values.shape[1]
    )
    model = PCA(n_components=component_spec, svd_solver="full")
    scores = model.fit_transform(values)
    component_names = [f"PC{i + 1}" for i in range(model.n_components_)]
    loadings = pd.DataFrame(
        model.components_, index=component_names, columns=returns_df.columns
    )
    factor_returns = pd.DataFrame(scores, index=returns_df.index, columns=component_names)
    explained = pd.Series(
        model.explained_variance_, index=component_names, name="explained_variance"
    )
    ratios = pd.Series(
        model.explained_variance_ratio_,
        index=component_names,
        name="explained_variance_ratio",
    )
    cumulative = ratios.cumsum().rename("cumulative_explained_variance_ratio")
    denoised = _denoised_covariance(values, model.n_components_)
    telemetry = pca_telemetry(
        returns_df,
        explained_variance_ratio=ratios.to_numpy(),
        denoised_covariance=denoised,
    )
    telemetry["latency_ms"] = (perf_counter() - start) * 1000.0
    return {
        "n_components": int(model.n_components_),
        "components": loadings,
        "factor_returns": factor_returns,
        "explained_variance": explained,
        "explained_variance_ratio": ratios,
        "cumulative_explained_variance_ratio": cumulative,
        "total_explained_variance_ratio": float(ratios.sum()),
        "telemetry": telemetry,
    }


def denoise_covariance_matrix(
    returns_df: pd.DataFrame, n_components: int = 3
) -> pd.DataFrame:
    r"""Denoise sample covariance by retaining common PCs and diagonal residuals.

    Given :math:`\Sigma=V\Lambda V^T`, this returns
    :math:`\Sigma_{den}=V_k\Lambda_kV_k^T+\operatorname{diag}(D)` with
    residual diagonal :math:`D`. A 10-stock Tech/Energy panel can retain
    broad market and sector factors while replacing noisy off-diagonal sample
    estimates with low-rank structure; if two PCs explain 90% of a 10-stock
    Tech/Energy panel, those modes retain its shared structure. The diagonal residual preserves each
    stock's variance. Complexity: :math:`O(TN^2+N^3)` time and
    :math:`O(N^2)` output space.
    """
    values = _prepare_returns(returns_df)
    component_count = _validate_component_count(
        n_components, values.shape[0], values.shape[1]
    )
    if not isinstance(component_count, int):
        raise ValueError("denoise_covariance_matrix requires an integer n_components")
    covariance = _denoised_covariance(values, component_count)
    return pd.DataFrame(covariance, index=returns_df.columns, columns=returns_df.columns)


def extract_eigen_portfolios(
    returns_df: pd.DataFrame, n_components: int = 3
) -> pd.DataFrame:
    r"""Return gross-exposure-normalized PCA loading portfolios.

    Component :math:`j` is the eigenvector :math:`v_j` satisfying
    :math:`\Sigma v_j=\lambda_jv_j`; this function scales it so
    :math:`\sum_i|w_{j,i}|=1`. For a 10-stock panel, PC1 may take positive
    weights across Tech names and negative weights across Energy names,
    expressing the dominant spread portfolio; for example, six Tech names may
    carry total long exposure 0.5 against 0.5 total short exposure in Energy.
    Its sign is arbitrary under
    eigenvector algebra, but sklearn applies deterministic sign orientation.
    Complexity: :math:`O(TN^2+N^3)` time and :math:`O(N^2)` output space.
    """
    values = _prepare_returns(returns_df)
    component_count = _validate_component_count(
        n_components, values.shape[0], values.shape[1]
    )
    if not isinstance(component_count, int):
        raise ValueError("extract_eigen_portfolios requires an integer n_components")
    model = PCA(n_components=component_count, svd_solver="full").fit(values)
    vectors = model.components_.copy()
    gross_exposure = np.abs(vectors).sum(axis=1, keepdims=True)
    portfolios = vectors / gross_exposure
    names = [f"PC{i + 1}" for i in range(component_count)]
    return pd.DataFrame(portfolios, index=names, columns=returns_df.columns)


def pca_telemetry(
    returns_df: pd.DataFrame,
    explained_variance_ratio: np.ndarray | None = None,
    denoised_covariance: pd.DataFrame | np.ndarray | None = None,
    n_components: int = 3,
) -> dict[str, Any]:
    r"""Report PCA latency, variance spectrum, and covariance conditioning.

    The helper reports :math:`\kappa(\Sigma)=\lambda_{max}/\lambda_{min}`
    for raw and denoised covariance, along with the explained-variance ratio
    vector. In a 10-stock Tech/Energy panel, a high raw condition number that
    falls after factor denoising indicates that weak noisy directions were
    regularized; removing small noisy eigenvalues from six Tech and four Energy
    names can lower that condition number. Complexity:
    :math:`O(TN^2+N^3)` time and :math:`O(N^2)` space.
    """
    start = perf_counter()
    values = _prepare_returns(returns_df)
    raw = _raw_covariance(values)
    if explained_variance_ratio is None:
        ratios = PCA(svd_solver="full").fit(values).explained_variance_ratio_
    else:
        try:
            ratios = np.asarray(explained_variance_ratio, dtype=float)
        except (TypeError, ValueError) as exc:
            raise TypeError("explained_variance_ratio must be numeric") from exc
        if ratios.ndim != 1 or not np.isfinite(ratios).all() or np.any(ratios < 0.0):
            raise ValueError("explained_variance_ratio must be a finite non-negative vector")

    if denoised_covariance is None:
        component_count = _validate_component_count(
            n_components, values.shape[0], values.shape[1]
        )
        if not isinstance(component_count, int):
            raise ValueError("telemetry n_components must be an integer")
        denoised = _denoised_covariance(values, component_count)
    elif isinstance(denoised_covariance, pd.DataFrame):
        if list(denoised_covariance.index) != list(returns_df.columns) or list(
            denoised_covariance.columns
        ) != list(returns_df.columns):
            raise ValueError("denoised_covariance labels must match returns_df columns")
        denoised = denoised_covariance.to_numpy(dtype=float, copy=True)
    else:
        try:
            denoised = np.asarray(denoised_covariance, dtype=float)
        except (TypeError, ValueError) as exc:
            raise TypeError("denoised_covariance must be numeric") from exc
    if denoised.shape != raw.shape or not np.isfinite(denoised).all():
        raise ValueError("denoised_covariance must be finite and match raw covariance shape")
    elapsed_ms = (perf_counter() - start) * 1000.0
    return {
        "latency_ms": float(elapsed_ms),
        "input_shape": [int(values.shape[0]), int(values.shape[1])],
        "explained_variance_ratio": [float(value) for value in ratios],
        "condition_number_raw": float(np.linalg.cond(raw)),
        "condition_number_denoised": float(np.linalg.cond(denoised)),
    }
