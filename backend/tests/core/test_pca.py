"""Tests for PCA factor extraction and covariance denoising."""

import numpy as np
import pandas as pd
import pytest

from app.core.research.pca import (
    denoise_covariance_matrix,
    extract_eigen_portfolios,
    fit_pca_factor_model,
    pca_telemetry,
)


def test_perfectly_correlated_assets_are_explained_by_one_component() -> None:
    first = np.array([-0.03, -0.01, 0.0, 0.02, 0.04])
    returns = pd.DataFrame({"MSFT": first, "AAPL": 2.0 * first})

    result = fit_pca_factor_model(returns, n_components=0.95)

    assert result["n_components"] == 1
    assert result["explained_variance_ratio"].iloc[0] == pytest.approx(1.0)
    assert result["total_explained_variance_ratio"] == pytest.approx(1.0)


def test_orthogonal_equal_variance_assets_have_equal_explained_ratios() -> None:
    returns = pd.DataFrame(
        {
            "Tech": [0.01, -0.01, 0.01, -0.01],
            "Energy": [0.01, 0.01, -0.01, -0.01],
        }
    )

    result = fit_pca_factor_model(returns, n_components=2)

    assert result["explained_variance_ratio"].to_numpy() == pytest.approx([0.5, 0.5])
    assert result["explained_variance_ratio"].sum() == pytest.approx(1.0)
    assert result["cumulative_explained_variance_ratio"].iloc[-1] == pytest.approx(1.0)


def test_float_component_threshold_selects_smallest_sufficient_prefix() -> None:
    rng = np.random.default_rng(40)
    common = rng.normal(size=300)
    returns = pd.DataFrame(
        {
            "Tech_1": common + rng.normal(scale=0.05, size=300),
            "Tech_2": common + rng.normal(scale=0.05, size=300),
            "Energy": rng.normal(scale=0.5, size=300),
        }
    )

    result = fit_pca_factor_model(returns, n_components=0.95)

    ratios = result["explained_variance_ratio"]
    assert result["cumulative_explained_variance_ratio"].iloc[-1] >= 0.95
    if len(ratios) > 1:
        assert result["cumulative_explained_variance_ratio"].iloc[-2] < 0.95


def test_denoised_covariance_is_symmetric_psd_and_preserves_variances() -> None:
    rng = np.random.default_rng(7)
    common = rng.normal(scale=0.02, size=250)
    returns = pd.DataFrame(
        {
            "Tech_1": common + rng.normal(scale=0.005, size=250),
            "Tech_2": common + rng.normal(scale=0.005, size=250),
            "Energy": rng.normal(scale=0.015, size=250),
        }
    )

    denoised = denoise_covariance_matrix(returns, n_components=1)

    np.testing.assert_allclose(denoised, denoised.T, atol=1e-14)
    assert np.linalg.eigvalsh(denoised.to_numpy()).min() >= -1e-12
    np.testing.assert_allclose(
        np.diag(denoised), returns.var(ddof=1).to_numpy(), rtol=1e-10, atol=1e-14
    )


def test_eigen_portfolios_have_unit_gross_exposure() -> None:
    rng = np.random.default_rng(10)
    returns = pd.DataFrame(rng.normal(size=(30, 5)), columns=list("ABCDE"))

    portfolios = extract_eigen_portfolios(returns, n_components=3)

    assert portfolios.shape == (3, 5)
    np.testing.assert_allclose(np.abs(portfolios.to_numpy()).sum(axis=1), 1.0)
    assert list(portfolios.index) == ["PC1", "PC2", "PC3"]


def test_pca_telemetry_reports_variance_and_covariance_condition_numbers() -> None:
    rng = np.random.default_rng(15)
    common = rng.normal(size=100)
    returns = pd.DataFrame(
        {
            "Tech_1": common + rng.normal(scale=0.1, size=100),
            "Tech_2": common + rng.normal(scale=0.1, size=100),
            "Energy": rng.normal(size=100),
        }
    )
    fit = fit_pca_factor_model(returns, n_components=2)
    telemetry = pca_telemetry(
        returns,
        explained_variance_ratio=fit["explained_variance_ratio"].to_numpy(),
        n_components=1,
    )

    assert telemetry["latency_ms"] >= 0.0
    assert telemetry["input_shape"] == [100, 3]
    assert len(telemetry["explained_variance_ratio"]) == 2
    assert telemetry["condition_number_raw"] > 0.0
    assert telemetry["condition_number_denoised"] > 0.0
    assert fit["telemetry"]["latency_ms"] >= 0.0


@pytest.mark.parametrize(
    "n_components",
    [0, -1, 4, 1.1, 0.0, True],
)
def test_invalid_component_counts_are_rejected(n_components: int | float) -> None:
    returns = pd.DataFrame(np.random.default_rng(1).normal(size=(5, 3)))
    with pytest.raises(ValueError):
        fit_pca_factor_model(returns, n_components)


@pytest.mark.parametrize(
    "returns",
    [
        pd.DataFrame(),
        pd.DataFrame({"A": [0.01]}),
        pd.DataFrame({"A": [0.01, np.nan]}),
        pd.DataFrame({"A": [0.01, np.inf]}),
        pd.DataFrame({"A": [0.01, 0.01], "B": [0.02, 0.02]}),
    ],
)
def test_invalid_or_degenerate_return_panels_are_rejected(returns: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        fit_pca_factor_model(returns)

