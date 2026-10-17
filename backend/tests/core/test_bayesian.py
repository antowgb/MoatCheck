"""Tests for conjugate Gaussian Bayesian updating."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from app.core.research.bayesian import (
    bayes_update_normal,
    calculate_credible_interval,
    update_factor_prior_series,
    update_factor_prior_with_telemetry,
)


def test_flat_infinite_prior_returns_sample_mean_and_sampling_uncertainty():
    mean, std = bayes_update_normal(
        prior_mean=0.0,
        prior_std=float("inf"),
        sample_mean=0.12,
        sample_std=0.2,
        n_obs=16,
    )

    assert mean == pytest.approx(0.12)
    assert std == pytest.approx(0.05)


def test_zero_observation_noise_adopts_observation_exactly():
    mean, std = bayes_update_normal(0.25, 0.1, 0.20, 0.0)

    assert mean == pytest.approx(0.20)
    assert std == 0.0


def test_factor_series_zero_noise_replaces_aligned_priors_and_preserves_index():
    prior = pd.Series([0.25, 0.30, 0.35], index=["A", "B", "C"], name="margin")
    observations = pd.Series([0.20, 0.32], index=["A", "B"])

    posterior = update_factor_prior_series(prior, observations, observation_noise=0.0)

    assert posterior.index.equals(prior.index)
    assert posterior.name == prior.name
    assert posterior.loc["A"] == pytest.approx(0.20)
    assert posterior.loc["B"] == pytest.approx(0.32)
    assert posterior.loc["C"] == pytest.approx(0.35)


def test_sequential_updates_equal_one_combined_batch_update():
    prior_mean, prior_std = 0.10, 0.20
    batch_1_mean, batch_1_n = 0.16, 5
    batch_2_mean, batch_2_n = 0.22, 8
    observation_std = 0.12

    after_first = bayes_update_normal(
        prior_mean, prior_std, batch_1_mean, observation_std, batch_1_n
    )
    sequential = bayes_update_normal(
        after_first[0],
        after_first[1],
        batch_2_mean,
        observation_std,
        batch_2_n,
    )
    combined_mean = (batch_1_mean * batch_1_n + batch_2_mean * batch_2_n) / (
        batch_1_n + batch_2_n
    )
    combined = bayes_update_normal(
        prior_mean,
        prior_std,
        combined_mean,
        observation_std,
        batch_1_n + batch_2_n,
    )

    assert sequential == pytest.approx(combined)


def test_credible_interval_is_central_95_percent_normal_interval():
    mean, std = 0.04, 0.02

    lower, upper = calculate_credible_interval(mean, std, confidence_level=0.95)

    assert lower == pytest.approx(mean - norm.ppf(0.975) * std)
    assert upper == pytest.approx(mean + norm.ppf(0.975) * std)
    assert lower < mean < upper


def test_credible_interval_covers_about_95_percent_of_standard_normal_mass():
    lower, upper = calculate_credible_interval(0.0, 1.0, confidence_level=0.95)
    grid = np.linspace(-5.0, 5.0, 100_001)
    density = norm.pdf(grid)
    covered_mass = np.trapezoid(density[(grid >= lower) & (grid <= upper)], grid[(grid >= lower) & (grid <= upper)])

    assert covered_mass == pytest.approx(0.95, abs=2e-4)


def test_telemetry_reports_latency_shift_and_uncertainty_contraction():
    prior = pd.Series([0.25, 0.30], index=["A", "B"])
    observation = pd.Series([0.20, 0.35], index=["A", "B"])

    posterior, telemetry = update_factor_prior_with_telemetry(prior, observation, 0.05)

    assert telemetry["latency_ms"] >= 0
    assert telemetry["n_updated"] == 2
    assert telemetry["uncertainty_contraction_ratio"] == pytest.approx(
        np.sqrt(0.05**2 / (1.0 + 0.05**2))
    )
    assert telemetry["uncertainty_reduction_ratio"] > 0.9
    assert telemetry["mean_absolute_shift"] > 0
    assert posterior.loc["A"] < prior.loc["A"]
    assert posterior.loc["B"] > prior.loc["B"]


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ((0.0, 0.0, 1.0, 1.0, 0), "n_obs"),
        ((0.0, -1.0, 1.0, 1.0, 1), "prior_std"),
        ((0.0, 1.0, 1.0, -1.0, 1), "sample_std"),
    ],
)
def test_bayes_update_rejects_invalid_parameters(args, message):
    with pytest.raises(ValueError, match=message):
        bayes_update_normal(*args)


@pytest.mark.parametrize("confidence", [0.0, 1.0, -0.1, float("nan")])
def test_credible_interval_rejects_invalid_confidence(confidence):
    with pytest.raises(ValueError, match="confidence_level"):
        calculate_credible_interval(0.0, 1.0, confidence)
