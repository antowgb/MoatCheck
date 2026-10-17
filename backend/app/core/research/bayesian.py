"""Pure Gaussian Bayesian updating utilities for financial research."""

from __future__ import annotations

import math
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm


def _validate_normal_parameters(
    prior_mean: float,
    prior_std: float,
    sample_mean: float,
    sample_std: float,
    n_obs: int,
) -> tuple[float, float, float, float, int]:
    r"""Validate scalar Gaussian prior and observation parameters.

    The prior :math:`\theta\sim N(\mu_0,\sigma_0^2)` and sample mean
    :math:`\bar x\mid\theta\sim N(\theta,\sigma^2/n)` must have valid
    standard deviations and finite means. A flat prior is represented by
    :math:`\sigma_0=\infty`. Example: a 25% prior margin and 20% observation
    are valid finite expectations. Complexity: :math:`O(1)` time and space.
    """
    try:
        prior_mean = float(prior_mean)
        prior_std = float(prior_std)
        sample_mean = float(sample_mean)
        sample_std = float(sample_std)
    except (TypeError, ValueError) as exc:
        raise ValueError("means and standard deviations must be numeric") from exc
    if not math.isfinite(prior_mean) or not math.isfinite(sample_mean):
        raise ValueError("prior_mean and sample_mean must be finite")
    if math.isnan(prior_std) or prior_std < 0.0:
        raise ValueError("prior_std must be non-negative and may be infinity")
    if not math.isfinite(sample_std) or sample_std < 0.0:
        raise ValueError("sample_std must be finite and non-negative")
    if isinstance(n_obs, (bool, np.bool_)) or not isinstance(n_obs, (int, np.integer)) or n_obs < 1:
        raise ValueError("n_obs must be a positive integer")
    return prior_mean, prior_std, sample_mean, sample_std, int(n_obs)


def bayes_update_normal(
    prior_mean: float,
    prior_std: float,
    sample_mean: float,
    sample_std: float,
    n_obs: int = 1,
) -> tuple[float, float]:
    r"""Update a Gaussian prior mean with a Gaussian sample likelihood.

    For prior :math:`\theta\sim N(\mu_0,\sigma_0^2)` and observations with
    known standard deviation :math:`\sigma`, the posterior is
    :math:`\sigma_n^2=(\sigma_0^{-2}+n/\sigma^2)^{-1}` and
    :math:`\mu_n=\sigma_n^2(\mu_0/\sigma_0^2+n\bar x/\sigma^2)`. A prior
    margin estimate of 25% updated with a 20% quarterly report moves toward
    20%; with comparable prior and observation uncertainty it would be about
    22.5%, while a more precise prior could yield 24%. An infinite prior
    standard deviation is flat and returns the sample mean; zero observation
    noise makes the observation exact. Complexity: :math:`O(1)` time and
    space.
    """
    mu0, std0, xbar, std, count = _validate_normal_parameters(
        prior_mean, prior_std, sample_mean, sample_std, n_obs
    )
    if std == 0.0:
        if std0 == 0.0 and not math.isclose(mu0, xbar, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("inconsistent exact prior and zero-noise observation")
        return (xbar if std0 > 0.0 else mu0), 0.0
    if math.isinf(std0):
        return xbar, float(std / math.sqrt(count))
    if std0 == 0.0:
        return mu0, 0.0

    prior_precision = 1.0 / (std0 * std0)
    observation_precision = count / (std * std)
    posterior_variance = 1.0 / (prior_precision + observation_precision)
    posterior_mean = posterior_variance * (
        prior_precision * mu0 + observation_precision * xbar
    )
    return float(posterior_mean), float(math.sqrt(posterior_variance))


def _update_factor_series(
    prior_series: pd.Series,
    observation_series: pd.Series,
    observation_noise: float,
) -> tuple[pd.Series, dict[str, Any]]:
    r"""Align factors and apply one conjugate update per available asset.

    Each standardized factor prior is modeled as :math:`N(\mu_i,1)` and a
    reported value as :math:`N(\theta_i,\tau^2)`. Posterior means are written
    only where prior and observation are both present; the input index and
    name are retained. In a 20-factor example, each report nudges the prior
    signal according to its observation noise. Complexity: :math:`O(N)` time
    and space for ``N`` factor entries.
    """
    if not isinstance(prior_series, pd.Series) or not isinstance(observation_series, pd.Series):
        raise TypeError("prior_series and observation_series must be pandas Series")
    if not prior_series.index.is_unique or not observation_series.index.is_unique:
        raise ValueError("prior and observation indexes must be unique")
    try:
        noise = float(observation_noise)
    except (TypeError, ValueError) as exc:
        raise ValueError("observation_noise must be finite and non-negative") from exc
    if not math.isfinite(noise) or noise < 0.0:
        raise ValueError("observation_noise must be finite and non-negative")
    try:
        prior_values = prior_series.astype(float)
        observed_values = observation_series.astype(float)
    except (TypeError, ValueError) as exc:
        raise TypeError("factor series must contain numeric values") from exc
    observed = pd.concat(
        [prior_values.rename("prior"), observed_values.rename("observation")], axis=1, join="inner"
    ).dropna()
    if not np.isfinite(observed.to_numpy(dtype=float)).all():
        raise ValueError("factor values must be finite wherever present")

    posterior = prior_values.copy()
    prior_std = 1.0
    posterior_std = (
        0.0
        if noise == 0.0
        else float(math.sqrt((prior_std**2 * noise**2) / (prior_std**2 + noise**2)))
    )
    for index, row in observed.iterrows():
        updated_mean, _ = bayes_update_normal(
            float(row["prior"]), prior_std, float(row["observation"]), noise
        )
        posterior.loc[index] = updated_mean
    posterior.name = prior_series.name
    shifts = (observed["observation"] - observed["prior"]).abs()
    telemetry = {
        "n_updated": int(len(observed)),
        "mean_absolute_shift": float((posterior.loc[observed.index] - observed["prior"]).abs().mean())
        if len(observed)
        else 0.0,
        "prior_std": prior_std,
        "posterior_std": posterior_std,
        "uncertainty_contraction_ratio": float(posterior_std / prior_std),
        "uncertainty_reduction_ratio": float(1.0 - posterior_std / prior_std),
        "mean_absolute_observation_gap": float(shifts.mean()) if len(shifts) else 0.0,
    }
    return posterior, telemetry


def update_factor_prior_series(
    prior_series: pd.Series,
    observation_series: pd.Series,
    observation_noise: float = 0.05,
) -> pd.Series:
    r"""Bayesian-update cross-sectional factor priors from new observations.

    For each aligned asset, the prior is :math:`\theta_i\sim N(p_i,1)` and
    the new statement-derived observation has known noise standard deviation
    ``observation_noise``. The resulting mean is a precision-weighted blend:
    :math:`\mu_i=(p_i+\bar x_i/\tau^2)/(1+1/\tau^2)` for one observation.
    For example, a 25% prior margin updated by a 20% observation moves toward
    20%, with the amount determined by reporting noise; setting noise to zero
    adopts the observation exactly. Complexity: :math:`O(N)` time and space.
    """
    posterior, _ = _update_factor_series(prior_series, observation_series, observation_noise)
    return posterior


def calculate_credible_interval(
    posterior_mean: float,
    posterior_std: float,
    confidence_level: float = 0.95,
) -> tuple[float, float]:
    r"""Calculate the central Gaussian credible interval for an updated alpha.

    For a normal posterior, the central interval is
    :math:`[\mu-z_{(1+c)/2}\sigma,\mu+z_{(1+c)/2}\sigma]`. This is also the
    highest-density interval for a Gaussian. For example, a 2% posterior
    standard deviation gives a 95% interval about :math:`\mu\pm3.92\%`.
    Complexity: :math:`O(1)` time and space.
    """
    try:
        mean = float(posterior_mean)
        std = float(posterior_std)
        confidence = float(confidence_level)
    except (TypeError, ValueError) as exc:
        raise ValueError("posterior parameters and confidence must be numeric") from exc
    if not math.isfinite(mean):
        raise ValueError("posterior_mean must be finite")
    if not math.isfinite(std) or std < 0.0:
        raise ValueError("posterior_std must be finite and non-negative")
    if not math.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("confidence_level must be strictly between 0 and 1")
    tail_probability = (1.0 - confidence) / 2.0
    z_value = float(norm.ppf(1.0 - tail_probability))
    return float(mean - z_value * std), float(mean + z_value * std)


def update_factor_prior_with_telemetry(
    prior_series: pd.Series,
    observation_series: pd.Series,
    observation_noise: float = 0.05,
) -> tuple[pd.Series, dict[str, Any]]:
    r"""Update factor priors and report latency, shift, and uncertainty change.

    Telemetry includes the posterior-minus-prior factor shift, absolute shift
    summary, and uncertainty contraction
    :math:`1-\sigma_{post}/\sigma_{prior}`. A standardized prior variance of
    one and report noise :math:`\tau=0.05` contract standard deviation to about
    0.05, while zero-noise observations contract uncertainty fully. Complexity:
    :math:`O(N)` time and space for ``N`` aligned factors.
    """
    start = perf_counter()
    posterior, telemetry = _update_factor_series(
        prior_series, observation_series, observation_noise
    )
    aligned_prior = prior_series.reindex(posterior.index).astype(float)
    aligned_observation = observation_series.reindex(posterior.index).astype(float)
    valid = aligned_prior.notna() & aligned_observation.notna()
    shift = (posterior - aligned_prior).where(valid)
    telemetry.update(
        {
            "latency_ms": float((perf_counter() - start) * 1000.0),
            "mean_signed_shift": float(shift.mean(skipna=True)) if shift.notna().any() else 0.0,
            "max_absolute_shift": float(shift.abs().max(skipna=True)) if shift.notna().any() else 0.0,
            "shift_by_factor": shift.rename("posterior_minus_prior"),
        }
    )
    return posterior, telemetry
