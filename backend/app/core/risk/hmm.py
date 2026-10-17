"""Gaussian Hidden Markov Models for unsupervised market-regime analysis."""

from __future__ import annotations

import math
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
try:
    from hmmlearn.hmm import GaussianHMM
except ImportError:  # Offline/dev fallback; requirements.txt installs hmmlearn in normal deployments.
    GaussianHMM = None  # type: ignore[assignment,misc]


def _logsumexp(values: np.ndarray, axis: int | None = None) -> np.ndarray:
    """Stable NumPy log-sum-exp reduction; O(n) time in the reduced values."""
    maximum = np.max(values, axis=axis, keepdims=True)
    result = maximum + np.log(np.sum(np.exp(values - maximum), axis=axis, keepdims=True))
    return np.squeeze(result, axis=axis) if axis is not None else result.squeeze()


class _Monitor:
    """Small convergence record compatible with hmmlearn's monitor fields."""

    def __init__(self, converged: bool, iterations: int) -> None:
        self.converged = converged
        self.iter = iterations


class _NumpyGaussianHMM:
    """Minimal diagonal Gaussian HMM fallback implementing EM and smoothing."""

    def __init__(self, n_components: int, n_iter: int, min_covar: float, random_state: int) -> None:
        self.n_components = n_components
        self.n_iter = n_iter
        self.min_covar = min_covar
        self.random_state = random_state

    def _emissions(self, observations: np.ndarray) -> np.ndarray:
        """Compute Gaussian log-emission densities in O(TK) time/space."""
        values = observations[:, 0, None]
        means = self.means_[None, :, 0]
        variances = np.maximum(self.covars_[:, 0, 0], self.min_covar)[None, :]
        return -0.5 * (np.log(2.0 * np.pi * variances) + (values - means) ** 2 / variances)

    def _forward_backward(self, observations: np.ndarray) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
        """Compute posterior state probabilities and sequence likelihood in O(TK²)."""
        emission = self._emissions(observations)
        log_start = np.log(np.maximum(self.startprob_, 1e-300))
        log_transition = np.log(np.maximum(self.transmat_, 1e-300))
        n_time = len(observations)
        forward = np.empty((n_time, self.n_components))
        backward = np.zeros_like(forward)
        forward[0] = log_start + emission[0]
        for t in range(1, n_time):
            forward[t] = emission[t] + _logsumexp(forward[t - 1, :, None] + log_transition, axis=0)
        likelihood = float(_logsumexp(forward[-1]))
        for t in range(n_time - 2, -1, -1):
            backward[t] = _logsumexp(
                log_transition + emission[t + 1][None, :] + backward[t + 1][None, :], axis=1
            )
        posterior = np.exp(forward + backward - likelihood)
        posterior /= posterior.sum(axis=1, keepdims=True)
        return posterior, likelihood, forward, backward, emission

    def fit(self, observations: np.ndarray) -> "_NumpyGaussianHMM":
        """Fit transition/emission parameters by Baum-Welch EM; O(ITK²)."""
        x = observations[:, 0]
        self.startprob_ = np.full(self.n_components, 1.0 / self.n_components)
        self.transmat_ = np.full((self.n_components, self.n_components), 0.1 / max(self.n_components - 1, 1))
        np.fill_diagonal(self.transmat_, 0.9)
        quantiles = (np.arange(self.n_components) + 0.5) / self.n_components
        self.means_ = np.quantile(x, quantiles).reshape(-1, 1)
        overall_variance = max(float(np.var(x)), self.min_covar)
        self.covars_ = np.full((self.n_components, 1, 1), overall_variance)
        previous = -math.inf
        converged = False
        iterations = 0
        for iterations in range(1, self.n_iter + 1):
            posterior, likelihood, forward, backward, emission = self._forward_backward(observations)
            log_transition = np.log(np.maximum(self.transmat_, 1e-300))
            xi_sum = np.zeros_like(self.transmat_)
            for t in range(len(x) - 1):
                log_xi = (
                    forward[t, :, None]
                    + log_transition
                    + emission[t + 1][None, :]
                    + backward[t + 1][None, :]
                )
                weights = np.exp(log_xi - _logsumexp(log_xi))
                xi_sum += weights
            self.startprob_ = np.maximum(posterior[0], 1e-12)
            self.startprob_ /= self.startprob_.sum()
            row_totals = xi_sum.sum(axis=1, keepdims=True)
            self.transmat_ = np.divide(xi_sum, row_totals, out=self.transmat_.copy(), where=row_totals > 1e-12)
            totals = np.maximum(posterior.sum(axis=0), 1e-12)
            self.means_[:, 0] = (posterior * x[:, None]).sum(axis=0) / totals
            for state in range(self.n_components):
                residual = x - self.means_[state, 0]
                self.covars_[state, 0, 0] = max(
                    float(np.sum(posterior[:, state] * residual**2) / totals[state]), self.min_covar
                )
            if math.isfinite(previous) and abs(likelihood - previous) <= 1e-5 * (1.0 + abs(previous)):
                converged = True
                break
            previous = likelihood
        self.monitor_ = _Monitor(converged, iterations)
        return self

    def predict_proba(self, observations: np.ndarray) -> np.ndarray:
        """Return smoothed posteriors by forward-backward in O(TK²)."""
        return self._forward_backward(observations)[0]

    def score(self, observations: np.ndarray) -> float:
        """Return observation-sequence log-likelihood in O(TK²)."""
        return self._forward_backward(observations)[1]


def _observations(returns: pd.Series | np.ndarray, *, min_size: int = 1) -> tuple[np.ndarray, pd.Index]:
    """Validate finite scalar return observations and keep their index; O(T)."""
    if not isinstance(returns, (pd.Series, np.ndarray)):
        raise TypeError("returns must be a pandas Series or NumPy array")
    try:
        values = (returns.to_numpy(dtype=float, na_value=np.nan, copy=True)
                  if isinstance(returns, pd.Series) else np.asarray(returns, dtype=float))
    except (TypeError, ValueError) as exc:
        raise TypeError("returns must contain numeric values") from exc
    if values.ndim != 1:
        raise ValueError("returns must be one-dimensional")
    if np.isinf(values).any():
        raise ValueError("returns cannot contain infinite values")
    mask = np.isfinite(values)
    values = values[mask]
    if values.size < min_size:
        raise ValueError(f"at least {min_size} finite observations are required")
    original_index = returns.index if isinstance(returns, pd.Series) else pd.RangeIndex(len(mask))
    return values.reshape(-1, 1), original_index[mask]


def _model_from_result(hmm_result: dict[str, Any]) -> Any:
    """Extract either supported fitted HMM implementation in O(1).

    ``Any`` is intentional here: deployments may use ``hmmlearn.GaussianHMM``
    or the local NumPy fallback when that optional import is unavailable.
    """
    model = hmm_result.get("model") if isinstance(hmm_result, dict) else None
    valid_hmmlearn = GaussianHMM is not None and isinstance(model, GaussianHMM)
    valid_fallback = isinstance(model, _NumpyGaussianHMM)
    if not (valid_hmmlearn or valid_fallback):
        raise TypeError("hmm_result must be returned by fit_market_regime_hmm")
    return model


def fit_market_regime_hmm(
    returns: pd.Series | np.ndarray, n_components: int = 2, n_iter: int = 100
) -> dict[str, Any]:
    r"""Fit a Gaussian HMM by Baum-Welch to discover latent market regimes.

    Hidden states evolve with :math:`P(S_t=j\mid S_{t-1}=i)=a_{ij}` and
    emissions satisfy :math:`R_t\mid S_t=k\sim N(\mu_k,\sigma_k^2)`. The
    Baum-Welch EM E-step computes posterior state/transition probabilities;
    the M-step re-estimates :math:`A,\mu_k,\sigma_k^2` to increase the data
    likelihood. A two-state example can represent low-volatility bull and
    high-volatility bear regimes, with diagonal transition probabilities
    describing persistence. Complexity is roughly :math:`O(TK^2I)` time and
    :math:`O(TK)` working memory for :math:`T` dates, :math:`K` states, and
    :math:`I` EM iterations.
    """
    started = perf_counter()
    try:
        observations, index = _observations(returns)
    except (TypeError, ValueError):
        raise
    if isinstance(n_components, bool) or not isinstance(n_components, int) or n_components < 2:
        raise ValueError("n_components must be an integer of at least 2")
    if isinstance(n_iter, bool) or not isinstance(n_iter, int) or n_iter < 1:
        raise ValueError("n_iter must be a positive integer")
    minimum = max(10, 3 * n_components)
    if observations.shape[0] < minimum:
        raise ValueError(f"at least {minimum} observations are required to fit {n_components} regimes")
    if float(np.var(observations)) <= np.finfo(float).eps:
        raise ValueError("returns must have non-zero variance to identify regimes")

    if GaussianHMM is None:
        model = _NumpyGaussianHMM(n_components, n_iter, 1e-8, random_state=42)
    else:
        model = GaussianHMM(
            n_components=n_components,
            covariance_type="diag",
            n_iter=n_iter,
            tol=1e-4,
            min_covar=1e-8,
            random_state=42,
            implementation="log",
        )
    try:
        model.fit(observations)
    except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
        raise ValueError(f"Gaussian HMM fitting failed: {exc}") from exc
    state_means = np.asarray(model.means_, dtype=float).reshape(n_components)
    covariances = np.asarray(model.covars_, dtype=float)
    if covariances.ndim == 3:
        state_variances = covariances[:, 0, 0]
    elif covariances.ndim == 2:
        state_variances = covariances[:, 0]
    else:
        state_variances = covariances.reshape(n_components)
    high_vol_state = int(np.argmax(state_variances))
    return {
        "model": model,
        "transition_matrix": np.asarray(model.transmat_, dtype=float).copy(),
        "start_probabilities": np.asarray(model.startprob_, dtype=float).copy(),
        "state_means": state_means.copy(),
        "state_variances": state_variances.copy(),
        "state_volatilities": np.sqrt(np.maximum(state_variances, 0.0)),
        "high_volatility_state": high_vol_state,
        "n_components": int(n_components),
        "n_observations": int(observations.shape[0]),
        "input_shape": [int(observations.shape[0])],
        "log_likelihood": float(model.score(observations)),
        "converged": bool(model.monitor_.converged),
        "n_iterations": int(model.monitor_.iter),
        "latency_ms": float((perf_counter() - started) * 1000.0),
        "index": index,
    }


def predict_regime_probabilities(
    returns: pd.Series | np.ndarray, hmm_result: dict[str, Any]
) -> pd.DataFrame:
    r"""Calculate smoothed posterior regime probabilities for each observation.

    Returns :math:`P(S_t=k\mid R_1,\ldots,R_T)` using the HMM forward-backward
    algorithm (the posterior, not only the most-likely Viterbi path). For
    example, a large negative return may shift posterior mass from the
    low-volatility bull state toward the high-volatility bear state. Rows sum
    to one. Complexity: :math:`O(TK^2)` time and :math:`O(TK)` memory.
    """
    observations, index = _observations(returns)
    model = _model_from_result(hmm_result)
    probabilities = np.asarray(model.predict_proba(observations), dtype=float)
    columns = [f"state_{i}" for i in range(model.n_components)]
    return pd.DataFrame(probabilities, index=index, columns=columns)


def classify_current_regime(
    returns: pd.Series | np.ndarray, hmm_result: dict[str, Any]
) -> dict[str, Any]:
    r"""Classify the final observation and estimate high-volatility risk/duration.

    The current regime is :math:`\arg\max_k P(S_T=k\mid R_{1:T})`; probability
    of the high-volatility state is its posterior mass. Under a Markov chain,
    expected duration in state :math:`k` is :math:`1/(1-a_{kk})` observations
    (infinite for an absorbing state). For a low-vol bull/high-vol bear model,
    this exposes both the present state and how long the transition matrix
    expects it to persist. Complexity: :math:`O(TK^2)` time and
    :math:`O(TK)` memory.
    """
    probabilities = predict_regime_probabilities(returns, hmm_result)
    current = probabilities.iloc[-1].to_numpy(dtype=float)
    state = int(np.argmax(current))
    high_state = int(hmm_result["high_volatility_state"])
    transition = np.asarray(hmm_result["transition_matrix"], dtype=float)
    persistence = float(transition[state, state])
    expected_duration = math.inf if persistence >= 1.0 else 1.0 / (1.0 - persistence)
    return {
        "current_state": state,
        "current_state_probability": float(current[state]),
        "probability_vector": current.copy(),
        "high_volatility_state": high_state,
        "high_volatility_probability": float(current[high_state]),
        "expected_regime_duration": float(expected_duration),
        "expected_duration_periods": float(expected_duration),
        "state_mean": float(hmm_result["state_means"][state]),
        "state_variance": float(hmm_result["state_variances"][state]),
    }


def hmm_telemetry(
    returns: pd.Series | np.ndarray, hmm_result: dict[str, Any]
) -> dict[str, Any]:
    r"""Report HMM inference latency, state parameters, probability vector, and shape.

    The summary records current state posterior and fitted means/variances so
    low-volatility bull versus high-volatility bear classifications are
    explainable. Complexity: :math:`O(TK^2)` time and :math:`O(TK)` memory.
    """
    started = perf_counter()
    observations, _ = _observations(returns)
    probabilities = predict_regime_probabilities(returns, hmm_result)
    current = classify_current_regime(returns, hmm_result)
    return {
        "latency_ms": float((perf_counter() - started) * 1000.0),
        "input_shape": [int(observations.shape[0])],
        "n_components": int(hmm_result["n_components"]),
        "state_means": np.asarray(hmm_result["state_means"]).tolist(),
        "state_variances": np.asarray(hmm_result["state_variances"]).tolist(),
        "current_state_probability_vector": current["probability_vector"].tolist(),
        "current_state": current["current_state"],
        "high_volatility_probability": current["high_volatility_probability"],
        "converged": bool(hmm_result["converged"]),
        "posterior_shape": [int(probabilities.shape[0]), int(probabilities.shape[1])],
    }
