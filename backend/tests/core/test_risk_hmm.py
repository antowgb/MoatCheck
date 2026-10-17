import numpy as np
import pandas as pd
import pytest
import app.core.risk.hmm as hmm_module

from app.core.risk.hmm import (
    GaussianHMM,
    classify_current_regime,
    fit_market_regime_hmm,
    hmm_telemetry,
    predict_regime_probabilities,
)


def _two_regime_returns(seed=5, n=2500):
    rng = np.random.default_rng(seed)
    transition = np.array([[0.98, 0.02], [0.06, 0.94]])
    states = np.zeros(n, dtype=int)
    for t in range(1, n):
        states[t] = rng.choice(2, p=transition[states[t - 1]])
    means = np.array([0.0005, -0.0015])
    volatilities = np.array([0.004, 0.035])
    returns = rng.normal(means[states], volatilities[states])
    return returns, states


def test_hmm_separates_low_and_high_volatility_regimes():
    values, true_states = _two_regime_returns()
    fitted = fit_market_regime_hmm(pd.Series(values), n_components=2, n_iter=100)
    variances = fitted["state_variances"]
    assert variances.shape == (2,)
    assert np.max(variances) > np.min(variances) * 5
    probabilities = predict_regime_probabilities(values, fitted)
    high_state = fitted["high_volatility_state"]
    predicted_states = probabilities.to_numpy().argmax(axis=1)
    high_prediction = predicted_states == high_state
    true_high = true_states == 1
    # Compare classification after removing EM start-up observations; state
    # labels are identified by fitted variance, not a fixed state number.
    accuracy = np.mean(high_prediction[100:] == true_high[100:])
    assert accuracy > 0.80


def test_optional_hmmlearn_and_numpy_fallback_paths_are_runtime_compatible():
    values, _ = _two_regime_returns(seed=6, n=300)
    fitted = fit_market_regime_hmm(values, n_components=2, n_iter=30)
    if GaussianHMM is None:
        assert fitted["model"].__class__.__name__ == "_NumpyGaussianHMM"
    else:
        assert isinstance(fitted["model"], GaussianHMM)
    probabilities = predict_regime_probabilities(values, fitted)
    assert probabilities.shape == (300, 2)
    assert np.isfinite(probabilities.to_numpy()).all()


def test_native_hmmlearn_path_is_exercised_with_a_controlled_gaussian_hmm(monkeypatch):
    class Monitor:
        converged = True
        iter = 3

    class FakeGaussianHMM:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.n_components = kwargs["n_components"]

        def fit(self, observations):
            assert observations.ndim == 2 and observations.shape[1] == 1
            self.means_ = np.array([[-0.01], [0.01]])
            self.covars_ = np.array([[[0.0001]], [[0.001]]])
            self.startprob_ = np.array([0.5, 0.5])
            self.transmat_ = np.array([[0.9, 0.1], [0.2, 0.8]])
            self.monitor_ = Monitor()
            return self

        def score(self, observations):
            return -12.5

        def predict_proba(self, observations):
            return np.tile([0.25, 0.75], (len(observations), 1))

    monkeypatch.setattr(hmm_module, "GaussianHMM", FakeGaussianHMM)
    values = np.linspace(-0.04, 0.04, 80)
    fitted = fit_market_regime_hmm(values, n_components=2, n_iter=15)

    assert isinstance(fitted["model"], FakeGaussianHMM)
    assert fitted["model"].kwargs["implementation"] == "log"
    assert np.allclose(fitted["transition_matrix"].sum(axis=1), 1.0)
    probabilities = predict_regime_probabilities(values, fitted)
    assert probabilities.shape == (80, 2)
    assert np.allclose(probabilities.iloc[0].to_numpy(), [0.25, 0.75])


def test_numpy_fallback_path_is_exercised_even_when_hmmlearn_is_installed(monkeypatch):
    monkeypatch.setattr(hmm_module, "GaussianHMM", None)
    values, _ = _two_regime_returns(seed=44, n=250)
    fitted = fit_market_regime_hmm(values, n_components=2, n_iter=20)

    assert isinstance(fitted["model"], hmm_module._NumpyGaussianHMM)
    assert np.allclose(fitted["transition_matrix"].sum(axis=1), 1.0)
    probabilities = predict_regime_probabilities(values, fitted)
    assert probabilities.shape == (250, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_transition_matrix_and_smoothed_posteriors_sum_to_one():
    values, _ = _two_regime_returns(seed=12, n=1000)
    fitted = fit_market_regime_hmm(values, n_components=2, n_iter=75)
    assert np.allclose(fitted["transition_matrix"].sum(axis=1), 1.0)
    probabilities = predict_regime_probabilities(values, fitted)
    assert probabilities.shape == (1000, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert np.all((probabilities.to_numpy() >= 0) & (probabilities.to_numpy() <= 1))


def test_current_regime_reports_probability_and_expected_duration():
    values, _ = _two_regime_returns(seed=20, n=500)
    fitted = fit_market_regime_hmm(values)
    current = classify_current_regime(values, fitted)
    assert 0 <= current["current_state"] < 2
    assert 0.0 <= current["current_state_probability"] <= 1.0
    assert 0.0 <= current["high_volatility_probability"] <= 1.0
    assert current["expected_regime_duration"] >= 1.0


def test_short_series_is_rejected_gracefully():
    with pytest.raises(ValueError, match="at least"):
        fit_market_regime_hmm(pd.Series([0.01, -0.02, 0.005, 0.0]))


def test_nan_observations_are_dropped_and_series_index_is_retained():
    values, _ = _two_regime_returns(seed=22, n=300)
    series = pd.Series(values, index=pd.date_range("2022-01-01", periods=300))
    series.iloc[10] = np.nan
    fitted = fit_market_regime_hmm(series)
    probabilities = predict_regime_probabilities(series, fitted)
    assert len(probabilities) == 299
    assert probabilities.index.equals(series.dropna().index)


def test_telemetry_reports_shape_state_parameters_and_probability_vector():
    values, _ = _two_regime_returns(seed=30, n=350)
    fitted = fit_market_regime_hmm(values)
    telemetry = hmm_telemetry(values, fitted)
    assert telemetry["input_shape"] == [350]
    assert len(telemetry["state_means"]) == 2
    assert len(telemetry["current_state_probability_vector"]) == 2
    assert sum(telemetry["current_state_probability_vector"]) == pytest.approx(1.0)
    assert telemetry["latency_ms"] >= 0


@pytest.mark.parametrize("kwargs", [{"n_components": 1}, {"n_iter": 0}])
def test_invalid_hmm_configuration_raises(kwargs):
    values, _ = _two_regime_returns(n=100)
    with pytest.raises(ValueError):
        fit_market_regime_hmm(values, **kwargs)
