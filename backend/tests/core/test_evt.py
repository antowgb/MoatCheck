import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from app.core.risk.evt import (
    calculate_evt_cvar,
    calculate_evt_var,
    evt_telemetry,
    fit_gpd_pot,
)


def _heavy_tailed_losses(n=20_000, seed=81):
    rng = np.random.default_rng(seed)
    # Absolute Student-t losses are positive and retain a polynomial tail.
    return pd.Series(np.abs(rng.standard_t(df=3, size=n)) * 0.01)


def test_student_t_pot_fit_finds_positive_shape_parameter():
    fitted = fit_gpd_pot(_heavy_tailed_losses())
    assert fitted["xi"] > 0.0
    assert fitted["beta"] > 0.0
    assert fitted["exceedance_count"] >= 10
    assert fitted["input_shape"] == [20_000]
    assert np.isfinite(fitted["log_likelihood"])
    assert fitted["optimization_success"]


def test_evt_999_var_exceeds_gaussian_var_for_heavy_tailed_losses():
    losses = _heavy_tailed_losses()
    gaussian_var = losses.mean() + losses.std(ddof=1) * norm.ppf(0.999)
    evt_var = calculate_evt_var(losses, confidence_level=0.999, threshold_quantile=0.95)
    assert evt_var > gaussian_var


def test_evt_cvar_exceeds_var_and_telemetry_reports_tail_fit():
    losses = _heavy_tailed_losses(n=8000, seed=13)
    var = calculate_evt_var(losses, confidence_level=0.999, threshold_quantile=0.95)
    cvar = calculate_evt_cvar(losses, confidence_level=0.999, threshold_quantile=0.95)
    assert cvar >= var
    telemetry = evt_telemetry(losses, confidence_level=0.999)
    assert telemetry["input_shape"] == [8000]
    assert telemetry["exceedance_count"] > 0
    assert "xi" in telemetry and "latency_ms" in telemetry
    assert telemetry["evt_var"] == pytest.approx(var)


def test_threshold_with_too_few_exceedances_is_rejected():
    losses = np.arange(1000.0)
    with pytest.raises(ValueError, match="at least 10 threshold exceedances"):
        fit_gpd_pot(losses, threshold_quantile=0.999)


def test_confidence_level_must_be_above_threshold_quantile():
    with pytest.raises(ValueError, match="must exceed threshold_quantile"):
        calculate_evt_var(_heavy_tailed_losses(5000), confidence_level=0.95, threshold_quantile=0.95)


@pytest.mark.parametrize("bad", [[], [1.0, np.nan], [[1.0, 2.0]]])
def test_invalid_loss_inputs_raise(bad):
    with pytest.raises(ValueError):
        fit_gpd_pot(np.asarray(bad))
