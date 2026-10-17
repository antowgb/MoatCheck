"""Database-independent portfolio risk metrics."""

from app.core.risk.downside import (
    calculate_downside_deviation,
    calculate_drawdown_series,
    calculate_max_drawdown,
    calculate_sortino_ratio,
    downside_risk_telemetry,
)
from app.core.risk.tail_attribution import (
    calculate_component_var,
    calculate_marginal_var_contribution,
    calculate_tail_cvar_attribution,
    tail_attribution_telemetry,
)
from app.core.risk.stationarity import (
    make_series_stationary,
    stationarity_telemetry,
    test_augmented_dickey_fuller,
    test_cointegration_pair,
)
from app.core.risk.garch import (
    fit_garch11,
    forecast_future_volatility,
    garch_telemetry,
    predict_conditional_volatility,
)
from app.core.risk.kalman import (
    kalman_beta_telemetry,
    kalman_predict_step,
    kalman_update_step,
    track_dynamic_beta,
)
from app.core.risk.gbm import (
    calculate_breach_probability,
    calculate_path_terminal_stats,
    gbm_telemetry,
    simulate_gbm_paths,
)
from app.core.risk.evt import (
    calculate_evt_cvar,
    calculate_evt_var,
    evt_telemetry,
    fit_gpd_pot,
)
from app.core.risk.backtesting import (
    christoffersen_independence_test,
    evaluate_var_model_backtest,
    kupiec_pof_test,
    var_backtest_telemetry,
)
from app.core.risk.hmm import (
    classify_current_regime,
    fit_market_regime_hmm,
    hmm_telemetry,
    predict_regime_probabilities,
)
from app.core.risk.copulas import (
    copula_telemetry,
    fit_clayton_copula,
    fit_gaussian_copula,
    simulate_copula_joint_returns,
)
from app.core.risk.stress_testing import (
    apply_macro_scenario_shock,
    replay_historical_crisis,
    run_reverse_stress_test,
    stress_testing_telemetry,
)
from app.core.risk.jump_diffusion import (
    calculate_jump_impact_diagnostics,
    jump_diffusion_telemetry,
    simulate_merton_jump_diffusion,
    simulate_poisson_jumps,
)
from app.core.risk.metrics import (
    as_loss_amount,
    as_return_quantile,
    calculate_cornish_fisher_var,
    calculate_cvar_expected_shortfall,
    calculate_historical_var,
    calculate_parametric_var,
    marginal_contribution_to_risk,
    portfolio_var_cvar,
    risk_metrics_telemetry,
    stress_test,
)

__all__ = [
    "calculate_jump_impact_diagnostics",
    "jump_diffusion_telemetry",
    "simulate_merton_jump_diffusion",
    "simulate_poisson_jumps",
    "apply_macro_scenario_shock",
    "replay_historical_crisis",
    "run_reverse_stress_test",
    "stress_testing_telemetry",
    "copula_telemetry",
    "fit_clayton_copula",
    "fit_gaussian_copula",
    "simulate_copula_joint_returns",
    "classify_current_regime",
    "fit_market_regime_hmm",
    "hmm_telemetry",
    "predict_regime_probabilities",
    "christoffersen_independence_test",
    "evaluate_var_model_backtest",
    "kupiec_pof_test",
    "var_backtest_telemetry",
    "calculate_evt_cvar",
    "calculate_evt_var",
    "evt_telemetry",
    "fit_gpd_pot",
    "calculate_breach_probability",
    "calculate_path_terminal_stats",
    "gbm_telemetry",
    "simulate_gbm_paths",
    "kalman_beta_telemetry",
    "kalman_predict_step",
    "kalman_update_step",
    "track_dynamic_beta",
    "fit_garch11",
    "forecast_future_volatility",
    "garch_telemetry",
    "predict_conditional_volatility",
    "make_series_stationary",
    "stationarity_telemetry",
    "test_augmented_dickey_fuller",
    "test_cointegration_pair",
    "calculate_component_var",
    "calculate_marginal_var_contribution",
    "calculate_tail_cvar_attribution",
    "tail_attribution_telemetry",
    "calculate_downside_deviation",
    "calculate_drawdown_series",
    "calculate_max_drawdown",
    "calculate_sortino_ratio",
    "downside_risk_telemetry",
    "calculate_cornish_fisher_var",
    "as_loss_amount",
    "as_return_quantile",
    "calculate_cvar_expected_shortfall",
    "calculate_historical_var",
    "calculate_parametric_var",
    "marginal_contribution_to_risk",
    "portfolio_var_cvar",
    "risk_metrics_telemetry",
    "stress_test",
]
