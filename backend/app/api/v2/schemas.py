"""Request/response contracts for the MoatCheck Quant persona APIs."""

from datetime import date
from typing import Literal

from pydantic import Field
from app.schemas.common import StrictModel


class OptimizeRequest(StrictModel):
    returns: list[list[float]] = Field(min_length=2, description="Rows are periods, columns are assets")
    assets: list[str] = Field(min_length=2)
    objective: Literal["min_variance", "max_sharpe", "max_return"] = "max_sharpe"
    periods_per_year: int = Field(default=252, ge=1, le=366)
    risk_free_rate: float = 0.0
    target_return: float | None = None
    target_volatility: float | None = Field(default=None, gt=0)
    lower_bounds: list[float] | None = None
    upper_bounds: list[float] | None = None
    factor_exposures: list[list[float]] | None = None
    factor_tilts: list[float] | None = None
    linear_exposures: list[list[float]] | None = None
    linear_lower_bounds: list[float] | None = None
    linear_upper_bounds: list[float] | None = None


class FactorScoreRequest(StrictModel):
    factor_scores: dict[str, dict[str, float]] = Field(description="Factor -> asset -> normalized score")
    factor_weights: dict[str, float]


class WalkForwardRequest(StrictModel):
    returns: list[list[float]] = Field(min_length=3)
    scores: list[list[float]] = Field(description="At each period, scores available for every asset")
    assets: list[str] = Field(min_length=2)
    lookback_periods: int = Field(default=52, ge=1)
    rebalance_every: int = Field(default=13, ge=1)
    top_n: int = Field(default=5, ge=1)
    transaction_cost_bps: float = Field(default=10, ge=0, le=1000)


class BenchmarkRequest(StrictModel):
    returns: list[list[float]] = Field(min_length=2)
    assets: list[str] = Field(min_length=2)
    iterations: int = Field(default=1000, ge=10, le=100_000)
    periods_per_year: int = Field(default=252, ge=1, le=366)


class RebalanceRequest(StrictModel):
    current_weights: dict[str, float]
    target_weights: dict[str, float]
    prices: dict[str, float]
    average_daily_volume: dict[str, float] = Field(default_factory=dict, description="Share volume per day")
    portfolio_value: float = Field(gt=0)
    transaction_cost_bps: float = Field(default=5, ge=0, le=1000)
    volatility: dict[str, float] = Field(default_factory=dict, description="Daily decimal volatility per asset")
    integer_shares: bool = True


class RiskRequest(StrictModel):
    returns: list[list[float]] = Field(min_length=2)
    assets: list[str] = Field(min_length=1)
    weights: list[float]
    confidence: float = Field(default=0.95, gt=0, lt=1)
    method: Literal["historical", "parametric"] = "historical"


class StressRequest(StrictModel):
    returns: list[list[float]] = Field(min_length=2)
    dates: list[date]
    assets: list[str] = Field(min_length=1)
    weights: list[float]
    scenarios: dict[str, tuple[date, date]]


class ContributionRequest(StrictModel):
    returns: list[list[float]] = Field(min_length=2)
    assets: list[str] = Field(min_length=1)
    weights: list[float]
    periods_per_year: int = Field(default=252, ge=1, le=366)
