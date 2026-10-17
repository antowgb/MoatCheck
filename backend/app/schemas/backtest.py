"""Execution-cost and rebalancing API contracts."""

import numpy as np
from datetime import date
from pydantic import Field, model_validator

from app.schemas.common import APIEnvelope, StrictModel
from app.schemas.portfolio import _validate_tickers


class ExecutionCostsRequest(StrictModel):
    trade_sizes: list[float] = Field(min_length=1, max_length=500, description="Signed trade notionals in currency units")
    prices: list[float] = Field(min_length=1, max_length=500)
    advs: list[float] = Field(min_length=1, max_length=500, description="Average daily traded value")
    volatilities: list[float] = Field(min_length=1, max_length=500, description="Daily decimal volatility")
    spreads: list[float] = Field(min_length=1, max_length=500, description="Full bid/ask spread as a decimal")
    fixed_fee_per_trade: float = Field(default=1.0, ge=0)
    pct_broker_fee: float = Field(default=0.0005, ge=0)

    @model_validator(mode="after")
    def validate_contract(self) -> "ExecutionCostsRequest":
        sizes = (self.trade_sizes, self.prices, self.advs, self.volatilities, self.spreads)
        if len({len(x) for x in sizes}) != 1:
            raise ValueError("all order arrays must have matching lengths")
        if any(not np.isfinite(x).all() for x in sizes) or not np.isfinite([self.fixed_fee_per_trade, self.pct_broker_fee]).all():
            raise ValueError("order inputs and fees must be finite")
        if any(x <= 0 for x in self.prices + self.advs) or any(x < 0 for x in self.volatilities + self.spreads):
            raise ValueError("prices/ADVs must be positive; volatilities/spreads must be non-negative")
        return self


class RebalanceOrdersRequest(StrictModel):
    current_weights: dict[str, float]
    target_weights: dict[str, float]
    prices: dict[str, float]
    portfolio_value: float = Field(gt=0)
    average_daily_volume: dict[str, float] = Field(default_factory=dict)
    volatility: dict[str, float] = Field(default_factory=dict)
    transaction_cost_bps: float = Field(default=5, ge=0, le=10_000)
    integer_shares: bool = True

    @model_validator(mode="after")
    def validate_contract(self) -> "RebalanceOrdersRequest":
        for label, weights in (("current_weights", self.current_weights), ("target_weights", self.target_weights)):
            if weights and (not np.isfinite(list(weights.values())).all() or any(x < 0 for x in weights.values())):
                raise ValueError(f"{label} must contain finite non-negative values")
        if not self.target_weights or not np.isclose(sum(self.target_weights.values()), 1.0, atol=1e-5):
            raise ValueError("target_weights must sum to 1")
        if self.current_weights and not np.isclose(sum(self.current_weights.values()), 1.0, atol=1e-5):
            raise ValueError("current_weights must sum to 1")
        return self


class RebalanceSimulationRequest(StrictModel):
    tickers: list[str] = Field(min_length=1, max_length=100)
    initial_weights: list[float] = Field(min_length=1, max_length=100)
    target_weights: list[list[float]] = Field(min_length=1, max_length=5_000)
    prices: list[list[float]] = Field(min_length=2, max_length=5_001)
    initial_wealth: float = Field(default=1_000_000, gt=0)
    execution_cost_bps: float = Field(default=5, ge=0, le=10_000)
    max_turnover: float | None = Field(default=None, ge=0, le=2)
    periods_per_year: int = Field(default=12, ge=1, le=366)

    @model_validator(mode="after")
    def validate_contract(self) -> "RebalanceSimulationRequest":
        _validate_tickers(self.tickers)
        n = len(self.tickers)
        targets, prices = np.asarray(self.target_weights, dtype=float), np.asarray(self.prices, dtype=float)
        if len(self.initial_weights) != n or targets.ndim != 2 or targets.shape[1] != n or prices.shape != (targets.shape[0] + 1, n):
            raise ValueError("weights and prices must have matching periods-by-ticker dimensions")
        if not np.isfinite(self.initial_weights).all() or not np.isclose(sum(self.initial_weights), 1.0, atol=1e-6):
            raise ValueError("initial_weights must be finite and sum to 1")
        if not np.isfinite(targets).all() or not np.allclose(targets.sum(axis=1), 1.0, atol=1e-6):
            raise ValueError("each target weight row must be finite and sum to 1")
        if not np.isfinite(prices).all() or np.any(prices <= 0):
            raise ValueError("prices must be finite and positive")
        return self


class StoredPortfolioBacktestRequest(StrictModel):
    """Dated prices used to backtest an owner-stored strategic allocation."""

    dates: list[date] = Field(min_length=2, max_length=5_001)
    prices: list[list[float]] = Field(min_length=2, max_length=5_001)
    initial_wealth: float = Field(default=1_000_000, gt=0)
    execution_cost_bps: float = Field(default=5, ge=0, le=10_000)
    max_turnover: float | None = Field(default=None, ge=0, le=2)

    @model_validator(mode="after")
    def validate_prices(self) -> "StoredPortfolioBacktestRequest":
        matrix = np.asarray(self.prices, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != len(self.dates):
            raise ValueError("prices must contain one row for each date")
        if len(set(self.dates)) != len(self.dates) or any(a >= b for a, b in zip(self.dates, self.dates[1:])):
            raise ValueError("dates must be unique and strictly increasing")
        if not np.isfinite(matrix).all() or np.any(matrix <= 0):
            raise ValueError("prices must be finite and positive")
        if not np.isfinite([self.initial_wealth, self.execution_cost_bps]).all():
            raise ValueError("wealth and execution cost must be finite")
        return self


class BacktestResponse(APIEnvelope):
    pass
