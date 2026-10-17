"""Risk, liquidity, and stress-test API contracts."""

from datetime import date
from typing import Literal

import numpy as np
from pydantic import Field, field_validator, model_validator

from app.schemas.common import APIEnvelope, StrictModel
from app.schemas.portfolio import _validate_tickers


class PortfolioReturnsRequest(StrictModel):
    tickers: list[str] = Field(min_length=1, max_length=100)
    weights: list[float] = Field(min_length=1, max_length=100)
    returns: list[list[float]] = Field(min_length=2, max_length=5_000)
    confidence_level: float = Field(default=0.95, gt=0, lt=1)
    method: Literal["historical", "parametric"] = "historical"

    @model_validator(mode="after")
    def validate_contract(self) -> "PortfolioReturnsRequest":
        _validate_tickers(self.tickers)
        if len(self.weights) != len(self.tickers):
            raise ValueError("weights must have one value per ticker")
        values = np.asarray(self.returns, dtype=float)
        if values.ndim != 2 or values.shape[1] != len(self.tickers) or not np.isfinite(values).all():
            raise ValueError("returns must be a finite observations-by-tickers matrix")
        if not np.isfinite(self.weights).all() or not np.isclose(sum(self.weights), 1.0, atol=1e-6):
            raise ValueError("weights must be finite and sum to 1")
        return self


class LiquidityVaRRequest(StrictModel):
    tickers: list[str] = Field(min_length=1, max_length=100)
    portfolio_returns: list[float] = Field(min_length=2, max_length=5_000)
    position_values: list[float] = Field(min_length=1, max_length=100)
    advs: list[float] = Field(min_length=1, max_length=100)
    volatilities: list[float] = Field(min_length=1, max_length=100)
    spreads: list[float] = Field(min_length=1, max_length=100)
    confidence_level: float = Field(default=0.99, gt=0, lt=1)
    max_participation_rate: float = Field(default=0.10, gt=0, le=1)
    method: Literal["historical", "parametric"] = "parametric"

    @model_validator(mode="after")
    def validate_contract(self) -> "LiquidityVaRRequest":
        _validate_tickers(self.tickers)
        n = len(self.tickers)
        if any(len(values) != n for values in (self.position_values, self.advs, self.volatilities, self.spreads)):
            raise ValueError("position, ADV, volatility, and spread arrays must match tickers")
        if not np.isfinite(self.portfolio_returns).all():
            raise ValueError("portfolio_returns must be finite")
        if not np.isfinite(self.position_values + self.volatilities + self.spreads).all():
            raise ValueError("position values, volatilities, and spreads must be finite")
        if any(x <= 0 for x in self.advs) or any(x < 0 for x in self.volatilities + self.spreads):
            raise ValueError("ADVs must be positive and volatility/spreads non-negative")
        return self


class HistoricalStressRequest(StrictModel):
    tickers: list[str] = Field(min_length=1, max_length=100)
    weights: list[float] = Field(min_length=1, max_length=100)
    dates: list[date] = Field(min_length=1, max_length=5_000)
    asset_returns: list[list[float]] = Field(min_length=1, max_length=5_000)
    scenario_start: date
    scenario_end: date

    @field_validator("dates", mode="before")
    @classmethod
    def parse_json_dates(cls, value: object) -> object:
        """Parse ISO date strings from JSON while keeping the rest strict."""
        if isinstance(value, list):
            return [date.fromisoformat(item) if isinstance(item, str) else item for item in value]
        return value

    @field_validator("scenario_start", "scenario_end", mode="before")
    @classmethod
    def parse_json_date(cls, value: object) -> object:
        """Parse an ISO calendar date supplied by JSON clients."""
        return date.fromisoformat(value) if isinstance(value, str) else value

    @model_validator(mode="after")
    def validate_contract(self) -> "HistoricalStressRequest":
        _validate_tickers(self.tickers)
        values = np.asarray(self.asset_returns, dtype=float)
        if len(self.weights) != len(self.tickers) or values.shape != (len(self.dates), len(self.tickers)):
            raise ValueError("weights and asset_returns dimensions must match dates and tickers")
        if len(set(self.dates)) != len(self.dates) or any(a >= b for a, b in zip(self.dates, self.dates[1:])):
            raise ValueError("dates must be unique and strictly increasing")
        if not np.isfinite(self.weights).all() or not np.isclose(sum(self.weights), 1.0, atol=1e-6):
            raise ValueError("weights must be finite and sum to 1")
        if not np.isfinite(values).all() or self.scenario_start > self.scenario_end:
            raise ValueError("asset returns must be finite and scenario dates ordered")
        return self


class HistoricalScenarioReplayRequest(StrictModel):
    """Optional benchmark portfolio weights for a packaged crisis replay."""

    weights: list[float] | None = None

    @model_validator(mode="after")
    def validate_weights(self) -> "HistoricalScenarioReplayRequest":
        if self.weights is not None:
            values = np.asarray(self.weights, dtype=float)
            if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
                raise ValueError("weights must be a non-empty finite vector")
            if np.any(values < 0) or not np.isclose(values.sum(), 1.0, atol=1e-6):
                raise ValueError("weights must be non-negative and sum to 1")
        return self


class ReverseStressRequest(StrictModel):
    tickers: list[str] = Field(min_length=1, max_length=100)
    weights: list[float] = Field(min_length=1, max_length=100)
    covariance_matrix: list[list[float]] = Field(min_length=1, max_length=100)
    factor_names: list[str] = Field(min_length=1, max_length=100)
    factor_loadings: list[list[float]] = Field(min_length=1, max_length=100)
    target_loss_threshold: float = Field(default=-0.30, lt=0)

    @model_validator(mode="after")
    def validate_contract(self) -> "ReverseStressRequest":
        _validate_tickers(self.tickers)
        if len(set(self.factor_names)) != len(self.factor_names) or not self.factor_names:
            raise ValueError("factor_names must be unique and non-empty")
        n, k = len(self.tickers), len(self.factor_names)
        cov, loadings = np.asarray(self.covariance_matrix, dtype=float), np.asarray(self.factor_loadings, dtype=float)
        if len(self.weights) != n or cov.shape != (n, n) or loadings.shape != (n, k):
            raise ValueError("weights, covariance, and factor loadings dimensions must match")
        if not all(np.isfinite(x).all() for x in (cov, loadings, self.weights)) or not np.isclose(sum(self.weights), 1.0, atol=1e-6):
            raise ValueError("all inputs must be finite and weights must sum to 1")
        if not np.allclose(cov, cov.T, atol=1e-10):
            raise ValueError("covariance_matrix must be symmetric")
        return self


class MacroShockRequest(StrictModel):
    tickers: list[str] = Field(min_length=1, max_length=100)
    weights: list[float] = Field(min_length=1, max_length=100)
    factor_names: list[str] = Field(min_length=1, max_length=100)
    factor_loadings: list[list[float]] = Field(min_length=1, max_length=100)
    macro_shocks: dict[str, float]

    @model_validator(mode="after")
    def validate_contract(self) -> "MacroShockRequest":
        _validate_tickers(self.tickers)
        matrix = np.asarray(self.factor_loadings, dtype=float)
        if len(self.weights) != len(self.tickers) or matrix.shape != (len(self.tickers), len(self.factor_names)):
            raise ValueError("factor loading dimensions must match tickers and factor_names")
        if not np.isfinite(matrix).all() or not np.isfinite(self.weights).all() or not np.isclose(sum(self.weights), 1.0, atol=1e-6):
            raise ValueError("loadings/weights must be finite and weights must sum to 1")
        if not set(self.macro_shocks).issubset(self.factor_names) or not np.isfinite(list(self.macro_shocks.values())).all():
            raise ValueError("macro_shocks must contain finite values for known factors")
        return self


class RiskResponse(APIEnvelope):
    pass
