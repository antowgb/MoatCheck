"""Portfolio optimization request and response contracts."""

from typing import Any

import numpy as np
from pydantic import Field, model_validator

from app.schemas.common import APIEnvelope, StrictModel


def _validate_tickers(tickers: list[str]) -> None:
    if not tickers or len(set(tickers)) != len(tickers):
        raise ValueError("tickers must be non-empty and unique")
    if len(tickers) > 100:
        raise ValueError("at most 100 assets may be processed in one request")
    if any(not ticker or len(ticker) > 20 or not ticker.replace(".", "").replace("-", "").replace("_", "").isalnum() for ticker in tickers):
        raise ValueError("tickers must contain 1-20 alphanumeric, dot, dash, or underscore characters")


class PortfolioOptimizationRequest(StrictModel):
    tickers: list[str] = Field(min_length=1, max_length=100)
    expected_returns: list[float] = Field(min_length=1, max_length=100)
    covariance_matrix: list[list[float]] = Field(min_length=1, max_length=100)
    risk_free_rate: float = 0.03
    points: int = Field(default=20, ge=2, le=200)

    @model_validator(mode="after")
    def validate_contract(self) -> "PortfolioOptimizationRequest":
        _validate_tickers(self.tickers)
        n = len(self.tickers)
        if len(self.expected_returns) != n or len(self.covariance_matrix) != n:
            raise ValueError("ticker, expected return, and covariance dimensions must match")
        if any(len(row) != n for row in self.covariance_matrix):
            raise ValueError("covariance_matrix must be square and match tickers")
        if not np.isfinite(np.asarray(self.expected_returns + [self.risk_free_rate])).all():
            raise ValueError("returns and risk_free_rate must be finite")
        covariance = np.asarray(self.covariance_matrix, dtype=float)
        if not np.isfinite(covariance).all() or not np.allclose(covariance, covariance.T, atol=1e-10):
            raise ValueError("covariance_matrix must be finite and symmetric")
        return self


class RiskParityRequest(StrictModel):
    tickers: list[str] = Field(min_length=1, max_length=100)
    covariance_matrix: list[list[float]] = Field(min_length=1, max_length=100)
    target_risk_budget: list[float] | None = None

    @model_validator(mode="after")
    def validate_contract(self) -> "RiskParityRequest":
        _validate_tickers(self.tickers)
        n = len(self.tickers)
        matrix = np.asarray(self.covariance_matrix, dtype=float)
        if matrix.shape != (n, n) or not np.isfinite(matrix).all():
            raise ValueError("covariance_matrix must be finite, square, and match tickers")
        if not np.allclose(matrix, matrix.T, atol=1e-10):
            raise ValueError("covariance_matrix must be symmetric")
        if self.target_risk_budget is not None and len(self.target_risk_budget) != n:
            raise ValueError("target_risk_budget must have one value per ticker")
        if self.target_risk_budget is not None and (not np.isfinite(self.target_risk_budget).all() or min(self.target_risk_budget) <= 0):
            raise ValueError("target_risk_budget values must be finite and positive")
        return self


class PortfolioOptimizationResponse(APIEnvelope):
    pass


class RiskParityResponse(APIEnvelope):
    pass
