"""Persistence contracts for portfolio lots and long-term plan settings."""

from __future__ import annotations

from datetime import date, datetime, timezone
import math
from uuid import UUID

from pydantic import Field, field_validator

from app.models._base import PersistenceModel


class PortfolioHolding(PersistenceModel):
    """A quantity lot summarized at its weighted average purchase price."""

    portfolio_id: UUID
    owner_id: UUID | None = Field(default=None, exclude=True)
    ticker: str = Field(min_length=1, max_length=32)
    quantity: float = Field(gt=0)
    average_buy_price: float = Field(gt=0)
    currency: str = Field(default="EUR", min_length=3, max_length=3)
    acquired_at: date = Field(default_factory=date.today)

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, value: str) -> str:
        return value.strip().upper()

    @field_validator("currency")
    @classmethod
    def normalize_currency(cls, value: str) -> str:
        value = value.strip().upper()
        if len(value) != 3 or not value.isalpha():
            raise ValueError("currency must be a three-letter ISO currency code")
        return value

    @field_validator("quantity", "average_buy_price")
    @classmethod
    def finite_positive(cls, value: float) -> float:
        if not math.isfinite(value) or value <= 0:
            raise ValueError("quantity and average_buy_price must be finite and positive")
        return value


class PortfolioSettings(PersistenceModel):
    """Monthly contribution and forecast horizon for one portfolio."""

    portfolio_id: UUID
    owner_id: UUID | None = Field(default=None, exclude=True)
    monthly_dca_amount: float = Field(default=0.0, ge=0)
    target_horizon_years: int = Field(default=5, ge=1, le=20)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("monthly_dca_amount")
    @classmethod
    def finite_dca(cls, value: float) -> float:
        if not math.isfinite(value) or value < 0:
            raise ValueError("monthly_dca_amount must be finite and non-negative")
        return value
