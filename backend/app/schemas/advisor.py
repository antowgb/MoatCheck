"""Strict contracts for medium- and long-horizon portfolio advice."""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal
from uuid import UUID

import numpy as np
from pydantic import AliasChoices, Field, field_validator, model_validator

from app.schemas.common import APIEnvelope, StrictModel
from app.schemas.portfolio import _validate_tickers


class AdvisorMarketSnapshot(StrictModel):
    """Dated simple-return observations and liquidity inputs for one universe.

    ``adv_values`` and prices are currency-denominated; spreads and returns
    are decimal fractions. The backend snapshots this payload's timestamp in
    every generated recommendation. A production market-data adapter can
    later replace caller-supplied values without changing the service contract.
    """

    as_of: datetime
    tickers: list[str] = Field(min_length=1, max_length=100)
    observation_dates: list[date] = Field(min_length=63, max_length=2_520)
    returns: list[list[float]] = Field(
        min_length=63,
        max_length=2_520,
        validation_alias=AliasChoices("returns", "daily_returns"),
    )
    prices: list[float] = Field(min_length=1, max_length=100)
    adv_values: list[float] = Field(min_length=1, max_length=100)
    spreads: list[float] = Field(min_length=1, max_length=100)
    price_currencies: list[str] | None = None

    @field_validator("observation_dates", mode="before")
    @classmethod
    def parse_iso_dates(cls, value: object) -> object:
        if isinstance(value, list):
            return [date.fromisoformat(item) if isinstance(item, str) else item for item in value]
        return value

    @field_validator("as_of", mode="before")
    @classmethod
    def parse_as_of(cls, value: object) -> object:
        if isinstance(value, str):
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        return value

    @model_validator(mode="after")
    def validate_snapshot(self) -> "AdvisorMarketSnapshot":
        _validate_tickers(self.tickers)
        if self.as_of.tzinfo is None or self.as_of.utcoffset() is None:
            raise ValueError("as_of must include a timezone offset")
        n_assets = len(self.tickers)
        matrix = np.asarray(self.returns, dtype=float)
        if matrix.ndim != 2 or matrix.shape != (len(self.observation_dates), n_assets):
            raise ValueError("returns must align with observation_dates and tickers")
        if not np.isfinite(matrix).all() or np.any(matrix <= -1.0):
            raise ValueError("returns must be finite simple returns greater than -100%")
        if len(set(self.observation_dates)) != len(self.observation_dates) or any(
            first >= second for first, second in zip(self.observation_dates, self.observation_dates[1:])
        ):
            raise ValueError("observation_dates must be unique and strictly increasing")
        if self.observation_dates[-1] > self.as_of.date():
            raise ValueError("observation_dates cannot be later than snapshot as_of")
        if any(len(values) != n_assets for values in (self.prices, self.adv_values, self.spreads)):
            raise ValueError("prices, adv_values, and spreads must match ticker count")
        if self.price_currencies is not None:
            if len(self.price_currencies) != n_assets or any(
                len(code) != 3 or not code.isalpha() for code in self.price_currencies
            ):
                raise ValueError("price_currencies must contain one three-letter currency code per ticker")
            self.price_currencies = [code.upper() for code in self.price_currencies]
        if not np.isfinite(self.prices).all() or min(self.prices) <= 0:
            raise ValueError("prices must be finite and positive")
        if not np.isfinite(self.adv_values).all() or min(self.adv_values) <= 0:
            raise ValueError("adv_values must be finite and positive")
        if not np.isfinite(self.spreads).all() or min(self.spreads) < 0:
            raise ValueError("spreads must be finite and non-negative")
        return self

    @property
    def daily_returns(self) -> list[list[float]]:
        """Backward-compatible alias for clients predating explicit cadence."""
        return self.returns


class AdvisorPolicy(StrictModel):
    """Explicit medium-term decision constraints and model preferences."""

    holding_period_days: int = Field(default=90, ge=30, le=365)
    minimum_holding_period_days: int = Field(default=30, ge=30, le=365)
    allocation_method: Literal["risk_parity", "max_sharpe", "minimum_variance"] = "risk_parity"
    risk_free_rate: float = Field(default=0.03, ge=-0.25, le=1.0)
    fractional_kelly: float = Field(default=0.20, ge=0.15, le=0.25)
    kelly_sizing_enabled: bool = True
    max_one_way_turnover: float = Field(default=0.10, ge=0.0, le=1.0)
    turnover_penalty_factor: float = Field(default=3.0, ge=0.0, le=20.0)
    confidence_level: float = Field(default=0.95, gt=0.5, lt=1.0)
    max_adv_participation: float = Field(default=0.10, gt=0.0, le=0.25)
    regime_lookback_months: int = Field(default=6, ge=3, le=12)
    capital: float = Field(default=1_000_000.0, gt=0.0)

    @model_validator(mode="after")
    def validate_horizon(self) -> "AdvisorPolicy":
        if self.minimum_holding_period_days > self.holding_period_days:
            raise ValueError("minimum_holding_period_days cannot exceed holding_period_days")
        if not np.isfinite([
            self.risk_free_rate, self.fractional_kelly, self.max_one_way_turnover,
            self.turnover_penalty_factor, self.confidence_level, self.max_adv_participation, self.capital,
        ]).all():
            raise ValueError("advisor policy numeric values must be finite")
        return self


class AdvisorAnalysisInput(StrictModel):
    """Market snapshot and medium/long-horizon policy used by an analysis."""
    market: AdvisorMarketSnapshot
    policy: AdvisorPolicy = Field(default_factory=AdvisorPolicy)


class AdvisorRecommendationCreateInput(AdvisorAnalysisInput):
    """Formal saved advice; email the account address after each saved result by default."""
    email_notification: bool = True


class AdvisorPreviewRequest(AdvisorAnalysisInput):
    """Preview either a saved portfolio or an inline allocation without saving."""

    portfolio_id: UUID | None = None
    portfolio_name: str | None = Field(default=None, min_length=1, max_length=120)
    tickers: list[str] | None = None
    current_weights: dict[str, float] | None = None
    monthly_dca_amount: float | None = Field(default=None, ge=0)
    target_horizon_years: int | None = Field(default=None, ge=1, le=20)

    @field_validator("portfolio_id", mode="before")
    @classmethod
    def parse_portfolio_uuid(cls, value: object) -> object:
        return UUID(value) if isinstance(value, str) else value

    @model_validator(mode="after")
    def validate_portfolio_source(self) -> "AdvisorPreviewRequest":
        if self.portfolio_id is not None and (self.tickers is not None or self.current_weights is not None):
            raise ValueError("provide portfolio_id or inline portfolio weights, not both")
        if self.portfolio_id is None:
            if self.tickers is None or self.current_weights is None:
                raise ValueError("inline preview requires both tickers and current_weights")
            _validate_tickers(self.tickers)
            if set(self.current_weights) != set(self.tickers):
                raise ValueError("current_weights must contain exactly one weight per ticker")
            values = np.asarray(list(self.current_weights.values()), dtype=float)
            if not np.isfinite(values).all() or np.any(values < 0) or not np.isclose(values.sum(), 1.0, atol=1e-6):
                raise ValueError("current_weights must be finite, non-negative, and sum to 1")
            if set(self.tickers) != set(self.market.tickers):
                raise ValueError("portfolio tickers must match the market snapshot universe")
        if self.monthly_dca_amount is not None and not np.isfinite(self.monthly_dca_amount):
            raise ValueError("monthly_dca_amount must be finite")
        return self


class PortfolioHoldingInput(StrictModel):
    """One positive quantity and purchase average for a portfolio asset."""
    ticker: str = Field(min_length=1, max_length=32)
    quantity: float = Field(gt=0)
    average_buy_price: float = Field(gt=0)
    currency: str = Field(default="EUR", min_length=3, max_length=3)
    acquired_at: date = Field(default_factory=date.today)

    @field_validator("currency", mode="before")
    @classmethod
    def default_blank_currency(cls, value: object) -> object:
        # Older clients can submit an empty select value. Match the declared
        # EUR default so those clients remain compatible during rollout.
        if isinstance(value, str) and not value.strip():
            return "EUR"
        return value

    @field_validator("ticker")
    @classmethod
    def clean_holding_ticker(cls, value: str) -> str:
        return value.strip().upper()

    @field_validator("quantity", "average_buy_price")
    @classmethod
    def finite_holding_value(cls, value: float) -> float:
        if not np.isfinite(value) or value <= 0:
            raise ValueError("holding quantity and average buy price must be finite and positive")
        return value

    @field_validator("currency")
    @classmethod
    def clean_currency(cls, value: str) -> str:
        value = value.strip().upper()
        if len(value) != 3 or not value.isalpha():
            raise ValueError("currency must be a three-letter ISO currency code")
        return value

    @field_validator("acquired_at", mode="before")
    @classmethod
    def parse_acquired_at(cls, value: object) -> object:
        # JSON has no date primitive; accept the standard date-only ISO form
        # while keeping strict typing for every other value.
        if isinstance(value, str):
            try:
                return date.fromisoformat(value)
            except ValueError as exc:
                raise ValueError("acquired_at must be an ISO date (YYYY-MM-DD)") from exc
        return value

    @field_validator("acquired_at")
    @classmethod
    def acquired_date_not_in_future(cls, value: date) -> date:
        if value > date.today():
            raise ValueError("acquired_at cannot be in the future")
        return value


class PortfolioHoldingsInput(StrictModel):
    """Complete holdings set; saving replaces the portfolio's current set."""
    holdings: list[PortfolioHoldingInput] = Field(max_length=100)

    @model_validator(mode="after")
    def unique_holding_tickers(self) -> "PortfolioHoldingsInput":
        tickers = [holding.ticker for holding in self.holdings]
        if len(set(tickers)) != len(tickers):
            raise ValueError("holding tickers must be unique")
        return self


class PortfolioTrackingSettingsInput(StrictModel):
    """Saved recurring contribution amount and projection horizon."""
    monthly_dca_amount: float = Field(ge=0)
    target_horizon_years: int = Field(default=5, ge=1, le=20)

    @field_validator("monthly_dca_amount")
    @classmethod
    def finite_monthly_amount(cls, value: float) -> float:
        if not np.isfinite(value) or value < 0:
            raise ValueError("monthly_dca_amount must be finite and non-negative")
        return value


class ApplyDiscoveryProposalInput(StrictModel):
    """Validated target structure selected by the owner from discovery results."""

    proposal_id: str = Field(min_length=1, max_length=128)
    tickers: list[str] = Field(min_length=1, max_length=100)
    target_weights: dict[str, float]

    @model_validator(mode="after")
    def validate_proposal_weights(self) -> "ApplyDiscoveryProposalInput":
        _validate_tickers(self.tickers)
        if set(self.tickers) != set(self.target_weights):
            raise ValueError("target_weights must match proposal tickers exactly")
        weights = np.asarray(list(self.target_weights.values()), dtype=float)
        if not np.isfinite(weights).all() or np.any(weights < 0) or not np.isclose(weights.sum(), 1.0, atol=1e-6):
            raise ValueError("proposal weights must be finite, non-negative, and sum to 1")
        return self


class PortfolioConfigCreate(StrictModel):
    """Owner-independent payload for creating a saved portfolio."""

    name: str = Field(min_length=1, max_length=120)
    tickers: list[str] = Field(min_length=1, max_length=100)
    target_weights: dict[str, float]
    risk_constraints: dict = Field(default_factory=dict)
    optimization_parameters: dict = Field(default_factory=dict)
    description: str | None = Field(default=None, max_length=2000)

    @model_validator(mode="after")
    def validate_config(self) -> "PortfolioConfigCreate":
        _validate_tickers(self.tickers)
        if set(self.tickers) != set(self.target_weights):
            raise ValueError("target_weights must match tickers exactly")
        weights = np.asarray(list(self.target_weights.values()), dtype=float)
        if not np.isfinite(weights).all() or np.any(weights < 0) or not np.isclose(weights.sum(), 1.0, atol=1e-6):
            raise ValueError("target_weights must be finite, non-negative, and sum to 1")
        return self


class AdvisorResponse(APIEnvelope):
    pass
