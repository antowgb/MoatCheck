---
title: MoatCheck Quant Learning Roadmap
category: quant-finance
tags:
  - quant-finance
  - implementation-status
  - learning
created: 2026-10-14
updated: 2026-10-17
---

# MoatCheck Quant Learning Roadmap

This checklist tracks the quantitative concepts implemented in the repository. The original learning sequence has been completed through Phase 5. It is not a product-launch checklist or a claim that every method has complete production or browser coverage.

For the current feature inventory and setup, start with the [README](README.md). For detailed test evidence and outstanding verification, see the [platform audit](docs/platform-audit-2026-10-17.md). For code boundaries and future engineering priorities, see the [refactoring strategy](docs/moatcheck-quant-refactor-strategy.md).

## Phase 1: Foundations and returns

- [x] Log returns and simple returns
- [x] Factor winsorization and Z-score standardization
- [x] Sharpe ratio, CAGR, and annualized volatility
- [x] Markowitz efficient frontier
- [x] CAPM beta and alpha
- [x] Information coefficient and factor predictability
- [x] Point-in-time and survivorship-bias guards
- [x] CLT aggregation and normality diagnostics

## Phase 2: Factor models and allocation

- [x] Fama-French 3-factor and 5-factor regressions
- [x] Risk parity
- [x] Black-Litterman posterior and allocation math
- [x] PCA and covariance denoising
- [x] Mean-reversion and Ornstein-Uhlenbeck estimation
- [x] Hurst exponent and regime classification
- [x] Regularized alpha models with chronological validation
- [x] Gaussian Bayesian updating

## Phase 3: Risk metrics and stress testing

- [x] Parametric, historical, and Cornish-Fisher VaR with CVaR
- [x] Downside deviation, Sortino, and drawdown
- [x] Marginal/component VaR and tail attribution
- [x] ADF stationarity and cointegration tests
- [x] GARCH volatility modeling
- [x] Dynamic beta with a causal Kalman filter
- [x] Geometric Brownian Motion simulation
- [x] Peak-over-threshold EVT
- [x] Kupiec and Christoffersen VaR backtests
- [x] Hidden Markov regime models with fallback
- [x] Gaussian and Clayton copulas
- [x] Historical benchmark replay and reverse stress testing
- [x] Poisson jump and Merton jump-diffusion simulation

## Phase 4: Execution, rebalancing, and costs

- [x] Kelly position sizing with fractional and leverage caps
- [x] Spread, commission, and square-root impact estimates
- [x] Portfolio turnover and rebalancing simulations
- [x] Liquidity-adjusted VaR
- [x] Cointegration spread and statistical-arbitrage simulation

## Phase 5: Architecture and profiling

- [x] Pure-core architecture and array contracts
- [x] Empirical complexity benchmarks
- [x] C/F memory layout and vectorization helpers
- [x] Monte Carlo simulation and variance reduction
- [x] Thread/process concurrency benchmarks
- [x] Optional native-extension interface with NumPy fallback
- [x] Markov chain and martingale diagnostics

## Product and application work beyond the original quant roadmap

- [x] FastAPI adapters, typed request/response schemas, and error mapping
- [x] Supabase persistence, owner-scoped repositories, and privacy controls
- [x] Authenticated portfolio tracking and the medium/long-term Advisor
- [x] Screener pagination, dynamic universe ingestion, and scheduled data pipeline
- [x] Next.js dashboard, Optimizer, Risk Lab, Backtest, Advisor, and explainability UI
- [x] Security headers, request protections, offline quality workflow, and audit report

## Current engineering priorities

The completed concept list does not mean all product/operations work is complete. The next priorities are:

1. Verify deployed Supabase migrations, RLS policies, service-only RPC grants, and owner isolation using disposable accounts.
2. Run browser-based signup, confirmation, password reset, multi-tab sign-in, mobile layout, and accessibility checks.
3. Confirm live daily-data provider coverage, timestamps, on-demand fetches, scheduled job completion, and storage/display rights.
4. Improve data provenance and total-return coverage; existing screens still use weekly prices in many calculations.
5. Add a Black-Litterman user view workflow and a true account-value dashboard if these are product requirements. Target-portfolio configurations now support create, edit, and delete.
6. Add durable background jobs and shared rate limiting before multi-instance scaling.

Track evidence and limitations in the platform audit rather than treating this file as a sign-off certificate.
