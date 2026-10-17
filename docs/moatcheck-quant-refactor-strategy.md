# MoatCheck Quant architecture and refactoring strategy

This document distinguishes the current architecture from future work. The current repo is the source of truth for implemented behavior; the [README](../README.md) describes product use, and the [platform audit](platform-audit-2026-10-17.md) records detailed verification and remaining limits.

## Architecture principles

- Keep quantitative functions pure. Pass aligned prices, returns, factors, weights, and constraints explicitly. Keep database access, HTTP, environment variables, clocks, and file reads outside `backend/app/core/`.
- Keep provider fetching and persistence in `backend/app/data/`, `backend/app/services/`, and repositories. API adapters validate request contracts and format results.
- Preserve assumptions in response metadata: sampling cadence, annualization, risk-free rate, price and return conventions, cost assumptions, constraints, and data timestamps.
- Validate point-in-time inputs before historical selection. Use chronological splits for time series and avoid fitting scalers or tuning parameters on holdout observations.
- Keep owner identity at the application boundary. Derive authorization from verified Supabase identity and explicitly scope repository operations because the backend's elevated key bypasses RLS.
- Version model changes and explain changed results rather than silently changing historical semantics.

## Current repository structure

```text
frontend/
  app/                             Dashboard, screener, stock, portfolio, advisor, risk, backtest
  components/                      Shared shell, charts, advisor panels, explainability UI
  lib/                             API client, portfolio context, local quantitative helpers
backend/app/
  core/                            Pure research, portfolio, risk, trading, backtest, profiling math
  api/v1/                          Product and authenticated persistence endpoints
  api/v2/                          Explicit-input research/quant endpoint set
  advisor/                         Medium/long horizon service and email orchestration
  data/                            Legacy company-data collection clients and batching
  services/                        Universe and market-data pipeline orchestration
  repositories/, models/, db/      Supabase persistence layer
  qualitative/                     SEC/RSS collection and optional classification
backend/tests/                     Core, API, DB, advisor, and service tests
```

`backend/app/main.py` mounts the legacy `/api` routes plus `/api/v1` and `/api/v2`. It also serves the statically exported Next.js frontend from `backend/static/` when that build artifact is present.

## Current API surfaces

### Product API v1

- `POST /api/v1/portfolio/optimize`, Markowitz endpoints, risk parity, and saved portfolio configs.
- `POST /api/v1/risk/evaluate`, VaR, liquidity-adjusted VaR, stress replay, reverse stress, and macro shock routes.
- `POST /api/v1/backtest/run`, cost estimates, rebalance simulations, and saved backtest records.
- `/api/v1/advisor/...` for preview, saved recommendations, latest result, acknowledgment, and tracking.
- `/api/v1/session` for Supabase token exchange into the API cookie.
- `/api/v1/privacy/data` for owner-scoped persisted-data deletion.
- `/api/v1/screener` for filters and cursor-based pages; `/api/v1/universe/ingest` for authorized on-demand symbol ingestion.

### Explicit-input API v2

- `POST /api/v2/researcher/factor-score`, `/researcher/optimize`, `/researcher/walk-forward`
- `POST /api/v2/quant_dev/benchmark`, `GET /api/v2/quant_dev/capabilities`
- `POST /api/v2/trader/rebalance`
- `POST /api/v2/risk/var-cvar`, `/risk/stress-test`, `/risk/contribution`

These contracts accept caller-supplied arrays/models and are separate from the stored portfolio workflows in v1.

## Quant module map

- `core/portfolio`: log/simple return conversion, Markowitz, risk parity, Black-Litterman.
- `core/research`: factor cleaning, CAPM/Fama-French, IC, PCA, mean reversion, Hurst, regularization, Bayesian updating, point-in-time guards, CLT.
- `core/risk`: VaR/CVaR, drawdown, tail attribution, stationarity, GARCH, Kalman, GBM, EVT, VaR backtesting, HMM, copulas, crisis/stress, jump diffusion.
- `core/trading`: Kelly sizing, execution-cost estimates, rebalancing, liquidity-adjusted VaR, cointegration.
- `core/backtest` and `core/profiling`: walk-forward utilities, benchmarks, memory layout, Monte Carlo, concurrency, optional extension interface, and stochastic processes.

The full pure-core isolation and test review is documented in the platform audit. Static historical crisis values are embedded for runtime isolation and checked against the canonical CSV/manifest by `backend/scripts/embed_crisis_snapshots.py --check`.

## Current product/model boundaries

- Existing user-facing analytics rely heavily on weekly stored prices. A separate daily pipeline stores data in separate tables and does not automatically change existing page inputs.
- Optimizer expected returns and covariance are historical estimates; estimates can vary sharply with sample window. The optimizer's automatic result is not a forward return promise.
- The Advisor constrains horizon, turnover, and position sizing. Recommendations are human-reviewed scenarios and do not create broker orders.
- Risk Lab crisis replay uses benchmark series and disclosed ETF proxies, not selected-holding histories.
- The Backtest page uses a fixed-target monthly experiment, explicit starting capital, and assumed costs. It is not a point-in-time strategy certification or brokerage replay.
- Holdings track quantity and average buy price but do not include broker sync, a complete transaction ledger, realized PnL, taxes, or FX conversion.
- Black-Litterman and risk-parity math exists in the core. Black-Litterman has no user view-entry workflow, and risk parity is not a dedicated Optimizer page control.
- The HMM's smoothed state probabilities use the full supplied sequence. They are retrospective diagnostics, not causal earlier-date trading signals.

## Refactoring priorities

1. **Data provenance:** carry provider, quote currency, adjustment convention, observation time, and freshness through every stored-data response.
2. **Point-in-time fundamentals:** replace estimated reporting lags with actual filing acceptance timestamps where licensing and source data permit.
3. **Research-safe analytics:** keep chronological model selection, documented lookback windows, survivorship-aware universes, and reproducible fixtures.
4. **Shared contracts:** reduce drift between Pydantic and TypeScript using generated OpenAPI types or a contract parity check.
5. **Durable work:** move long data ingestion and expensive backtests into tracked jobs with idempotency, status, retry policy, and persisted artifacts.
6. **Database scale:** inspect query plans and paginate all broad legacy reads; measure actual universe size and response/query cost before promising large-universe performance.
7. **UI completeness:** add missing account-value dashboard semantics, dedicated Black-Litterman view controls, and educational explanations for remaining metrics. Saved target configurations can now be edited in place.
8. **Security/operations:** use shared rate limiting when horizontally scaled, verify live Supabase grants/RLS, and run browser-driven account-isolation and accessibility checks.
9. **Model governance:** record model version, assumptions, dataset identifiers, and estimator diagnostics with persisted recommendations and backtests.

## Verification policy

Run backend tests with warnings treated as errors, frontend typecheck and production build, numerical fixtures, and the crisis-data `--check` when changing quantitative or data contracts. Offline test results do not prove provider availability, Supabase production configuration, email delivery, browser accessibility, or data licensing. Consult the current verification evidence and exact limitations in the [platform audit](platform-audit-2026-10-17.md).
