# MoatCheck platform audit and regression report

Audit date: 17 October 2026. Scope: the backend, frontend, SQL contracts, data ingestion, and deployment configuration in this repository.

## Result and scope of assurance

The local regression gate passes after the corrections described below. This is a code review and offline integration audit, with a production frontend build. It is not a certification that every live service, browser interaction, or deployed database policy has been exercised.

The audit includes 16 application pages, 44 core compute modules, shared frontend components, API contracts, persistence repositories, and the existing test suite. Core modules have an automated import and direct data-I/O boundary check. Code inspection and tests were used together; passing tests alone were not treated as proof of correctness.

## Verification evidence

Commands from the repository root:

```bash
npm --prefix frontend run typecheck
npm --prefix frontend run build
(cd backend && .venv/bin/python -m pytest tests/ -W error)
(cd frontend && node lib/portfolio-math.test.mjs)
(cd frontend && node lib/alignedReturn.test.mjs)
python3 backend/scripts/embed_crisis_snapshots.py --check
```

- **666 backend tests pass, with 0 failures, 0 skips, and 0 emitted warnings**, including core, advisor, API, database, service, and legacy regression tests.
- Eight new frontend numerical regression tests pass. The existing cross-language aligned-return fixture also passes.
- TypeScript validation passes. Next.js compiles and exports the application pages.
- Backend tests run with warnings promoted to errors. No warning suppression remains in `pytest.ini`.
- A coverage run measures approximately 72% backend statement coverage. Core compute coverage is approximately 82%, API coverage approximately 62%, and application advisor coverage approximately 80%. These are statement measurements, not proof of branch, browser, statistical, or production-service coverage.
- The tests use mocked network/database boundaries and an in-process ASGI client. They do not write user data or send email. The ASGI adapter deliberately skips lifespan hooks; the current app does not define such hooks.

The test environment initially contained NumPy 2.5.1, despite `requirements.txt` requiring `numpy<2.5` for hmmlearn 0.3.3 compatibility. A strict run exposed six HMM failures hidden by the old warning filter. Installing the already declared compatible NumPy 2.4.6 and removing that filter resolved the issue. No vendor files or HMM algorithms were patched.

## Corrections made

| Area | Finding | Correction |
| --- | --- | --- |
| Portfolio return inputs | Risk Lab weighted log returns as if they were simple returns. | Shared frontend arithmetic contracts now supply simple returns for portfolio risk and arithmetic annualized moments for allocation. Log returns remain separately named. |
| Annualization | Mean calendar spacing could produce inconsistent inputs relative to the Advisor's exchange-session convention. | Shared cadence inference uses 252 for daily exchange observations and approximately 52 for weekly observations. |
| Advisor allocation | Expected return and covariance used different conventions from the Optimizer. | The Advisor uses arithmetic simple-return mean and sample covariance for allocation. Holding-period growth is estimated from the weighted portfolio's log growth, with an explicit cash yield. Untracked saved portfolios start from cash and use only the configured monthly contribution; projections start without assumed lump-sum capital. Model version is `advisor-medium-horizon-1.3`. |
| Optimizer cash overlay | Charts showed half weights but headline return and risk remained fully invested. | Expected return, volatility, and Sharpe now reflect the displayed cash allocation. Cash is explicitly assumed to earn 0% in this exploratory overlay. |
| Optimizer empty state | Enabling cash without a feasible allocation could display a lone 50% cash slice. | No allocation slice is generated without a selected feasible result. Undefined Sharpe is shown as unavailable. |
| Backtest drawdown | Maximum drawdown was recomputed against the largest future peak in the whole curve. | Drawdown uses the running peak, including starting capital. Calmar uses that corrected maximum drawdown. |
| Backtest ratios | Zero denominators displayed a misleading zero ratio. | Sharpe, Sortino, and Calmar display `N/A` when their denominator is zero. |
| Positive-loss VaR | The portfolio helper could return a negative loss value for a profitable tail. | Portfolio VaR and CVaR are floored at zero. Scalar signed-return APIs retain their documented convention. |
| JSON transport | NumPy arrays/scalars could bypass recursive normalization of non-finite values. | Nested non-finite metrics become JSON `null`, allowing consumers to show unavailable values. |
| Screener cursor | A valid base64 payload containing a list or scalar could cause a 500. Boolean sort values were also accepted as numbers. | Invalid cursor structures and values return 422 before querying the database. |
| Screener request ordering | A slow old filter request or pagination result could overwrite or append to a newer query. | Generation guards discard old responses. Load-more uses the last applied filter snapshot and deduplicates ticker rows. Sort defaults retain the selected ordering. |
| Comparison and stock detail | Slow responses could show the previously selected ticker or comparison inputs. | Stock details reset on ticker changes. Comparison requests are invalidated when inputs change. |
| Shared portfolio state | Overlapping portfolio refreshes could restore an older list after session changes. | Refresh generations invalidate stale responses. Logout clears portfolio state. Browser storage failures leave selection usable in memory. |
| Workspace state | An Advisor or chart state could survive a portfolio switch. | Page state is scoped to owner, portfolio ID, and saved version through the AppShell. A switch remounts the workspace without a browser reload. |
| Portfolio setup | The selected saved portfolio was not loaded into the setup form. | Its assets and saved weights now load automatically. Saving a separate portfolio is explicitly labelled as saving a copy. Quantity/cost-basis editing remains in the Advisor. |
| Auth lifecycle | Header handling omitted `SIGNED_IN` and silently ignored session exchange errors. | Sign-in and refresh events exchange the API cookie, errors are surfaced, and pending header exchanges are cancelled on sign-out or cleanup. |
| Network waiting | Frontend fetches had no explicit time limit. | API requests/session exchanges have a 30-second limit; health checks have a 10-second limit. Only reads retry. |
| Tracking and currency | Current liquidity comparison and trade notionals could use the default €1m capital instead of tracked value. Some cost labels always said EUR. | Comparisons use the recommendation's actual starting weights and tracked market value. Cost and order labels use the portfolio quote currency. |
| FX guardrail | Mixed quoted currencies could pass when no holdings or DCA had been entered. | Advisor calculations reject mixed quote currencies. Missing frontend currency metadata produces a clear data error. Single-currency untracked projections use the known quote currency. |
| Risk comparison | A failed current L-VaR calculation silently substituted a different-horizon approximation. | The missing comparison displays `N/A`. Holding-period VaR and liquidation-window L-VaR have distinct labels. |
| Advisor comparison colors | A rise in expected return was colored as worse. | The return row and risk rows use their respective favorable directions. |
| Projections | Fractional iteration counts could reach array constructors, and extreme inputs could overflow. | Integer counts and numerical range are checked. Projection overflow returns a validation error without emitting runtime warnings. |
| Core isolation | The crisis module opened JSON/CSV files at runtime. Returned metadata could also be mutated by callers. | Core scenarios now use embedded constants and return independent metadata/arrays. Tests verify CSV parity and source checksums. |
| Calculation drawer | The custom overlay did not confine keyboard focus and could inherit transformed card positioning. | A native modal dialog is rendered in a body portal, locks body scrolling, supports Escape, and restores focus. |
| Metric help | `details` elements nested inside paragraph labels created invalid HTML and could lead to hydration problems. Tooltips could be clipped by cards. | Inline button/span markup and a body portal provide keyboard access, outside dismissal, and unclipped content. |
| Navigation | The More menu lacked complete outside/Escape dismissal. | It closes on route changes, outside clicks, and Escape. |
| Visual consistency | Recharts tooltips had inconsistent light/default surfaces. | A shared dark tooltip surface uses the existing palette, rounded corners, and tabular figures. Additional valuation/projection tiles expose metric help. |

## Page and feature review

| Surface | Verified code path and behavior | Limits or remaining scope |
| --- | --- | --- |
| Login | Email/password sign-in, signup, reset request, session exchange, safe internal destination. | Delivery, callback allowlists, and real project credentials require a live Auth check. |
| Auth callback and reset | PKCE/token handling and password-reset page remain client boundaries. | Email opening on another device/tab is not covered by a real browser test. |
| Header | Shared portfolio selector, measured round-trip latency, health polling, server metadata, sign-in/out. | Additional telemetry is available in the status tooltip. A permanent global search field is not present; asset search lives in the relevant workspaces. |
| Dashboard | Active saved target, concentration/coverage cards, asset links, research table, onboarding, and loading/error states. | This is a saved-target overview. It is not an actual account-value dashboard or a realized-PnL ledger. |
| Screener | Database-side filters and keyset cursor contract, bounded pages, sort validation, selection/correlation, CSV export. | A real 3,000+ row deployment and PostgreSQL query plans were not measured. Legacy stock-list queries used by other pages still warrant pagination work. |
| Stock detail | Price history, company/fundamental/risk scores, qualitative events, missing-data handling, ETF branch. | Price and score availability depend on the latest successful provider refresh. A failed optional price/news request is not replaced with invented data. |
| Comparison | Multiple normalized price series, start-date selection, ticker fallback, excluded assets. | Live provider availability and browser chart interactions were not exercised. |
| My Portfolio | Selected saved assets and target weights, new/copy creation, authenticated save, on-demand symbol ingestion UI. | The current endpoint creates a configuration. This audit makes copy behavior explicit; it does not add an in-place target-allocation update endpoint. |
| Tracking | Quantity and average buy price editing, quote-currency checks, market value, purchase cost, unrealized gain/loss. | No FX conversion, broker synchronization, transaction ledger, realized PnL, taxes, or fees in cost basis. |
| Advisor | 30–365 day policies, minimum decision interval, fractional Kelly, damped turnover, capped target changes, risk/cost previews, persisted recommendations, acknowledgments. | Minimum holding constraints apply between stored recommendations, not to individual tax lots. Preview market snapshots are client-supplied and validated, not server-attested market data. |
| Smart DCA | Positive-deficit buy allocation, non-negative amounts, monthly deposit accounting, amount and horizon settings. | Fractional quantities are estimates. Broker lot sizes, cash execution, taxes, and order routing are absent. |
| Long-term projection | Seeded monthly log-normal Monte Carlo, end-month contributions, ordered 10/50/90 percentiles, deposited-capital subtraction, overflow guards. | Historical moments are estimation inputs, not forecasts. The model assumes stable moments; it does not model inflation or tax. |
| Optimizer | Real aligned stored prices, arithmetic moments, long-only frontier, max Sharpe/min variance, feasible automatic choice, manual target/risk ceiling, cash overlay. | Risk parity is available in the Advisor/API, not as a dedicated Optimizer control. Black-Litterman exists in core but has no user view-entry flow/API route here. No diversification/concentration cap is added by the automatic choice. |
| Risk Lab | Selected-portfolio simple-return VaR/CVaR and correlation, annualized volatility, historical benchmark crisis catalog and paths. | Crisis charts deliberately remain benchmark scenarios, with proxy/source disclosures. They do not claim to reconstruct the selected stocks' 2008/2020 holdings. L-VaR is available in the Advisor/API rather than a separate Lab distribution control. |
| Backtest | Stored prices, saved target weights, monthly sampling, rebalancing engine, equity/drawdown paths, flat-cost and recent-impact estimates. | Uses an explicit €1m starting balance and 8 bps assumption. It is a historical constant-target experiment, not a broker account replay or a point-in-time strategy selection test. Recent execution estimates are not historical order-book costs. |
| Qualitative feed | SEC/RSS provenance, classified-event tally, source health UI, independent numerical scores. | Model-generated labels can be wrong; remote collection, feed parsing across all providers, and licensing require separate checks. |
| Admin | Admin-key boundary remains server-side with restricted writes. | The browser admin experience is a restricted operational tool. It is not an end-user portfolio permission system. |
| Methodology | Existing explanations and medium/long-term positioning are retained. | Some detailed formula views use accessible mathematical markup rather than KaTeX. Not every legacy list/table metric has a dedicated formula drawer. |
| Privacy | Authenticated owner-data deletion route and restricted SQL RPC reviewed; owner ID comes from verified auth. | The RPC deletes portfolio data and cascading records. It does not delete the Supabase Auth identity or already delivered emails. Actual deployed grants/cascades need live verification. |

## Security and architecture review

- Protected portfolio, recommendation, tracking, and privacy lookups derive the owner from server-verified Supabase identity. Owner IDs are not accepted from the caller as authorization proof.
- SQL defines owner policies and service-role-only deletion/holdings RPCs. The backend secret client bypasses RLS, so explicit repository owner checks remain essential even with database policies.
- The API session cookie is HttpOnly, scoped to `/api`, SameSite Lax, and Secure in the production configuration. Mutating cookie requests require the custom request proof header; CORS uses configured exact origins. Existing security tests cover these boundaries.
- Supabase browser Auth still stores its access/refresh session in tab-scoped session storage. The whole Auth lifecycle is therefore not exclusively HttpOnly. The API cookie and browser Auth session are separate mechanisms. Cross-tab/multiple-account behavior requires browser testing because same-origin cookies are shared while session storage is tab-scoped.
- Production configuration disables public OpenAPI docs, limits request bodies and per-client request rate, and applies CSP/security headers. Rate limiting is process-local, and forwarded-IP trust depends on deployment proxy configuration.
- A tracked-file check confirms `.env` and `.env.local` are not tracked; example environment files are tracked. This is not a claim that all Git history or external deployment logs have been scanned for secrets.
- Core static checks reject direct database/web imports, application-service imports, and direct file/data I/O. Timing, NumPy/Pandas/SciPy, process workers, and optional native library loading remain explicit compute/profiling capabilities.
- Kalman initialization uses a fixed prior rather than future calibration observations. Regularization defaults to chronological validation with training-only selection and fold-local scaling. Cointegration has historical beta estimation. Their regression suites pass.
- HMM posterior histories are explicitly smoothed using the full supplied sequence. They are retrospective diagnostics; they must not be treated as causal trading signals for earlier dates without training/filtering boundaries.

## Data provenance and freshness

The application reads stored market observations. The daily multi-provider pipeline, FRED inputs, index universe manager, owner-holding priority, rotating universe batches, and on-demand lookup are application services outside the math core. The scheduled workflow runs at 22:30 UTC on weekdays and supports manual dispatch.

A successful mocked pipeline test establishes normalization/upsert behavior, not provider uptime or a guarantee that every ticker is refreshed daily. Provider quotas are process-local. Multiple processes or workflow restarts need shared quota accounting before such a guarantee can be made.

The crisis panels retain the checked-in source manifest and hashes. The 2008 panel has no BTC. VT and URTH are disclosed equity proxies; IEF is a Treasury proxy; the S&P price index excludes dividends. Static return values are reproduced in Python constants for zero data-I/O runtime execution.

To update a bundled snapshot, update the canonical CSV and provenance/hash together, then run:

```bash
python3 backend/scripts/embed_crisis_snapshots.py
python3 backend/scripts/embed_crisis_snapshots.py --check
```

Parity tests fail if canonical artifacts and embedded values diverge. No database migration was needed for this audit.

## Continuous verification

Automated GitHub Actions CI workflows have been removed. Run the local verification commands in the repository README before publishing changes. Scheduled Actions workflows for data refresh and monthly digests remain operational jobs and do not validate code changes.

## Production sign-off status

Local regression and build checks are satisfied. Full production sign-off remains conditional on the following concrete checks:

1. Browser-driven signup, confirmation, password reset, token refresh, sign-out, multi-tab account changes, and two distinct owners' portfolio isolation.
2. Confirmation of the deployed RLS policies, service-only RPC grants, migrations, and a disposable-account deletion check. The audit did not delete production user data.
3. Desktop/mobile checks for focus order, drawer behavior, ticker popovers, screen-reader labels, chart sizing, contrast, and slow/error network states. No browser automation package is installed in the repository, and no claim of complete visual/accessibility certification is made.
4. Real provider/CRON freshness, on-demand price caching, known-price currency/split adjustment checks, and email receipt for a deliberately generated recommendation.
5. Integration of the presently missing product flows if they are required for launch: a true account-value dashboard, in-place target-configuration edits, a Black-Litterman view editor/API, and complete per-metric educational drawers.
6. Additional negative/error-path tests for legacy routes, provider parsers, mail delivery, and persistence. Current statement coverage is materially below 100%.

A passing suite establishes that the exercised contracts pass in the tested environment. It does not establish 100% functional coverage, no future regressions, absence of memory leaks, or production-security certification.

## Reference checks

- [React effect cleanup and stale fetch responses](https://react.dev/reference/react/useEffect#fetching-data-with-effects) supports discarding outdated request results.
- [Supabase session architecture](https://supabase.com/docs/guides/auth/sessions) distinguishes access/refresh sessions from the application's API cookie.
- [NumPy 2.5 release notes](https://numpy.org/devdocs/release/2.5.0-notes.html) document deprecation of assigning array shape.
- [hmmlearn GaussianHMM API](https://hmmlearn.readthedocs.io/en/stable/api.html#hmmlearn.hmm.GaussianHMM) documents the EM model and covariance configuration used by this repository.
