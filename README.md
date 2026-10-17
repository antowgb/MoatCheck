# MoatCheck

MoatCheck is an open-source company research and portfolio analysis workspace. It brings company fundamentals, stored market data, portfolio tracking, risk analysis, historical scenarios, and medium to long term allocation research into one application.

[Open the live app](https://moatcheck.onrender.com) · [Methodology](https://weissgerber.dev/products/moatcheck/methodology) · [Project overview](https://weissgerber.dev/en/products/moatcheck/)

MoatCheck is an educational research project. It does not place orders, connect to a brokerage, or provide personal financial advice. Results depend on historical data and model assumptions. Review the inputs and limitations before using an estimate.

## Product overview

The application is built for portfolio decisions held over weeks or months. The Advisor supports a configurable 30 to 365 day horizon. Research pages can cover longer histories, but the available price series and models do not make the product an intraday trading system.

| Workspace | What it provides |
| --- | --- |
| Dashboard | Overview of the selected saved portfolio, research coverage, and linked assets. It summarizes target allocations, not brokerage account balances. |
| Screener | Search and filter the stored stock and ETF universe by supported sector, company, market-cap, score, valuation, and risk fields. Results use bounded cursor pagination. |
| Stock detail | Stored price history, fundamental and risk scores, company metrics, and available SEC/RSS qualitative events. |
| Compare | Normalized historical price comparison for selected assets. |
| My Portfolio | Create, select, edit, and delete saved target configurations. Holdings, average buy price, and DCA settings are managed separately from target weights. |
| Portfolio Tracking | Record quantities and average purchase prices, then estimate market value and unrealized PnL when usable stored prices are available. It is not connected to a broker or transaction ledger. |
| Optimizer | Analyze real aligned stored price history with a long-only Markowitz frontier and supported allocation controls. Historical moments are estimates, not return forecasts. |
| Risk & Stress Lab | Portfolio VaR/CVaR, correlations, and historical benchmark crisis scenarios with explicit proxy/source notes. The crisis replay is not the selected portfolio's historical holdings. |
| Portfolio Advisor | Review a medium/long-term allocation proposal, turnover and cost estimates, risk comparisons, optional Smart DCA, regime context, projections, rationale, and human acknowledgment. Saving a recommendation may email the account address when email delivery is configured. |
| Backtest | Run a historical fixed-target rebalancing experiment with equity and drawdown paths and estimated cost assumptions. It is not a broker statement or point-in-time strategy certification. |
| Qualitative Events | Browse available filing and RSS-derived event summaries. Model-generated labels can be wrong and are separate from numerical scores. |
| Methodology and Privacy | Read model/data assumptions and request deletion of persisted portfolio records. |

Changing the selected saved portfolio in the header updates the shared application context. Some pages use a local draft or explicit asset selection, so their displayed inputs can still require review before calculation.

## Architecture

```text
frontend/                       Next.js App Router, React, TypeScript
  app/                           Dashboard, research, portfolio, risk, advisor pages
  components/                    Shared shell, charts, forms, explainability UI
  lib/                           API client, portfolio context, browser calculations
backend/app/
  api/v1/                         Authenticated product endpoints
  api/v2/                         Explicit-input research and quant endpoints
  advisor/                        Medium/long-horizon application orchestration
  core/                           Pure quantitative and profiling modules
  data/, services/                Provider ingestion, normalization, persistence orchestration
  repositories/, models/, db/     Supabase persistence adapters and contracts
  qualitative/                    SEC/RSS collection and event classification
backend/tests/                    Offline core, API, advisor, database, and service tests
```

The math core is kept separate from FastAPI, databases, and market-data acquisition. Application services validate and align inputs, call core calculations, and serialize results. API v1 powers the product; v2 exposes additional explicit-input research contracts. The user interface is statically exported and served by FastAPI in the Render deployment.

## Data sources and interpretation

The application primarily analyzes stored observations. Alpha Vantage supplies the existing weekly price/fundamental refresh path. The optional daily ingestion workflow can use Yahoo Finance, FRED, Tiingo, FMP, and Binance where applicable and permitted. SEC EDGAR and enabled RSS sources provide qualitative event metadata; Groq classification is optional. Finnhub is a constrained fallback for selected fundamental fields.

The daily market pipeline is not the same dataset as the application's existing weekly `price_history`. The pipeline stores daily bars separately. Many existing company and portfolio views still use the weekly series. Weekly observations can hide within-week losses and provide less detail for risk estimates. Local split adjustment is applied to the relevant stored price path; dividends are not included as total return. Provider coverage, licensing, refresh success, and observation dates vary. A page may show the last successful stored snapshot rather than a live quote.

Historical crisis panels are bundled benchmark data, not reconstructions of a user's historical portfolio. The 2008 panel has no BTC history; equity and Treasury series use disclosed proxies. See the data guide for source, cadence, and provider licensing details.

For each result, check the observation date, sample length, currency, price convention, annualization, and cost assumptions. Latency describes application processing, not market-data freshness. Execution drag is an estimate, not an order-book quote.

## Accounts and privacy

Supabase Auth provides email/password sign-in. Users create accounts in MoatCheck; they do not need to create their own Supabase project. The browser exchanges the verified Supabase session for a short-lived HttpOnly API cookie. The browser-side Supabase Auth SDK also maintains its own session storage, so the entire authentication lifecycle is not exclusively HttpOnly.

Persisted portfolios, holdings, recommendations, backtests, and stress records are associated with the verified user. The backend uses a server-only Supabase credential for repository access and performs owner checks. Never expose `SUPABASE_SECRET_KEY` or `SUPABASE_SERVICE_ROLE_KEY` to frontend code. The privacy action deletes persisted quant records covered by the database routine; it does not delete the Supabase Auth account or emails already delivered.

Read [Security](SECURITY.md), [Auth setup](frontend/AUTH_SETUP.md), and the in-app Privacy page before changing deployment settings. The current rate limiter is process-local and is not distributed DDoS protection.

## Run locally

### 1. Set up Supabase

Create or use one Supabase project for the application. Apply the base schema and the migrations required by the features you use from `backend/` in order. At minimum, review:

- `supabase_schema.sql`
- `supabase_migration_v2_qualitative.sql`
- `supabase_migration_quant_persistence.sql`
- `supabase_migration_advisor.sql`
- `supabase_migration_portfolio_tracking.sql`
- `supabase_migration_holding_acquisition_dates.sql`
- `supabase_migration_privacy.sql`
- `supabase_migration_screener_indexes.sql`
- `supabase_migration_market_data_pipeline.sql`

The files are SQL migrations. Running the Supabase CLI locally does not apply them to the hosted project unless the CLI is linked/authenticated and a push is explicitly run. See the Auth guide for email provider and redirect URL setup.

### 2. Start the backend

From the repository root:

```bash
cd backend
python3 -m venv .venv                 # once, if needed
./.venv/bin/python -m pip install -r requirements.txt
cp -n .env.example .env               # preserves an existing .env
```

Edit `backend/.env` with the Supabase project URL, a public `SUPABASE_KEY` (publishable/anon), and a server-only `SUPABASE_SECRET_KEY` (or the legacy service-role variable). Add only the provider credentials needed for your use. Never paste or commit the contents of a real environment file.

```bash
./.venv/bin/python -m uvicorn app.main:app --reload --port 8000
```

The API health route is `http://localhost:8000/health`. Development OpenAPI docs are at `http://localhost:8000/docs`.

### 3. Start the frontend

In a second terminal:

```bash
cd frontend
npm ci
cp -n .env.example .env.local
```

Set the following public values in `frontend/.env.local`:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_SUPABASE_URL=https://<project-ref>.supabase.co
NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY=<publishable-key>
```

The legacy `NEXT_PUBLIC_SUPABASE_ANON_KEY` is also supported. These values are intended for browser use. Never put a Supabase secret/service-role key in a `NEXT_PUBLIC_*` variable.

```bash
npm run dev
```

Open `http://localhost:3000`. If needed, run `PORT=3001 npm run dev`.

### Optional settings

- `NEXT_PUBLIC_SITE_URL`: canonical origin for confirmation/reset links. Set it in the frontend build environment and add its `/auth/callback` URL to Supabase's redirect allowlist.
- `RESEND_API_KEY`, `ADVISOR_EMAIL_FROM`: backend-only email delivery for saved recommendations. The sender domain must be verified with Resend. Preview calculations do not send email.
- `FRED_API_KEY`, `FMP_API_KEY`, `TIINGO_API_KEY`: optional market pipeline providers.
- `EDGAR_CONTACT_EMAIL`: descriptive SEC EDGAR User-Agent contact.
- `GROQ_API_KEY`: optional event classification.
- `ADMIN_API_KEY`: protects restricted operational write endpoints.

See `backend/.env.example` and `frontend/.env.example` for the full list. The examples document names and safe placeholders; they are not production credentials.

## Data pipeline operations

The weekday GitHub Actions workflow runs at 22:30 UTC and can be dispatched manually. It prioritizes configured/held tickers and the core universe, then rotates extended symbols. It performs incremental upserts and keeps daily prices separate from the weekly series. Provider limits and license flags are described in [Market data pipeline](docs/market-data-pipeline.md). Enabling a provider key does not itself grant storage or redistribution rights.

## Verification

Run from the repository root:

```bash
cd backend
./.venv/bin/python -m pytest tests/ -W error
cd ../frontend
npm run typecheck
node lib/portfolio-math.test.mjs
node lib/alignedReturn.test.mjs
npm run build
```

There is no automated GitHub Actions CI workflow. Run these checks locally before publishing changes. The scheduled GitHub Actions workflows are operational jobs for data refresh and monthly digests, not CI; they do not validate code changes.

## Documentation index

- [Platform audit and remaining verification](docs/platform-audit-2026-10-17.md)
- [Data sources, freshness, and advisory horizon](docs/data-sources-and-advisory-horizon.md)
- [Market data pipeline operations](docs/market-data-pipeline.md)
- [Security and responsible disclosure](SECURITY.md)
- [Supabase Auth setup](frontend/AUTH_SETUP.md)
- [Gmail newsletter collector setup](backend/GMAIL_SETUP.md), optional and disabled by default
- [Quant refactoring strategy](docs/moatcheck-quant-refactor-strategy.md)
- [Quant finance learning roadmap](Quant%20Finance%20Master%20Roadmap%20%26%20Checklist.md)
- [Historical pre-auth code audit](docs/moatcheck-quant-audit.md), retained for engineering history

The detailed [platform audit](docs/platform-audit-2026-10-17.md) records what was verified, the exact local test results and known product limitations. Passing tests do not establish complete browser, operational, security, or investment-model coverage.
