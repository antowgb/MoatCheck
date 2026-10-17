# Market and macro data pipeline

The ingestion code runs in the application layer. Provider clients, normalization, quota guards, and Supabase writes live outside `backend/app/core/`, which remains a pure calculation boundary.

## What the pipeline does

`backend/app/services/universe_manager.py` refreshes candidate index/ETF membership, applies symbol and liquidity eligibility checks where data is available, queues new symbols, and assigns priority tiers. User-held/configured tickers and the core universe receive daily priority. The extended universe rotates across the configured multi-day window. Public list sources and ETF holdings are imperfect proxies; they do not certify complete or licensed index membership.

`backend/app/services/data_pipeline_orchestrator.py` routes requests across configured price and macro providers, applies local quotas and fallbacks, normalizes records, removes duplicates, derives analysis inputs, and upserts daily prices, macro indicators, and supported company data. An unknown symbol requested by a user can be queued for an on-demand end-of-day fetch and cached if a provider returns valid data.

The scheduled job prioritizes portfolio holdings/configurations and core candidates, then processes a rotating batch of the remaining universe. This is a scheduling policy, not a guarantee that every ticker will have a new observation every day. Ticker support, rate limits, provider errors, holidays, and database availability can all delay an update.

## Database preparation

Apply `backend/supabase_migration_market_data_pipeline.sql` to the linked Supabase project before enabling daily writes. Review the SQL file and use the intended project/credentials. A local Supabase CLI status command checks local containers; it does not prove that the hosted project has this migration.

The job writes daily bars to `prices`, separately from the existing weekly `price_history` used by many user-facing analytics. Missing market sessions may be forward-filled in derived calculation panels, but invented quote rows are not written as market observations. Upserts use stable keys so overlapping reruns do not create duplicate records.

## GitHub Actions setup

`.github/workflows/data_pipeline.yml` runs Monday to Friday at 22:30 UTC and supports manual dispatch. Configure these repository secrets as needed:

- `SUPABASE_URL`
- `SUPABASE_SECRET_KEY` or the legacy `SUPABASE_SERVICE_ROLE_KEY`
- `FRED_API_KEY`
- `FMP_API_KEY` (optional)
- `TIINGO_API_KEY` (optional)
- `GROQ_API_KEY` (optional classification)
- `EDGAR_CONTACT_EMAIL`

Configure repository variables only after reviewing vendor terms:

- `FMP_DATA_LICENSE_CONFIRMED=true` only if the applicable FMP contract permits this database persistence and public display.
- `TIINGO_PERSISTENCE_ALLOWED=true` only if the applicable Tiingo terms permit the same use.

The workflow uses explicit defaults for core batch size, rotation days, liquidity threshold, and local provider quotas. The universe service runs before the market-data orchestration. The workflow uses a concurrency group to avoid overlapping scheduled runs in this repository.

## Provider order and quota controls

- Equities: Yahoo Finance first, then Tiingo and FMP when configured and allowed by the pipeline policy.
- Crypto: Binance for supported pairs.
- Macro: FRED first, with supported FMP fallback.
- Qualitative events: existing SEC, IR, press, and optional Groq pipeline.

The current local defaults include an FMP daily request cap and Tiingo hourly/daily caps. FRED requests are paced and 429 responses can trigger fallback. Provider limits and plan terms change; confirm current limits before increasing usage. The counters are local to a process/run, not a shared global quota service.

## Local use

From `backend/`, with the Python environment and server-side variables configured:

```bash
./.venv/bin/python -m app.services.universe_manager
./.venv/bin/python -m app.services.data_pipeline_orchestrator
```

These commands contact real external providers and write to the configured Supabase project. For offline development, run the mocked tests instead:

```bash
./.venv/bin/python -m pytest tests/services/test_data_pipeline.py tests/services/test_universe_manager.py -W error
```

## Data handling and limitations

- Do not place service-role keys or provider credentials in frontend variables or commit them.
- Provider API access is not a data redistribution license. Keep persistence flags off until permitted.
- Yahoo Finance availability and terms may not support durable public redistribution. Review current terms before relying on it for a public dataset.
- The pipeline's daily history does not automatically replace weekly inputs in the existing dashboard, optimizer, risk, or Advisor pages.
- Test success demonstrates mocked normalization/upsert logic. Check the actual GitHub Actions run, Supabase records, timestamps, currencies, and provider coverage to verify operations.
- The generated `rolling_covariance` and ADV are analysis data; transaction-cost outputs remain estimates rather than exchange quotes.
