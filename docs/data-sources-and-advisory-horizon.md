# Data sources, freshness, and investment horizon

## Product scope

MoatCheck is a public research and education tool for medium and long-term investing. Portfolio analysis and the Advisor are designed around holding windows of roughly 30 to 365 days. Company research can inform longer horizons, but the application does not place broker orders, issue intraday signals, or guarantee outcomes. Recommendations are model outputs for human review, not personal financial advice.

The Advisor uses aligned market history, conservative fractional Kelly sizing, configurable decision intervals, a one-way turnover cap, and explicit spread and market-impact assumptions. These controls reduce sensitivity to short-lived noise. They do not remove estimation error or investment risk.

## Current source map

| Data | Source and product use | Timing and limitations |
| --- | --- | --- |
| Existing equity and ETF history | Alpha Vantage weekly closes feed many company, portfolio, optimizer, risk, and Advisor views. | Weekly data can hide within-week drawdowns. Split adjustment is applied locally; dividends are not included in total returns. The Advisor infers cadence from observed dates and annualizes weekly data using about 52 periods per year. It converts weekly volatility to a daily proxy only where ADV-based liquidation-cost formulas require daily units. |
| Daily market history | The separate ingestion workflow tries Yahoo Finance first, then enabled Tiingo/FMP fallbacks. Binance is used for supported crypto pairs. | Daily bars are stored separately from the existing weekly `price_history`. Many current screens continue to use weekly history. Provider availability, terms, and symbol coverage vary. |
| Macro observations | FRED is primary; enabled FMP fallback may supply supported observations. | Coverage depends on configured series, keys, quotas, and successful scheduled runs. |
| Quarterly fundamentals | Alpha Vantage statements and company overview. | Provider fields and coverage vary. Historical tests use a 60-day availability lag where verified filing acceptance dates are unavailable; this is an approximation, not official filing knowledge time. |
| Selected alternate fundamentals | Finnhub is optional; selected fields are substituted only after fiscal-date and magnitude checks. | This is a constrained fallback, not independent verification of every field. |
| Filings and news | SEC EDGAR, enabled investor-relations and press RSS feeds, and an optional newsletter collector. | Coverage and publication times differ. A source may be unavailable or stale. Gmail collection is disabled by default. |
| Event labels | Optional Groq classification of event descriptions. | Model labels can be wrong or incomplete. Qualitative tallies remain separate from numerical fundamental, risk, and composite scores. |
| Persistence | Supabase stores research data, daily pipeline outputs, and authenticated user portfolio/advisory records. | A successful page load may show the latest stored snapshot rather than a live quote. Check observation dates and source metadata. |

## Universe and refresh behavior

The universe manager imports current reference lists for the S&P 500, NASDAQ 100, STOXX Europe 600, a Russell 2000 ETF holdings proxy, and selected ETFs. New symbols can be ingested on demand. These lists are candidate sources, not a guarantee of 3,000 fully covered, liquid securities. The Russell ETF's holdings can differ from the licensed index constituent list, and public constituents pages can change format or timing.

The weekday pipeline prioritizes symbols held or configured by users plus core candidates. Remaining eligible symbols rotate through several execution days. Unknown symbols can enter a background end-of-day fetch and be added to the ingestion queue. Actual freshness depends on provider success, quotas, symbol support, database connectivity, and job completion. A priority designation is not proof that a fresh quote was received.

Provider credentials do not automatically grant rights to persist or publicly display data. FMP and Tiingo storage are disabled unless the corresponding license confirmation flags are enabled. Confirm applicable vendor terms before changing those flags. The local quota guards are process-local and do not coordinate across independent workers or workflow runs.

## Interpreting results

For each result, review the observation timestamp, sample size, currency, price adjustment, cadence, annualization, missing-data policy, and transaction-cost assumptions. API latency is not data freshness. Execution-cost values are model estimates, not current order-book quotes. Historical return and covariance estimates can be unstable, especially with short samples or large expected-return estimates.

Historical crisis panels are bundled benchmark scenarios, not reconstructions of a user's historical holdings. The 2008 panel has no BTC observations. Selected global equity and Treasury series use named ETF proxies; the S&P series is a price index and excludes dividends. Refer to the Risk & Stress Lab labels and the [platform audit](platform-audit-2026-10-17.md) for details.

## Improvement priorities

1. Preserve provider, source timestamp, ingestion timestamp, currency, and price-adjustment convention with every observation.
2. Use SEC filing acceptance timestamps for point-in-time fundamentals wherever provider terms permit.
3. Add dividend-adjusted total-return history while retaining raw and split-adjusted prices separately.
4. Reconcile important fields against a second licensed source and show discrepancies.
5. Keep per-asset freshness and missing-coverage indicators visible in every relevant analysis.
6. Expand daily data into user-facing calculations only after checking coverage and storage/display rights, then document cadence and annualization changes.

## Email policy for the Advisor

Each saved formal recommendation is emailed to the authenticated account address when email delivery is configured. Preview calculations are not saved and do not send email. Delivery requires backend Resend credentials and a verified sender. If delivery is unavailable or fails, the recommendation remains saved and its delivery status is returned to the workspace. Email delivery must not change the calculation or expose another user's portfolio data.
