-- One-time data cleanup. Run after the reduced-universe pipeline is deployed.
-- Keep the 250 largest S&P 500 / NASDAQ-100 equities, curated major ETFs,
-- user-added symbols, benchmarks, and every symbol referenced by a portfolio.
-- Retain the wider S&P/NASDAQ metadata catalog for future ranking.
do $cleanup$
declare
  removed_prices bigint;
  removed_weekly_prices bigint;
  removed_stocks bigint;
begin
  create temporary table _moatcheck_catalog_keep (
    ticker text primary key
  ) on commit drop;
  create temporary table _moatcheck_price_keep (
    ticker text primary key
  ) on commit drop;

  insert into _moatcheck_catalog_keep (ticker)
  select m.ticker
  from public.market_universe_memberships m
  where m.memberships && array['sp500', 'nasdaq100', 'major_etf', 'user_search']::text[]
  union
  select upper(ticker) from public.stocks where is_benchmark
  union
  select upper(portfolio_ticker.ticker)
  from public.portfolios p
  cross join lateral unnest(p.tickers) as portfolio_ticker(ticker)
  union
  select upper(ticker) from public.portfolio_holdings;

  insert into _moatcheck_catalog_keep (ticker)
  select distinct s.sector_benchmark_ticker
  from public.stocks s
  join _moatcheck_catalog_keep keep on keep.ticker = s.ticker
  where s.sector_benchmark_ticker is not null
  on conflict do nothing;

  insert into _moatcheck_price_keep (ticker)
  with ranked_core as (
    select m.ticker,
           row_number() over (
             order by m.market_cap_usd desc nulls last,
                      m.adv_usd desc nulls last,
                      m.ticker
           ) as core_rank
    from public.market_universe_memberships m
    where m.asset_type = 'equity'
      and m.memberships && array['sp500', 'nasdaq100']::text[]
  )
  select ticker from ranked_core where core_rank <= 250
  union
  select m.ticker
  from public.market_universe_memberships m
  where 'major_etf' = any(m.memberships)
     or 'user_search' = any(m.memberships)
  union
  select upper(ticker) from public.stocks where is_benchmark
  union
  select upper(portfolio_ticker.ticker)
  from public.portfolios p
  cross join lateral unnest(p.tickers) as portfolio_ticker(ticker)
  union
  select upper(ticker) from public.portfolio_holdings;

  insert into _moatcheck_price_keep (ticker)
  select distinct s.sector_benchmark_ticker
  from public.stocks s
  join _moatcheck_price_keep keep on keep.ticker = s.ticker
  where s.sector_benchmark_ticker is not null
  on conflict do nothing;

  delete from public.prices p
  where not exists (
    select 1 from _moatcheck_price_keep keep where keep.ticker = p.ticker
  );
  get diagnostics removed_prices = row_count;

  delete from public.price_history p
  where not exists (
    select 1 from _moatcheck_price_keep keep where keep.ticker = p.ticker
  );
  get diagnostics removed_weekly_prices = row_count;

  delete from public.company_fundamentals f
  where not exists (
    select 1 from _moatcheck_catalog_keep keep where keep.ticker = f.ticker
  );

  delete from public.refresh_queue q
  where not exists (
    select 1 from _moatcheck_price_keep keep where keep.ticker = q.ticker
  );

  update public.stocks s
  set sector_benchmark_ticker = null
  where s.sector_benchmark_ticker is not null
    and not exists (
      select 1 from _moatcheck_catalog_keep keep
      where keep.ticker = s.sector_benchmark_ticker
    );

  delete from public.stocks s
  where not exists (
    select 1 from _moatcheck_catalog_keep keep where keep.ticker = s.ticker
  );
  get diagnostics removed_stocks = row_count;

  raise notice 'Market cleanup removed % daily price rows, % weekly price rows, and % stock catalog rows',
    removed_prices, removed_weekly_prices, removed_stocks;
end
$cleanup$;
