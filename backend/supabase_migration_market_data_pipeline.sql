-- Canonical daily provider data used by the Market & Macro Data Orchestrator.
-- Apply once in the Supabase SQL editor. Upsert keys make scheduled retries safe.

create table if not exists prices (
    ticker text not null,
    date date not null,
    open double precision,
    high double precision,
    low double precision,
    close double precision not null,
    adjusted_close double precision not null,
    volume bigint,
    log_return double precision,
    adv double precision,
    provider text not null,
    ingested_at timestamptz not null default now(),
    primary key (ticker, date)
);
create index if not exists idx_prices_date on prices (date desc);
alter table prices enable row level security;
drop policy if exists "anon read prices" on prices;
create policy "anon read prices" on prices for select to anon, authenticated using (true);

create table if not exists macro_indicators (
    series_id text not null,
    observation_date date not null,
    provider text not null,
    name text not null,
    value double precision not null,
    ingested_at timestamptz not null default now(),
    primary key (series_id, observation_date, provider)
);
create index if not exists idx_macro_indicators_date on macro_indicators (observation_date desc);
alter table macro_indicators enable row level security;
drop policy if exists "anon read macro indicators" on macro_indicators;
create policy "anon read macro indicators" on macro_indicators for select to anon, authenticated using (true);

create table if not exists company_fundamentals (
    ticker text not null,
    as_of_date date not null,
    provider text not null,
    metrics jsonb not null default '{}'::jsonb,
    ingested_at timestamptz not null default now(),
    primary key (ticker, as_of_date, provider)
);
create index if not exists idx_company_fundamentals_ticker_date
    on company_fundamentals (ticker, as_of_date desc);
alter table company_fundamentals enable row level security;
drop policy if exists "anon read company fundamentals" on company_fundamentals;
create policy "anon read company fundamentals" on company_fundamentals for select to anon, authenticated using (true);
