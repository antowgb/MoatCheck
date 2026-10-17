-- Persistence tables for the MoatCheck quant application layer.
-- Apply this migration in the Supabase SQL editor before enabling repositories.

create table if not exists public.portfolios (
    id uuid primary key default gen_random_uuid(),
    owner_id uuid references auth.users (id) on delete cascade,
    name text not null check (length(name) between 1 and 120),
    description text,
    tickers text[] not null check (cardinality(tickers) > 0),
    target_weights jsonb not null check (jsonb_typeof(target_weights) = 'object'),
    risk_constraints jsonb not null default '{}'::jsonb,
    optimization_parameters jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create index if not exists idx_portfolios_owner_updated
    on public.portfolios (owner_id, updated_at desc);

create or replace function public.set_portfolio_updated_at()
returns trigger
language plpgsql
as $$
begin
    new.updated_at = now();
    return new;
end;
$$;

drop trigger if exists set_portfolios_updated_at on public.portfolios;
create trigger set_portfolios_updated_at
    before update on public.portfolios
    for each row execute function public.set_portfolio_updated_at();

create table if not exists public.backtest_runs (
    id uuid primary key default gen_random_uuid(),
    portfolio_id uuid references public.portfolios (id) on delete set null,
    strategy_name text not null check (length(strategy_name) between 1 and 120),
    start_date date not null,
    end_date date not null check (end_date >= start_date),
    tickers text[] not null check (cardinality(tickers) > 0),
    net_pnl_series jsonb not null default '[]'::jsonb,
    observation_dates jsonb,
    turnover_series jsonb not null default '[]'::jsonb,
    total_turnover double precision not null default 0 check (total_turnover >= 0),
    average_annual_turnover double precision not null default 0 check (average_annual_turnover >= 0),
    total_transaction_cost double precision not null default 0 check (total_transaction_cost >= 0),
    total_transaction_drag_bps double precision not null default 0 check (total_transaction_drag_bps >= 0),
    execution_cost_bps_of_traded_notional double precision not null default 0 check (execution_cost_bps_of_traded_notional >= 0),
    metrics jsonb not null default '{}'::jsonb,
    telemetry jsonb not null default '{}'::jsonb,
    completed_at timestamptz,
    created_at timestamptz not null default now()
);

create index if not exists idx_backtest_runs_portfolio_created
    on public.backtest_runs (portfolio_id, created_at desc);

create table if not exists public.stress_tests (
    id uuid primary key default gen_random_uuid(),
    portfolio_id uuid references public.portfolios (id) on delete set null,
    stress_type text not null check (stress_type in ('historical_crisis', 'reverse_stress', 'macro_scenario')),
    scenario_name text not null check (length(scenario_name) between 1 and 160),
    scenario_start date,
    scenario_end date,
    inputs jsonb not null default '{}'::jsonb,
    results jsonb not null default '{}'::jsonb,
    portfolio_return double precision,
    worst_drawdown double precision,
    minimum_shock_magnitude double precision check (minimum_shock_magnitude is null or minimum_shock_magnitude >= 0),
    telemetry jsonb not null default '{}'::jsonb,
    completed_at timestamptz,
    created_at timestamptz not null default now(),
    check ((scenario_start is null) = (scenario_end is null)),
    check (scenario_start is null or scenario_end >= scenario_start)
);

create index if not exists idx_stress_tests_portfolio_created
    on public.stress_tests (portfolio_id, created_at desc);

alter table public.portfolios enable row level security;
alter table public.backtest_runs enable row level security;
alter table public.stress_tests enable row level security;

drop policy if exists "owners manage portfolios" on public.portfolios;
create policy "owners manage portfolios" on public.portfolios
    for all to authenticated
    using (owner_id = (select auth.uid()))
    with check (owner_id = (select auth.uid()));

drop policy if exists "owners read backtest runs" on public.backtest_runs;
create policy "owners read backtest runs" on public.backtest_runs
    for select to authenticated
    using (exists (
        select 1 from public.portfolios p
        where p.id = portfolio_id and p.owner_id = (select auth.uid())
    ));

drop policy if exists "owners read stress tests" on public.stress_tests;
create policy "owners read stress tests" on public.stress_tests
    for select to authenticated
    using (exists (
        select 1 from public.portfolios p
        where p.id = portfolio_id and p.owner_id = (select auth.uid())
    ));
