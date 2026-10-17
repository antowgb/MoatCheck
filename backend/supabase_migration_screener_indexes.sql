-- Query support and normalized membership metadata for the expanding universe.
-- Safe to re-run. Screener reads still pass through the existing RLS policies.

create index if not exists idx_stocks_sector_active
    on public.stocks (sector, ticker)
    where status = 'active' and is_benchmark = false;

create index if not exists idx_fundamentals_market_cap_latest
    on public.fundamentals (market_cap desc nulls last, ticker, report_date desc)
    where market_cap is not null;

create index if not exists idx_scores_composite_latest
    on public.scores (composite_score desc nulls last, ticker, computed_at desc)
    where composite_score is not null;

create index if not exists idx_scores_fundamental_latest
    on public.scores (fundamental_score desc nulls last, ticker, computed_at desc)
    where fundamental_score is not null;

create table if not exists public.market_universe_memberships (
    ticker text primary key references public.stocks (ticker) on delete cascade,
    memberships text[] not null default '{}',
    asset_type text not null default 'equity' check (asset_type in ('equity', 'etf')),
    tier text not null default 'extended' check (tier in ('core', 'extended')),
    rotation_batch smallint not null default 0 check (rotation_batch between 0 and 2),
    adv_usd double precision check (adv_usd is null or (adv_usd >= 0 and adv_usd < 'Infinity'::float8)),
    market_cap_usd double precision check (market_cap_usd is null or (market_cap_usd >= 0 and market_cap_usd < 'Infinity'::float8)),
    minimum_adv_usd double precision not null default 200000 check (minimum_adv_usd > 0 and minimum_adv_usd < 'Infinity'::float8),
    is_liquid boolean not null default false,
    updated_at timestamptz not null default now()
);

create index if not exists idx_universe_tier_batch_adv
    on public.market_universe_memberships (tier, rotation_batch, adv_usd desc nulls last, ticker);
create index if not exists idx_universe_memberships_gin
    on public.market_universe_memberships using gin (memberships);

alter table public.market_universe_memberships enable row level security;
drop policy if exists "public read universe memberships" on public.market_universe_memberships;
create policy "public read universe memberships" on public.market_universe_memberships
    for select to anon, authenticated using (true);
drop policy if exists "service manages universe memberships" on public.market_universe_memberships;
create policy "service manages universe memberships" on public.market_universe_memberships
    for all to service_role using (true) with check (true);
grant select on public.market_universe_memberships to anon, authenticated;
grant select, insert, update, delete on public.market_universe_memberships to service_role;
grant select, insert, update, delete on public.refresh_queue to service_role;
grant usage, select on sequence public.refresh_queue_id_seq to service_role;

-- Retire removed index members from the refresh universe without deleting
-- stocks, price history, or symbols explicitly added by a user search.
create or replace function public.prune_market_universe_memberships(p_current_tickers text[])
returns integer
language plpgsql
security invoker
set search_path = public
as $$
declare
    deleted_count integer;
begin
    delete from public.market_universe_memberships m
     where not (m.ticker = any(coalesce(p_current_tickers, '{}'::text[])))
       and not ('user_search' = any(m.memberships));
    get diagnostics deleted_count = row_count;
    return deleted_count;
end;
$$;
revoke all on function public.prune_market_universe_memberships(text[]) from public, anon, authenticated;
grant execute on function public.prune_market_universe_memberships(text[]) to service_role;

-- Ignore duplicate queue rows using the existing partial unique index on pending tickers.
create or replace function public.enqueue_refresh_tickers(p_tickers text[])
returns integer
language plpgsql
security invoker
set search_path = public
as $$
declare
    inserted_count integer;
begin
    insert into public.refresh_queue (ticker, priority)
    select distinct upper(trim(ticker)), 'high'
    from unnest(coalesce(p_tickers, '{}'::text[])) as ticker
    where nullif(trim(ticker), '') is not null
    on conflict do nothing;
    get diagnostics inserted_count = row_count;
    return inserted_count;
end;
$$;
revoke all on function public.enqueue_refresh_tickers(text[]) from public, anon, authenticated;
grant execute on function public.enqueue_refresh_tickers(text[]) to service_role;

-- Update only measured USD ADV values. Unknown/non-USD quote currencies must
-- not be compared to a USD threshold and remain unverified (NULL ADV).
create or replace function public.update_market_universe_liquidity(p_adv_by_ticker jsonb)
returns integer
language plpgsql
security invoker
set search_path = public
as $$
declare
    updated_count integer;
begin
    update public.market_universe_memberships m
       set adv_usd = parsed.adv_usd,
           is_liquid = parsed.adv_usd > m.minimum_adv_usd,
           updated_at = now()
      from (
        select upper(key) as ticker, value::double precision as adv_usd
          from jsonb_each_text(coalesce(p_adv_by_ticker, '{}'::jsonb))
         where value ~ '^[+]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$'
      ) parsed
     where m.ticker = parsed.ticker
       and parsed.adv_usd >= 0
       and parsed.adv_usd < 'Infinity'::float8;
    get diagnostics updated_count = row_count;
    return updated_count;
end;
$$;
revoke all on function public.update_market_universe_liquidity(jsonb) from public, anon, authenticated;
grant execute on function public.update_market_universe_liquidity(jsonb) to service_role;

-- Keyset pagination over the latest score/fundamental per ticker. The cursor
-- is (sort value, ticker); NULL score/value rows sort after measured values.
create or replace function public.screener_page(
    p_sector text default null,
    p_min_score double precision default null,
    p_min_growth double precision default null,
    p_min_risk_score double precision default null,
    p_market_cap_min double precision default null,
    p_market_cap_max double precision default null,
    p_pe_max double precision default null,
    p_debt_to_ebitda_max double precision default null,
    p_include_etfs boolean default false,
    p_sort_by text default 'composite_score',
    p_sort_dir text default 'desc',
    p_cursor_value double precision default null,
    p_cursor_ticker text default null,
    p_limit integer default 100
)
returns table (
    ticker text,
    name text,
    sector text,
    industry text,
    currency text,
    is_benchmark boolean,
    asset_type text,
    status text,
    composite_score double precision,
    fundamental_score double precision,
    risk_score double precision,
    revenue_growth_yoy double precision,
    market_cap double precision,
    pe_trailing double precision,
    debt_to_ebitda double precision
)
language plpgsql
stable
security invoker
set search_path = public
as $$
begin
    if p_sort_by not in ('composite_score', 'fundamental_score', 'risk_score', 'revenue_growth_yoy', 'market_cap', 'pe_trailing', 'debt_to_ebitda') then
        raise exception 'Unsupported screener sort column';
    end if;
    if p_sort_dir not in ('asc', 'desc') then
        raise exception 'Unsupported screener sort direction';
    end if;
    if p_limit < 1 or p_limit > 201 then
        raise exception 'Internal screener fetch size must be between 1 and 201';
    end if;

    return query
    with enriched as (
        select s.ticker, s.name, s.sector, s.industry, s.currency,
               s.is_benchmark, s.asset_type, s.status,
               latest_score.composite_score, latest_score.fundamental_score,
               latest_score.risk_score, latest_fund.revenue_growth_yoy,
               latest_fund.market_cap, latest_fund.pe_trailing,
               latest_fund.debt_to_ebitda,
               case p_sort_by
                   when 'composite_score' then latest_score.composite_score
                   when 'fundamental_score' then latest_score.fundamental_score
                   when 'risk_score' then latest_score.risk_score
                   when 'revenue_growth_yoy' then latest_fund.revenue_growth_yoy
                   when 'market_cap' then latest_fund.market_cap
                   when 'pe_trailing' then latest_fund.pe_trailing
                   when 'debt_to_ebitda' then latest_fund.debt_to_ebitda
               end as sort_value
        from public.stocks s
        left join lateral (
            select sc.composite_score, sc.fundamental_score, sc.risk_score
            from public.scores sc
            where sc.ticker = s.ticker
            order by sc.computed_at desc
            limit 1
        ) latest_score on true
        left join lateral (
            select f.revenue_growth_yoy, f.market_cap, f.pe_trailing, f.debt_to_ebitda
            from public.fundamentals f
            where f.ticker = s.ticker
            order by f.report_date desc nulls last
            limit 1
        ) latest_fund on true
        where s.status = 'active'
          and s.is_benchmark = false
          and (p_include_etfs or s.asset_type <> 'etf')
          and not exists (
              select 1 from public.market_universe_memberships m
              where m.ticker = s.ticker and m.adv_usd is not null and not m.is_liquid
          )
          and (p_sector is null or lower(trim(s.sector)) = lower(trim(p_sector)))
          and (p_min_score is null or latest_score.composite_score >= p_min_score)
          and (p_min_growth is null or latest_fund.revenue_growth_yoy >= p_min_growth)
          and (p_min_risk_score is null or latest_score.risk_score >= p_min_risk_score)
          and (p_market_cap_min is null or latest_fund.market_cap >= p_market_cap_min)
          and (p_market_cap_max is null or latest_fund.market_cap <= p_market_cap_max)
          and (p_pe_max is null or latest_fund.pe_trailing <= p_pe_max)
          and (p_debt_to_ebitda_max is null or latest_fund.debt_to_ebitda <= p_debt_to_ebitda_max)
    )
    select e.ticker, e.name, e.sector, e.industry, e.currency,
           e.is_benchmark, e.asset_type, e.status, e.composite_score,
           e.fundamental_score, e.risk_score, e.revenue_growth_yoy,
           e.market_cap, e.pe_trailing, e.debt_to_ebitda
    from enriched e
    where p_cursor_ticker is null
       or (p_cursor_value is null and e.sort_value is null and e.ticker > p_cursor_ticker)
       or (p_cursor_value is not null and (
            e.sort_value is null
            or (p_sort_dir = 'desc' and e.sort_value < p_cursor_value)
            or (p_sort_dir = 'asc' and e.sort_value > p_cursor_value)
            or (e.sort_value = p_cursor_value and e.ticker > p_cursor_ticker)
       ))
    order by
        case when p_sort_dir = 'asc' then e.sort_value end asc nulls last,
        case when p_sort_dir = 'desc' then e.sort_value end desc nulls last,
        e.ticker asc
    limit p_limit;
end;
$$;
revoke all on function public.screener_page(text,double precision,double precision,double precision,double precision,double precision,double precision,double precision,boolean,text,text,double precision,text,integer) from public;
grant execute on function public.screener_page(text,double precision,double precision,double precision,double precision,double precision,double precision,double precision,boolean,text,text,double precision,text,integer) to anon, authenticated, service_role;
