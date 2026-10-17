-- User-entered holdings and long-term contribution settings.
create table if not exists portfolio_holdings (
    id uuid primary key default gen_random_uuid(),
    portfolio_id uuid not null references portfolios(id) on delete cascade,
    ticker text not null,
    quantity double precision not null check (quantity > 0 and quantity < 'Infinity'::float8),
    average_buy_price double precision not null check (average_buy_price > 0 and average_buy_price < 'Infinity'::float8),
    currency text not null default 'EUR' check (currency ~ '^[A-Z]{3}$'),
    created_at timestamptz not null default now(),
    unique (portfolio_id, ticker)
);
create index if not exists idx_portfolio_holdings_portfolio on portfolio_holdings (portfolio_id);
alter table portfolio_holdings enable row level security;
drop policy if exists "owner manages portfolio holdings" on portfolio_holdings;
create policy "owner manages portfolio holdings" on portfolio_holdings
    for all to authenticated using (
        exists (select 1 from portfolios p where p.id = portfolio_id and p.owner_id = auth.uid())
    ) with check (
        exists (select 1 from portfolios p where p.id = portfolio_id and p.owner_id = auth.uid())
    );

create table if not exists portfolio_settings (
    id uuid primary key default gen_random_uuid(),
    portfolio_id uuid not null unique references portfolios(id) on delete cascade,
    monthly_dca_amount double precision not null default 0 check (monthly_dca_amount >= 0 and monthly_dca_amount < 'Infinity'::float8),
    target_horizon_years integer not null default 5 check (target_horizon_years between 1 and 20),
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);
alter table portfolio_settings enable row level security;
drop policy if exists "owner manages portfolio settings" on portfolio_settings;
create policy "owner manages portfolio settings" on portfolio_settings
    for all to authenticated using (
        exists (select 1 from portfolios p where p.id = portfolio_id and p.owner_id = auth.uid())
    ) with check (
        exists (select 1 from portfolios p where p.id = portfolio_id and p.owner_id = auth.uid())
    );

-- Keep a full holdings-set replacement atomic. The API verifies portfolio ownership
-- before calling this service-role-only RPC.
create or replace function public.replace_portfolio_holdings(
    p_portfolio_id uuid,
    p_holdings jsonb
) returns setof public.portfolio_holdings
language plpgsql
security definer
set search_path = public
as $$
begin
    delete from public.portfolio_holdings existing
    where existing.portfolio_id = p_portfolio_id
      and not exists (
          select 1
          from jsonb_array_elements(coalesce(p_holdings, '[]'::jsonb)) incoming
          where upper(incoming->>'ticker') = existing.ticker
      );

    insert into public.portfolio_holdings (
        portfolio_id, ticker, quantity, average_buy_price, currency
    )
    select
        p_portfolio_id,
        upper(incoming->>'ticker'),
        (incoming->>'quantity')::double precision,
        (incoming->>'average_buy_price')::double precision,
        upper(incoming->>'currency')
    from jsonb_array_elements(coalesce(p_holdings, '[]'::jsonb)) incoming
    on conflict (portfolio_id, ticker) do update set
        quantity = excluded.quantity,
        average_buy_price = excluded.average_buy_price,
        currency = excluded.currency;

    return query
    select * from public.portfolio_holdings holding
    where holding.portfolio_id = p_portfolio_id
    order by holding.ticker;
end;
$$;
revoke all on function public.replace_portfolio_holdings(uuid, jsonb) from public, anon, authenticated;
grant execute on function public.replace_portfolio_holdings(uuid, jsonb) to service_role;
