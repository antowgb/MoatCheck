-- Store a user-entered acquisition date so Advisor trade signals can enforce
-- minimum holding periods per affected holding instead of relying only on
-- the date the user added it to MoatCheck.
alter table public.portfolio_holdings
    add column if not exists acquired_at date;

update public.portfolio_holdings
set acquired_at = created_at::date
where acquired_at is null;

alter table public.portfolio_holdings
    alter column acquired_at set default current_date,
    alter column acquired_at set not null;

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
        portfolio_id, ticker, quantity, average_buy_price, currency, acquired_at
    )
    select
        p_portfolio_id,
        upper(incoming->>'ticker'),
        (incoming->>'quantity')::double precision,
        (incoming->>'average_buy_price')::double precision,
        upper(incoming->>'currency'),
        coalesce((incoming->>'acquired_at')::date, current_date)
    from jsonb_array_elements(coalesce(p_holdings, '[]'::jsonb)) incoming
    on conflict (portfolio_id, ticker) do update set
        quantity = excluded.quantity,
        average_buy_price = excluded.average_buy_price,
        currency = excluded.currency,
        acquired_at = excluded.acquired_at;

    return query
    select * from public.portfolio_holdings holding
    where holding.portfolio_id = p_portfolio_id
    order by holding.ticker;
end;
$$;

revoke all on function public.replace_portfolio_holdings(uuid, jsonb) from public, anon, authenticated;
grant execute on function public.replace_portfolio_holdings(uuid, jsonb) to service_role;
