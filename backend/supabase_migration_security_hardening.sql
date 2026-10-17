-- Explicit owner keys on portfolio child tables allow simple, auditable RLS.
-- Safe to reapply after the existing portfolio tracking and advisor migrations.
alter table public.portfolio_holdings add column if not exists owner_id uuid references auth.users (id) on delete cascade;
alter table public.portfolio_settings add column if not exists owner_id uuid references auth.users (id) on delete cascade;

update public.portfolio_holdings h
   set owner_id = p.owner_id
  from public.portfolios p
 where p.id = h.portfolio_id and h.owner_id is distinct from p.owner_id;
update public.portfolio_settings s
   set owner_id = p.owner_id
  from public.portfolios p
 where p.id = s.portfolio_id and s.owner_id is distinct from p.owner_id;

create or replace function public.set_portfolio_child_owner()
returns trigger
language plpgsql
security definer
set search_path = ''
as $$
begin
    select p.owner_id into new.owner_id
      from public.portfolios p where p.id = new.portfolio_id;
    if new.owner_id is null then
        raise exception 'portfolio owner is required';
    end if;
    return new;
end;
$$;
revoke all on function public.set_portfolio_child_owner() from public, anon, authenticated;

drop trigger if exists set_portfolio_holdings_owner on public.portfolio_holdings;
create trigger set_portfolio_holdings_owner before insert or update of portfolio_id
    on public.portfolio_holdings for each row execute function public.set_portfolio_child_owner();
drop trigger if exists set_portfolio_settings_owner on public.portfolio_settings;
create trigger set_portfolio_settings_owner before insert or update of portfolio_id
    on public.portfolio_settings for each row execute function public.set_portfolio_child_owner();

-- Legacy unowned portfolio rows can retain NULL here; owner-scoped RLS denies
-- them to every authenticated user. New child rows get their owner by trigger.

drop policy if exists "owner manages portfolio holdings" on public.portfolio_holdings;
create policy "owner manages portfolio holdings" on public.portfolio_holdings
    for all to authenticated using (auth.uid() = owner_id) with check (auth.uid() = owner_id);
drop policy if exists "owner manages portfolio settings" on public.portfolio_settings;
create policy "owner manages portfolio settings" on public.portfolio_settings
    for all to authenticated using (auth.uid() = owner_id) with check (auth.uid() = owner_id);

drop policy if exists "owners manage advisor recommendations" on public.advisor_recommendations;
create policy "owners manage advisor recommendations" on public.advisor_recommendations
    for all to authenticated using (auth.uid() = owner_id)
    with check (auth.uid() = owner_id and exists (
        select 1 from public.portfolios p where p.id = portfolio_id and p.owner_id = auth.uid()
    ));

-- Owner-scoped data deletion is intentionally backend-only.
revoke all on function public.delete_owner_quant_data(uuid) from public, anon, authenticated;
grant execute on function public.delete_owner_quant_data(uuid) to service_role;
