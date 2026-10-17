-- Owner-scoped, immutable recommendations with an explicit review marker.
create table if not exists public.advisor_recommendations (
    id uuid primary key default gen_random_uuid(),
    owner_id uuid not null references auth.users (id) on delete cascade,
    portfolio_id uuid not null references public.portfolios (id) on delete cascade,
    horizon_days integer not null check (horizon_days between 30 and 365),
    minimum_holding_period_days integer not null check (
        minimum_holding_period_days between 30 and horizon_days
    ),
    snapshot_timestamp timestamptz not null,
    status text not null default 'pending_review'
        check (status in ('pending_review', 'acknowledged')),
    acknowledged_at timestamptz,
    recommendation jsonb not null,
    rationale jsonb not null default '[]'::jsonb,
    telemetry jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    check ((status = 'acknowledged') = (acknowledged_at is not null))
);

create index if not exists idx_advisor_recommendations_owner_portfolio_created
    on public.advisor_recommendations (owner_id, portfolio_id, created_at desc);

alter table public.advisor_recommendations enable row level security;
drop policy if exists "owners manage advisor recommendations" on public.advisor_recommendations;
create policy "owners manage advisor recommendations" on public.advisor_recommendations
    for all to authenticated
    using (owner_id = (select auth.uid()))
    with check (
        owner_id = (select auth.uid())
        and exists (
            select 1 from public.portfolios p
            where p.id = portfolio_id and p.owner_id = (select auth.uid())
        )
    );
