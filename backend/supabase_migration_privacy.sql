-- Account-scoped deletion for persisted quant data.
-- Apply after the persistence and advisor migrations.

create or replace function public.delete_owner_quant_data(p_owner_id uuid)
returns jsonb
language plpgsql
security definer
set search_path = ''
as $$
declare
    v_portfolio_ids uuid[];
    v_backtests integer := 0;
    v_stress_tests integer := 0;
    v_portfolios integer := 0;
begin
    select coalesce(array_agg(p.id), array[]::uuid[])
      into v_portfolio_ids
      from public.portfolios as p
     where p.owner_id = p_owner_id;

    if cardinality(v_portfolio_ids) > 0 then
        delete from public.backtest_runs as b
         where b.portfolio_id = any(v_portfolio_ids);
        get diagnostics v_backtests = row_count;

        delete from public.stress_tests as s
         where s.portfolio_id = any(v_portfolio_ids);
        get diagnostics v_stress_tests = row_count;
    end if;

    -- Recommendations cascade from portfolio deletion.
    delete from public.portfolios as p where p.owner_id = p_owner_id;
    get diagnostics v_portfolios = row_count;

    return pg_catalog.jsonb_build_object(
        'portfolios_deleted', v_portfolios,
        'backtest_runs_deleted', v_backtests,
        'stress_tests_deleted', v_stress_tests
    );
end;
$$;

revoke all on function public.delete_owner_quant_data(uuid) from public, anon, authenticated;
grant execute on function public.delete_owner_quant_data(uuid) to service_role;
