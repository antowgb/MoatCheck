"use client";

import { useEffect, useMemo, useState } from "react";
import { ArrowUpRight, BriefcaseBusiness, Info, Plus, RefreshCw } from "lucide-react";
import Link from "next/link";
import Card from "@/components/Card";
import Disclaimer from "@/components/Disclaimer";
import PageHeader from "@/components/PageHeader";
import Skeleton from "@/components/Skeleton";
import StatTile from "@/components/StatTile";
import WorkspaceOnboarding from "@/components/WorkspaceOnboarding";
import { usePortfolios } from "@/components/PortfolioProvider";
import ScoreBadge from "@/components/ScoreBadge";
import { api, type Stock } from "@/lib/api";

const primaryAction = "inline-flex items-center justify-center gap-2 rounded-xl bg-emerald-300 px-3.5 py-2.5 text-xs font-semibold text-[#08110d] transition hover:bg-emerald-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-300";
const secondaryAction = "inline-flex items-center justify-center gap-2 rounded-xl border border-white/[0.09] bg-white/[0.025] px-3.5 py-2.5 text-xs font-medium text-slate-300 transition hover:border-white/[0.15] hover:bg-white/[0.055] hover:text-white";

function formatWeight(weight: number): string {
  return `${(weight * 100).toFixed(1)}%`;
}

export default function DashboardPage() {
  const [stocks, setStocks] = useState<Stock[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const { activePortfolio, loading: portfoliosLoading, error: portfolioError, refreshPortfolios } = usePortfolios();

  useEffect(() => {
    if (portfoliosLoading) return;
    let active = true;
    setError(null);
    if (!activePortfolio) {
      setStocks([]);
      return () => { active = false; };
    }
    setStocks(null);
    void api.stockDetails(activePortfolio.tickers).then((details) => {
      if (!active) return;
      const loaded: Stock[] = [];
      const failed: string[] = [];
      const detailsByTicker = new Map(details.map((detail) => [detail.stock.ticker, detail]));
      activePortfolio.tickers.forEach((ticker) => {
        const detail = detailsByTicker.get(ticker);
        if (!detail) { failed.push(ticker); return; }
        loaded.push({
          ...detail.stock,
          composite_score: detail.score?.composite_score ?? null,
          computed_at: detail.score?.computed_at ?? null,
        });
      });
      setStocks(loaded);
      if (failed.length) setError(`Could not load saved research for ${failed.join(", ")}. Open the stock page to retry.`);
    }).catch(() => { if (active) setError("Could not load saved research right now. Please retry in a moment."); });
    return () => { active = false; };
  }, [activePortfolio?.id, activePortfolio?.updated_at, activePortfolio?.tickers.join(","), portfoliosLoading]);

  const stockByTicker = useMemo(() => new Map((stocks ?? []).map((stock) => [stock.ticker, stock])), [stocks]);
  const positions = useMemo(() => {
    if (!activePortfolio) return [];
    return activePortfolio.tickers
      .map((ticker) => ({ ticker, weight: activePortfolio.target_weights[ticker] ?? 0, stock: stockByTicker.get(ticker) }))
      .sort((a, b) => b.weight - a.weight);
  }, [activePortfolio, stockByTicker]);

  const coverage = useMemo(() => {
    if (!stocks) return null;
    const active = stocks;
    const scored = active.filter((stock) => stock.composite_score != null);
    const avg = scored.length ? scored.reduce((sum, stock) => sum + (stock.composite_score ?? 0), 0) / scored.length : null;
    return { active: active.length, scored: scored.length, average: avg, pending: Math.max(0, (activePortfolio?.tickers.length ?? 0) - scored.length) };
  }, [stocks, activePortfolio?.tickers.length]);

  const largestPosition = positions[0];
  const weightTotal = positions.reduce((sum, position) => sum + position.weight, 0);

  return (
    <div className="w-full space-y-7">
      <PageHeader
        title="Portfolio dashboard"
        subtitle="See your saved portfolio, its largest positions, and the research behind them."
      />
      <Disclaimer />
      <WorkspaceOnboarding />

      {error && <p role="alert" className="rounded-xl border border-rose-300/15 bg-rose-300/[0.05] px-4 py-3 text-sm text-rose-200">Market coverage could not be loaded: {error}</p>}

      {portfoliosLoading && !activePortfolio ? (
        <Skeleton className="h-52" />
      ) : activePortfolio ? (
        <Card className="overflow-hidden !p-0">
          <div className="grid grid-cols-1 xl:grid-cols-[minmax(0,1.3fr)_minmax(340px,.7fr)]">
            <div className="p-5 sm:p-6">
              <div className="flex flex-wrap items-start justify-between gap-4">
                <div className="flex items-start gap-3.5">
                  <span className="grid h-11 w-11 shrink-0 place-items-center rounded-2xl border border-emerald-300/15 bg-emerald-300/[0.07] text-emerald-200"><BriefcaseBusiness size={19} /></span>
                  <div>
                    <p className="text-[10px] font-semibold uppercase tracking-[0.16em] text-slate-500">Active account portfolio</p>
                    <h2 className="mt-1 text-xl font-semibold tracking-tight text-slate-100">{activePortfolio.name}</h2>
                    <p className="mt-1 text-xs text-slate-500">Saved target allocation · {positions.length} {positions.length === 1 ? "asset" : "assets"} · updated {new Date(activePortfolio.updated_at).toLocaleDateString()}</p>
                  </div>
                </div>
                <div className="flex gap-2">
                  <Link href="/advisor" className={primaryAction}>Review Advisor <ArrowUpRight size={14} /></Link>
                  <Link href="/portfolio" className={secondaryAction}>Manage</Link>
                </div>
              </div>

              <div className="mt-7 grid grid-cols-2 gap-3 sm:grid-cols-4">
                <StatTile label="Portfolio assets" value={positions.length} />
                <StatTile label="Target allocation" value={formatWeight(weightTotal)} hint={Math.abs(weightTotal - 1) < 1e-6 ? "Fully allocated" : "Check saved weights"} />
                <StatTile label="Largest target" value={largestPosition ? formatWeight(largestPosition.weight) : "N/A"} hint={largestPosition?.ticker ?? "No positions"} />
                <StatTile label="Coverage scored" value={coverage ? `${positions.filter((p) => p.stock?.composite_score != null).length}/${positions.length}` : "N/A"} hint="Company and risk scores" />
              </div>
            </div>

            <div className="border-t border-white/[0.06] bg-black/[0.12] p-5 sm:p-6 xl:border-l xl:border-t-0">
              <div className="mb-4 flex items-center justify-between gap-3">
                <div><p className="text-[10px] font-semibold uppercase tracking-[0.15em] text-slate-500">Target allocation</p><p className="mt-1 text-xs text-slate-400">Saved weights for the active portfolio</p></div>
                <Link href="/portfolio" aria-label="Edit target allocation" className="grid h-8 w-8 place-items-center rounded-xl border border-white/[0.08] text-slate-400 transition hover:border-emerald-300/25 hover:text-emerald-200"><ArrowUpRight size={15} /></Link>
              </div>
              <div className="space-y-3">
                {positions.slice(0, 6).map(({ ticker, weight, stock }) => (
                  <div key={ticker}>
                    <div className="mb-1.5 flex items-center justify-between gap-3 text-xs">
                      <Link href={`/stock/?ticker=${ticker}`} className="font-mono font-semibold text-slate-200 hover:text-emerald-200">{ticker}<span className="ml-2 font-sans font-normal text-slate-500">{stock?.name ?? ""}</span></Link>
                      <span className="font-mono tabular-nums text-slate-300">{formatWeight(weight)}</span>
                    </div>
                    <div className="h-1.5 overflow-hidden rounded-full bg-white/[0.06]"><div className="h-full rounded-full bg-gradient-to-r from-emerald-500 to-emerald-300" style={{ width: `${Math.min(100, Math.max(0, weight * 100))}%` }} /></div>
                  </div>
                ))}
                {positions.length > 6 && <p className="pt-1 text-[10px] text-slate-500">Plus {positions.length - 6} assets · <Link href="/portfolio" className="text-emerald-200 hover:underline">view full portfolio</Link></p>}
                {positions.length === 0 && <p className="text-sm text-slate-500">This saved portfolio has no target weights yet.</p>}
              </div>
            </div>
          </div>
        </Card>
      ) : (
        <Card className="flex flex-col items-start justify-between gap-5 border-amber-300/10 bg-amber-300/[0.025] sm:flex-row sm:items-center">
          <div className="flex items-start gap-3.5">
            <span className="grid h-10 w-10 shrink-0 place-items-center rounded-2xl border border-amber-300/15 bg-amber-300/[0.06] text-amber-200"><BriefcaseBusiness size={17} /></span>
            <div><h2 className="text-base font-semibold text-slate-100">Start with your portfolio</h2><p className="mt-1 max-w-2xl text-xs leading-relaxed text-slate-400">{portfolioError ? "Sign in with your MoatCheck account to load saved portfolios." : "Build an asset list and save its target weights to your account. It will become the active portfolio here and in the Advisor."}</p></div>
          </div>
          <div className="flex shrink-0 gap-2">
            {portfolioError && <button type="button" onClick={() => void refreshPortfolios()} className={secondaryAction}><RefreshCw size={13} />Retry</button>}
            <Link href="/portfolio" className={primaryAction}><Plus size={14} />Set up portfolio</Link>
          </div>
        </Card>
      )}

      {activePortfolio && <div className="grid grid-cols-2 gap-3 xl:grid-cols-4">
        {coverage ? <>
          <StatTile label="Portfolio assets" value={activePortfolio.tickers.length} hint="Saved target positions" />
          <StatTile label="Scored positions" value={`${coverage.scored}/${activePortfolio.tickers.length}`} hint="Current company and risk score" />
          <StatTile label="Average score" value={coverage.average != null ? coverage.average.toFixed(1) : "N/A"} hint="Scored positions only" />
          <StatTile label="Research missing" value={coverage.pending} hint={coverage.pending ? "Price or score data is unavailable" : "All positions loaded"} />
        </> : Array.from({ length: 4 }).map((_, index) => <Skeleton key={index} className="h-[84px]" />)}
      </div>}

      {activePortfolio && <Card className="overflow-hidden !p-0">
        <div className="flex flex-wrap items-center justify-between gap-3 px-5 pt-5 sm:px-6">
          <div><h2 className="text-[10px] font-semibold uppercase tracking-[0.15em] text-slate-500">Portfolio positions</h2><p className="mt-1 text-xs text-slate-500">Saved target weights, with available research coverage</p></div>
          <Link href="/screener" className="inline-flex items-center gap-1 text-[11px] font-medium text-emerald-200 hover:text-emerald-100">Explore screener <ArrowUpRight size={13} /></Link>
        </div>
        <div className="mt-4 overflow-x-auto">
          <table className="min-w-full border-separate border-spacing-0 text-left">
            <thead><tr className="bg-white/[0.025] text-[9px] font-semibold uppercase tracking-[0.13em] text-slate-500"><th className="border-y border-white/[0.055] px-5 py-3 sm:px-6">Asset</th><th className="border-y border-white/[0.055] px-4 py-3">Sector</th><th className="border-y border-white/[0.055] px-4 py-3 text-right">Target weight</th><th className="border-y border-white/[0.055] px-4 py-3">Composite score</th><th className="border-y border-white/[0.055] px-5 py-3 sm:px-6">Data</th></tr></thead>
            <tbody>
              {positions.map(({ ticker, weight, stock }) => <tr key={ticker} className="transition hover:bg-white/[0.025]">
                <td className="border-b border-white/[0.045] px-5 py-3 sm:px-6"><Link href={`/stock/?ticker=${ticker}`} className="font-mono text-xs font-semibold text-emerald-200 hover:text-emerald-100">{ticker}<span className="ml-2 font-sans font-normal text-slate-400">{stock?.name ?? ""}</span></Link></td>
                <td className="border-b border-white/[0.045] px-4 py-3 text-xs text-slate-400">{stock?.sector ?? "N/A"}</td>
                <td className="border-b border-white/[0.045] px-4 py-3 text-right font-mono text-xs tabular-nums text-slate-200">{formatWeight(weight)}</td>
                <td className="border-b border-white/[0.045] px-4 py-3">{stock?.composite_score != null ? <ScoreBadge score={stock.composite_score} /> : stock ? <span className="text-xs text-slate-500">Not scored</span> : <span className="text-xs text-amber-300">Data unavailable</span>}</td>
                <td className="border-b border-white/[0.045] px-5 py-3 text-xs text-slate-500 sm:px-6">{stock?.updated_at ? new Date(stock.updated_at).toLocaleDateString() : "No stored data"}</td>
              </tr>)}
              {!positions.length && <tr><td colSpan={5} className="px-5 py-10 text-center text-sm text-slate-500">No saved positions in this portfolio.</td></tr>}
            </tbody>
          </table>
        </div>
      </Card>}

      {!activePortfolio && !portfoliosLoading && <Card className="flex flex-wrap items-center justify-between gap-4">
        <div><h2 className="text-sm font-semibold text-slate-100">Build your portfolio first</h2><p className="mt-1 text-xs text-slate-500">Use the screener to research companies, then save the assets you want to track.</p></div>
        <div className="flex gap-2"><Link href="/screener" className={secondaryAction}>Browse screener <ArrowUpRight size={14} /></Link><Link href="/portfolio" className={primaryAction}><Plus size={14} />Create portfolio</Link></div>
      </Card>}
      <p className="flex items-center gap-1.5 text-[10px] text-slate-600"><Info size={12} />Portfolio weights are saved targets, not broker holdings or live positions.</p>
    </div>
  );
}
