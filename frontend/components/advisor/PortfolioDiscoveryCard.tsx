"use client";

import { useState } from "react";
import { ArrowRight, Check, ChevronRight, Compass, LoaderCircle, ShieldAlert, Sparkles } from "lucide-react";
import Card from "@/components/Card";
import { api, type DiscoveryPortfolioMetrics, type PortfolioDiscoveryData, type DiscoverySwapProposal } from "@/lib/api";

function percent(value: number | null | undefined): string {
  return Number.isFinite(value) ? `${(Number(value) * 100).toFixed(1)}%` : "—";
}

function metric(value: number | null | undefined): string {
  return Number.isFinite(value) ? Number(value).toFixed(2) : "—";
}

function ModelPortfolio({ title, value, onApply }: { title: string; value: DiscoveryPortfolioMetrics; onApply: () => void }) {
  const projection = value.long_term_projection_per_unit?.find((item) => item.years === 5);
  return <div className="rounded-2xl border border-white/[0.07] bg-black/20 p-4">
    <div className="flex items-start justify-between gap-3"><div><p className="text-xs font-semibold text-slate-100">{title}</p><p className="mt-1 text-[10px] text-slate-500">Three-year historical estimate</p></div><Sparkles size={15} className="text-cyan-300" /></div>
    <div className="mt-4 grid grid-cols-3 gap-2">
      <div><p className="text-[9px] uppercase text-slate-500">Return</p><p className="mt-1 font-mono text-sm text-emerald-200">{percent(value.expected_return)}</p></div>
      <div><p className="text-[9px] uppercase text-slate-500">Volatility</p><p className="mt-1 font-mono text-sm text-slate-200">{percent(value.volatility)}</p></div>
      <div><p className="text-[9px] uppercase text-slate-500">Sharpe</p><p className="mt-1 font-mono text-sm text-cyan-200">{metric(value.sharpe_ratio)}</p></div>
    </div>
    {projection && <div className="mt-4 rounded-xl border border-white/[0.06] bg-white/[0.02] p-3"><p className="text-[9px] uppercase tracking-wide text-slate-500">5-year value estimate per €1, parametric</p><div className="mt-2 flex justify-between gap-2 font-mono text-xs"><span className="text-amber-200">10th {projection.p10_value_per_unit.toFixed(2)}</span><span className="text-slate-200">50th {projection.p50_value_per_unit.toFixed(2)}</span><span className="text-emerald-200">90th {projection.p90_value_per_unit.toFixed(2)}</span></div></div>}
    <button type="button" onClick={onApply} disabled={!value.weights} className="mt-4 inline-flex w-full items-center justify-center gap-2 rounded-lg border border-white/10 px-3 py-2 text-xs font-semibold text-slate-200 hover:border-emerald-300/30 hover:text-emerald-200 disabled:opacity-40">Review this allocation<ChevronRight size={14} /></button>
  </div>;
}

export default function PortfolioDiscoveryCard({ portfolioId, onApplied }: { portfolioId: string; onApplied?: () => void }) {
  const [result, setResult] = useState<PortfolioDiscoveryData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<{ id: string; weights: Record<string, number>; title: string } | null>(null);
  const [applying, setApplying] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);

  async function discover() {
    setLoading(true); setError(null); setNotice(null);
    try { setResult((await api.discoverPortfolioProposals(portfolioId)).data); }
    catch (cause) { setError(cause instanceof Error ? cause.message : "Could not load portfolio discovery."); }
    finally { setLoading(false); }
  }

  async function apply() {
    if (!selected) return;
    setApplying(true); setError(null);
    try {
      await api.applyPortfolioProposal(portfolioId, selected.id, selected.weights);
      setNotice(`${selected.title} is now saved as the portfolio target. No orders were placed.`);
      setSelected(null); onApplied?.();
    } catch (cause) { setError(cause instanceof Error ? cause.message : "Could not save this allocation."); }
    finally { setApplying(false); }
  }

  const swaps = result?.swap_proposals ?? [];
  return <Card title="Portfolio Discovery" className="col-span-12 overflow-visible border-cyan-300/10">
    <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
      <div className="flex items-start gap-3"><span className="rounded-xl border border-cyan-300/15 bg-cyan-300/[0.07] p-2.5 text-cyan-200"><Compass size={17} /></span><div><p className="text-sm font-semibold text-slate-100">Compare your holdings with liquid core assets</p><p className="mt-1 max-w-3xl text-xs leading-relaxed text-slate-500">Reviews up to three years of saved price history for up to 250 core assets ranked by stored liquidity, plus your holdings. It tests replacement ideas against portfolio return and diversification, and builds two long-only alternatives. Estimates can be noisy and never place trades.</p></div></div>
      <button type="button" onClick={() => void discover()} disabled={loading} className="inline-flex shrink-0 items-center justify-center gap-2 rounded-lg bg-cyan-300 px-4 py-2.5 text-xs font-semibold text-slate-950 hover:bg-cyan-200 disabled:opacity-50">{loading ? <LoaderCircle size={14} className="animate-spin" /> : <Compass size={14} />}{loading ? "Reviewing price history" : result ? "Refresh discovery" : "Find portfolio ideas"}</button>
    </div>
    {error && <p role="alert" className="mt-4 rounded-lg border border-rose-300/15 bg-rose-300/[0.05] px-3 py-2 text-xs text-rose-200">{error}</p>}
    {notice && <p role="status" className="mt-4 rounded-lg border border-emerald-300/15 bg-emerald-300/[0.05] px-3 py-2 text-xs text-emerald-200">{notice}</p>}
    {result && <div className="mt-5 space-y-5">
      <div className="flex flex-wrap items-center justify-between gap-2 rounded-xl border border-white/[0.06] bg-black/15 px-3 py-2 text-[10px] text-slate-500"><span>{result.universe_assets_evaluated.toLocaleString()} assets · {result.observations.toLocaleString()} dated return rows · as of {result.as_of}</span><span>Calculation {result.telemetry.execution_time_ms.toFixed(1)} ms</span></div>
      <div><div className="mb-3"><p className="text-[10px] font-semibold uppercase tracking-[0.14em] text-cyan-300">Asset swap suggestions</p><p className="mt-1 text-[10px] text-slate-500">A suggestion must improve estimated portfolio Sharpe and lower estimated volatility, while improving diversification against remaining holdings.</p></div>
        {swaps.length ? <div className="grid gap-3 lg:grid-cols-2">{swaps.map((proposal: DiscoverySwapProposal) => { const improvement = proposal.sharpe_ratio - (result.current_portfolio.sharpe_ratio ?? proposal.sharpe_ratio); return <div key={proposal.proposal_id} className="rounded-2xl border border-white/[0.07] bg-black/20 p-4"><div className="flex flex-wrap items-center justify-between gap-2"><p className="text-xs font-semibold text-slate-100">{proposal.swaps.map((swap) => `${swap.replace_ticker} → ${swap.with_ticker}`).join(" · ")}</p><span className="rounded-full border border-emerald-300/15 bg-emerald-300/[0.06] px-2 py-1 text-[10px] text-emerald-200">{proposal.swap_count} asset{proposal.swap_count > 1 ? "s" : ""}</span></div><div className="mt-3 flex flex-wrap items-center gap-3 font-mono text-xs"><span className="text-slate-500">Portfolio Sharpe {result.current_portfolio.sharpe_ratio?.toFixed(2) ?? "—"}</span><ArrowRight size={13} className="text-slate-600" /><span className="text-emerald-200">{proposal.sharpe_ratio.toFixed(2)} ({improvement >= 0 ? "+" : ""}{improvement.toFixed(2)})</span><span className="ml-auto text-slate-500">Volatility {percent(result.current_portfolio.volatility)} → {percent(proposal.volatility)}</span></div><button type="button" onClick={() => setSelected({ id: proposal.proposal_id, weights: proposal.target_weights, title: "Asset swap suggestion" })} className="mt-4 w-full rounded-lg border border-white/10 px-3 py-2 text-xs font-semibold text-slate-200 hover:border-emerald-300/30">Review swap proposal</button></div>; })}</div> : <div className="rounded-xl border border-white/[0.06] px-4 py-5 text-xs text-slate-500">No replacement passed the strict return, diversification, and portfolio-risk checks. That is a valid result.</div>}
      </div>
      <div><p className="mb-3 text-[10px] font-semibold uppercase tracking-[0.14em] text-cyan-300">Alternative model portfolios</p><div className="grid gap-3 lg:grid-cols-2"><ModelPortfolio title="Maximum Sharpe" value={result.model_portfolios.maximum_sharpe} onApply={() => { const model = result.model_portfolios.maximum_sharpe; if (model.weights) setSelected({ id: "model:maximum-sharpe", weights: model.weights, title: "Maximum Sharpe model" }); }} /><ModelPortfolio title="Minimum risk" value={result.model_portfolios.minimum_risk} onApply={() => { const model = result.model_portfolios.minimum_risk; if (model.weights) setSelected({ id: "model:minimum-risk", weights: model.weights, title: "Minimum risk model" }); }} /></div></div>
      <p className="text-[10px] leading-relaxed text-slate-600">Projection ranges use a log-normal return approximation based on historical annualized estimates. They are scenarios, not forecasts or guarantees. Models use long-only weights, and short or incomplete histories may reduce the screened universe.</p>
    </div>}
    {selected && <div className="fixed inset-0 z-[100] flex items-end justify-center bg-black/70 p-4 backdrop-blur-sm sm:items-center" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget && !applying) setSelected(null); }}><section role="dialog" aria-modal="true" aria-labelledby="proposal-confirm-title" className="w-full max-w-lg rounded-3xl border border-white/10 bg-[#111720] p-5 shadow-2xl"><div className="flex items-start gap-3"><span className="rounded-xl bg-amber-300/[0.08] p-2 text-amber-200"><ShieldAlert size={18} /></span><div><h3 id="proposal-confirm-title" className="text-sm font-semibold text-slate-100">Save this target allocation?</h3><p className="mt-2 text-xs leading-relaxed text-slate-400">This changes the saved portfolio tickers and target weights. It does not trade or move money. Review the new weights before continuing.</p></div></div><div className="mt-4 max-h-48 space-y-2 overflow-auto rounded-xl border border-white/[0.06] bg-black/20 p-3">{Object.entries(selected.weights).sort((a, b) => b[1] - a[1]).map(([ticker, weight]) => <div key={ticker} className="flex justify-between font-mono text-xs"><span className="text-slate-300">{ticker}</span><span className="text-emerald-200">{(weight * 100).toFixed(2)}%</span></div>)}</div><div className="mt-5 flex justify-end gap-2"><button type="button" onClick={() => setSelected(null)} disabled={applying} className="rounded-lg border border-white/10 px-3 py-2 text-xs text-slate-300">Cancel</button><button type="button" onClick={() => void apply()} disabled={applying} className="inline-flex items-center gap-2 rounded-lg bg-emerald-300 px-3 py-2 text-xs font-semibold text-slate-950 disabled:opacity-50">{applying ? <LoaderCircle size={13} className="animate-spin" /> : <Check size={13} />}{applying ? "Saving" : "Save target weights"}</button></div></section></div>}
  </Card>;
}
