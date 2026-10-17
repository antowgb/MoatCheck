"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import dynamic from "next/dynamic";
import { Activity, RefreshCw, SlidersHorizontal, Sparkles } from "lucide-react";
import Card from "@/components/Card";
import PageHeader from "@/components/PageHeader";
import MetricCard from "@/components/quant/MetricCard";
import { api, type QuantFrontierPoint } from "@/lib/api";
import { loadAlignedPortfolioHistory } from "@/lib/portfolio-data";
import { usePortfolios } from "@/components/PortfolioProvider";

const COLORS = ["#22C55E", "#38bdf8", "#fbbf24", "#8B5CF6"];
const chartLoading = () => <div className="h-full w-full animate-pulse rounded-xl bg-white/[0.025]" aria-hidden="true" />;
const FrontierScatterChart = dynamic(() => import("@/components/charts/OptimizerCharts").then((module) => module.FrontierScatterChart), { ssr: false, loading: chartLoading });
const AllocationPieChart = dynamic(() => import("@/components/charts/OptimizerCharts").then((module) => module.AllocationPieChart), { ssr: false, loading: chartLoading });
const AllocationBarChart = dynamic(() => import("@/components/charts/OptimizerCharts").then((module) => module.AllocationBarChart), { ssr: false, loading: chartLoading });

export default function OptimizerPage() {
  const { activePortfolio, loading: portfoliosLoading } = usePortfolios();
  const [frontier, setFrontier] = useState<QuantFrontierPoint[]>([]);
  const [maxSharpe, setMaxSharpe] = useState<QuantFrontierPoint | null>(null);
  const [minimumVariance, setMinimumVariance] = useState<QuantFrontierPoint | null>(null);
  const [targetReturn, setTargetReturn] = useState(0);
  const [riskLimit, setRiskLimit] = useState(0.25);
  const [useAutomaticRecommendation, setUseAutomaticRecommendation] = useState(true);
  const [holdCash, setHoldCash] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [latency, setLatency] = useState<number | null>(null);
  const [dataWindow, setDataWindow] = useState<{ start: string; end: string; observations: number; frequency: number } | null>(null);
  const requestGeneration = useRef(0);
  const activeRequest = useRef<AbortController | null>(null);

  const refresh = useCallback(async () => {
    activeRequest.current?.abort();
    const controller = new AbortController();
    activeRequest.current = controller;
    const generation = ++requestGeneration.current;
    if (!activePortfolio) {
      setFrontier([]); setMaxSharpe(null); setMinimumVariance(null); setDataWindow(null);
      setError(null);
      setLoading(false);
      return;
    }
    setLoading(true);
    setError(null);
    setFrontier([]); setMaxSharpe(null); setMinimumVariance(null); setDataWindow(null); setLatency(null);
    try {
      const market = await loadAlignedPortfolioHistory(activePortfolio.tickers, controller.signal);
      if (generation !== requestGeneration.current) return;
      setDataWindow({ start: market.dates[0], end: market.dates.at(-1) ?? market.dates[0], observations: market.returns.length, frequency: market.periodsPerYear });
      const input = { tickers: market.tickers, expected_returns: market.expectedReturns, covariance_matrix: market.covariance, risk_free_rate: 0.03, points: 24 };
      if (market.tickers.length === 1) {
        const variance = Math.max(0, market.covariance[0][0]);
        const only = { expected_return: market.expectedReturns[0], volatility: Math.sqrt(variance), sharpe_ratio: (market.expectedReturns[0] - 0.03) / Math.sqrt(variance || Number.EPSILON), weights: { [market.tickers[0]]: 1 }, success: true };
        setFrontier([only]); setMaxSharpe(only); setMinimumVariance(only); setLatency(0);
        return;
      }
      const [frontierResult, sharpeResult, minimumVarianceResult] = await Promise.all([
        api.optimizeFrontier(input, controller.signal),
        api.optimizeMaxSharpe(input, controller.signal),
        api.optimizeMinVariance(input, controller.signal),
      ]);
      if (generation !== requestGeneration.current) return;
      const points = frontierResult.data.frontier.filter((point) => point.expected_return != null);
      if (!points.length) throw new Error("The optimizer returned an empty frontier.");
      const optimal = sharpeResult.data;
      setFrontier(points);
      setMaxSharpe({
        expected_return: optimal.expected_return ?? points[points.length - 1].expected_return,
        volatility: optimal.volatility ?? points[points.length - 1].volatility,
        sharpe_ratio: optimal.sharpe_ratio ?? null,
        weights: optimal.weights ?? points[points.length - 1].weights,
        success: true,
      });
      setMinimumVariance({
        expected_return: minimumVarianceResult.data.expected_return ?? null,
        volatility: minimumVarianceResult.data.volatility ?? Infinity,
        sharpe_ratio: minimumVarianceResult.data.sharpe_ratio ?? null,
        weights: minimumVarianceResult.data.weights ?? {},
        success: true,
      });
      setUseAutomaticRecommendation(true);
      setLatency(Math.max(frontierResult.meta.execution_time_ms, sharpeResult.meta.execution_time_ms, minimumVarianceResult.meta.execution_time_ms));
    } catch (cause) {
      if (generation === requestGeneration.current && !controller.signal.aborted) setError(cause instanceof Error ? cause.message : "Quant service unavailable");
    } finally {
      if (generation === requestGeneration.current && !controller.signal.aborted) setLoading(false);
    }
  }, [activePortfolio]);

  useEffect(() => {
    void refresh();
    return () => activeRequest.current?.abort();
  }, [refresh]);

  const feasible = useMemo(() => frontier.filter((point) => point.volatility <= riskLimit), [frontier, riskLimit]);
  const feasibleCandidates = useMemo(() => {
    const candidates = [...feasible];
    if (minimumVariance && minimumVariance.volatility <= riskLimit) candidates.push(minimumVariance);
    return candidates;
  }, [feasible, minimumVariance, riskLimit]);
  const automaticRecommendation = useMemo(() => {
    if (!feasibleCandidates.length) return null;
    if (maxSharpe && maxSharpe.volatility <= riskLimit) return maxSharpe;
    return feasibleCandidates.reduce((best, point) =>
      (point.sharpe_ratio ?? -Infinity) > (best.sharpe_ratio ?? -Infinity) ? point : best,
    feasibleCandidates[0]);
  }, [feasibleCandidates, maxSharpe, riskLimit]);
  const manualSelection = useMemo(() => {
    if (!feasibleCandidates.length) return null;
    return feasibleCandidates.reduce((best, point) =>
      Math.abs((point.expected_return ?? 0) - targetReturn) < Math.abs((best.expected_return ?? 0) - targetReturn) ? point : best,
    feasibleCandidates[0]);
  }, [feasibleCandidates, targetReturn]);
  const selected = useAutomaticRecommendation ? automaticRecommendation : manualSelection;
  const weights = useMemo(() => {
    const base = selected?.weights ?? {};
    const scaled = Object.entries(base).map(([name, weight]) => ({ name, weight: weight * (holdCash ? 0.5 : 1) }));
    if (holdCash && selected) scaled.push({ name: "Cash", weight: 0.5 });
    return scaled;
  }, [holdCash, selected]);
  const displayedReturn = selected?.expected_return == null ? null : selected.expected_return * (holdCash ? 0.5 : 1);
  const displayedVolatility = selected ? selected.volatility * (holdCash ? 0.5 : 1) : null;
  const displayedSharpe = displayedReturn != null && displayedVolatility != null && displayedVolatility > 0
    ? (displayedReturn - 0.03) / displayedVolatility : null;
  const frontierChart = frontier.map((point) => ({ ...point, returnPct: (point.expected_return ?? 0) * 100 }));
  const minReturn = frontier.length ? Math.min(...frontier.map((point) => (point.expected_return ?? 0) * 100)) : 0;
  const maxReturn = frontier.length ? Math.max(...frontier.map((point) => (point.expected_return ?? 0) * 100)) : 0;
  const safeMaxReturn = Math.max(maxReturn, minReturn + 0.1);
  const maxRisk = Math.max(0.1, Math.ceil(Math.max(...frontier.map((point) => point.volatility), 0.1) * 100 / 5) * 5 / 100);

  return (
    <div className="space-y-6">
      <PageHeader title="Portfolio Optimizer" subtitle={activePortfolio ? `Explore long-term allocations for ${activePortfolio.name} using its saved assets and stored price history.` : "Choose or create a portfolio to optimize its allocation."} />

      {!activePortfolio && !portfoliosLoading && <div className="rounded-2xl border border-amber-300/15 bg-amber-300/[0.05] p-5 text-sm text-amber-100">No saved portfolio is selected. Create one in My Portfolio to see calculations based on your assets.</div>}
      {activePortfolio && <div className="flex flex-wrap items-center justify-between gap-2 rounded-2xl border border-white/[0.07] bg-white/[0.025] px-4 py-3 text-xs text-slate-400"><span>Active portfolio: <b className="text-slate-200">{activePortfolio.name}</b> · {activePortfolio.tickers.join(", ")}</span><span>{dataWindow ? `${dataWindow.start} to ${dataWindow.end} · ${dataWindow.observations} return observations · about ${dataWindow.frequency} periods/year` : "Loading portfolio price history…"}</span></div>}

      <div className="grid grid-cols-12 gap-3">
        <div className="col-span-12 sm:col-span-6 xl:col-span-3"><MetricCard label="Expected return" value={displayedReturn == null ? "N/A" : `${(displayedReturn * 100).toFixed(2)}%`} detail={holdCash ? "Includes 50% cash, assumed to earn 0%" : "Annualized arithmetic return estimate"} tone="positive" /></div>
        <div className="col-span-12 sm:col-span-6 xl:col-span-3"><MetricCard label="Annualized volatility" value={displayedVolatility == null ? "N/A" : `${(displayedVolatility * 100).toFixed(2)}%`} detail={`Risk ceiling ${(riskLimit * 100).toFixed(0)}%`} tone="accent" /></div>
        <div className="col-span-12 sm:col-span-6 xl:col-span-3"><MetricCard label="Sharpe ratio" value={displayedSharpe?.toFixed(2) ?? "N/A"} detail="Risk-free rate 3.0%" /></div>
        <div className="col-span-12 sm:col-span-6 xl:col-span-3"><MetricCard label="Frontier points" value={`${frontier.length}`} detail={latency == null ? "Waiting for portfolio data" : `Engine · ${latency.toFixed(0)} ms`} /></div>
      </div>

      <div className="grid grid-cols-12 gap-5">
        <Card title="Return and risk" delay={0.04} className="col-span-12 min-w-0 xl:col-span-8">
          <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
            <div className="flex items-center gap-2 text-xs text-slate-500 dark:text-slate-400">
              <span className="inline-block h-2 w-2 rounded-full bg-emerald-400" /> Within your risk limit
              <span className="ml-3 inline-block h-2 w-2 rounded-full bg-amber-400" /> Unconstrained max Sharpe
            </div>
              <span className="rounded-full bg-emerald-400/10 px-2.5 py-1 text-[10px] font-semibold uppercase tracking-wider text-emerald-200">
              {frontier.length ? "Selected portfolio" : "No result"}
            </span>
          </div>
          <div className="h-[330px] w-full text-slate-400">
            <FrontierScatterChart frontier={frontierChart} maxSharpe={maxSharpe} minimumVariance={minimumVariance} recommendation={automaticRecommendation} riskLimit={riskLimit} />
          </div>
          <p className="mt-2 text-xs text-slate-400 dark:text-slate-500">Each dot shows a portfolio option. The red line marks your risk limit.</p>
        </Card>

        <Card title="Allocation controls" delay={0.1} className="col-span-12 h-fit xl:col-span-4">
          <div className="space-y-6">
            <div className="rounded-xl border border-cyan-300/15 bg-cyan-300/[0.045] p-3.5">
              <div className="flex items-start justify-between gap-3">
                <div>
                  <p className="text-[10px] font-semibold uppercase tracking-[0.14em] text-cyan-200">Automatic starting point</p>
                  {automaticRecommendation ? <p className="mt-1 text-xs leading-relaxed text-slate-300">Highest estimated Sharpe ratio among allocations that meet your volatility ceiling. Long-only, based on historical prices.</p> : <p className="mt-1 text-xs leading-relaxed text-amber-200">No allocation in this sample meets the current volatility ceiling. Raise the ceiling or review the asset mix.</p>}
                </div>
                <button type="button" disabled={!automaticRecommendation} onClick={() => setUseAutomaticRecommendation(true)} className={`shrink-0 rounded-lg px-2.5 py-1.5 text-[10px] font-semibold transition ${useAutomaticRecommendation ? "bg-cyan-300/15 text-cyan-100" : "bg-white/[0.06] text-slate-300 hover:bg-white/[0.1]"} disabled:opacity-40`}>{useAutomaticRecommendation ? "In use" : "Use this"}</button>
              </div>
              {automaticRecommendation && <p className="mt-2 font-mono text-[11px] text-cyan-100">Return {(100 * (automaticRecommendation.expected_return ?? 0)).toFixed(1)}% · Volatility {(100 * automaticRecommendation.volatility).toFixed(1)}% · Sharpe {automaticRecommendation.sharpe_ratio?.toFixed(2) ?? "N/A"}</p>}
              {useAutomaticRecommendation && <p className="mt-2 text-[10px] text-slate-500">Adjust the return slider below to switch to a manual what-if.</p>}
              {activePortfolio && automaticRecommendation && <div className="mt-3 border-t border-white/[0.06] pt-3"><p className="mb-2 text-[10px] font-medium uppercase tracking-wider text-slate-500">Saved target vs model weights</p><div className="space-y-1.5"><div className="grid grid-cols-[1fr_auto_auto] gap-4 text-[9px] text-slate-600"><span>Asset</span><span>Saved</span><span>Suggested</span></div>{activePortfolio.tickers.map((ticker) => <div key={ticker} className="grid grid-cols-[1fr_auto_auto] gap-4 font-mono text-[10px]"><span className="text-slate-400">{ticker}</span><span className="text-slate-500">{((activePortfolio.target_weights[ticker] ?? 0) * 100).toFixed(1)}%</span><span className="text-cyan-100">{((automaticRecommendation.weights?.[ticker] ?? 0) * 100).toFixed(1)}%</span></div>)}</div></div>}
            </div>
            <label className="block">
              <span className="flex justify-between text-sm font-medium text-slate-700 dark:text-slate-200"><span>Manual target return</span><span className="font-mono text-emerald-600 dark:text-emerald-400">{(((useAutomaticRecommendation ? automaticRecommendation?.expected_return : targetReturn) ?? 0) * 100).toFixed(1)}%</span></span>
              <input aria-label="Manual target annual return" disabled={!frontier.length} className="mt-4 w-full accent-emerald-500 disabled:opacity-40" type="range" min={minReturn} max={safeMaxReturn} step="0.1" value={Math.min(Math.max(((useAutomaticRecommendation ? automaticRecommendation?.expected_return : targetReturn) ?? 0) * 100, minReturn), safeMaxReturn)} onChange={(e) => { setTargetReturn(Number(e.target.value) / 100); setUseAutomaticRecommendation(false); }} />
              <span className="mt-1 flex justify-between text-[10px] text-slate-400"><span>{minReturn.toFixed(1)}%</span><span>{maxReturn.toFixed(1)}%</span></span>
            </label>
            <label className="block">
              <span className="flex justify-between text-sm font-medium text-slate-700 dark:text-slate-200"><span>Volatility ceiling</span><span className="font-mono text-rose-500">{(riskLimit * 100).toFixed(0)}%</span></span>
              <input aria-label="Maximum portfolio volatility" disabled={!frontier.length} className="mt-4 w-full accent-rose-400 disabled:opacity-40" type="range" min="0" max={maxRisk * 100} step="1" value={Math.min(riskLimit * 100, maxRisk * 100)} onChange={(e) => setRiskLimit(Number(e.target.value) / 100)} />
              <span className="mt-1 flex justify-between text-[10px] text-slate-400"><span>0%</span><span>{(maxRisk * 100).toFixed(0)}%</span></span>
            </label>
            <div className="flex items-center justify-between rounded-xl border border-slate-200 p-3 dark:border-slate-800">
              <div className="flex items-center gap-2.5">
                <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-violet-50 text-violet-600 dark:bg-violet-400/10 dark:text-violet-300"><Sparkles size={15} /></div>
                <div><p className="text-sm font-medium text-slate-800 dark:text-slate-100">Hold 50% in cash</p><p className="text-[11px] text-slate-400">Scale each asset weight by half; cash earns 0%</p></div>
              </div>
              <button type="button" role="switch" aria-checked={holdCash} onClick={() => setHoldCash((value: any) => !value)} className={`relative h-6 w-11 rounded-full transition ${holdCash ? "bg-violet-500" : "bg-slate-300 dark:bg-slate-700"}`}><span className={`absolute top-0.5 h-5 w-5 rounded-full bg-white shadow transition ${holdCash ? "left-[22px]" : "left-0.5"}`} /></button>
            </div>
            <button onClick={() => void refresh()} disabled={loading} className="inline-flex w-full items-center justify-center gap-2 rounded-lg bg-slate-900 px-4 py-2.5 text-sm font-semibold text-white transition hover:bg-slate-700 disabled:opacity-60 dark:bg-emerald-500 dark:text-slate-950 dark:hover:bg-emerald-400">
              <RefreshCw size={15} className={loading ? "animate-spin" : ""} /> {loading ? "Optimizing…" : "Update portfolio options"}
            </button>
            {error && <p role="status" className="text-xs leading-relaxed text-amber-300">Could not calculate this portfolio from stored price history. {error}</p>}
            {feasibleCandidates.length === 0 && <p className="text-xs text-rose-300">No option meets this volatility ceiling. No out-of-limit weights are shown as a recommendation.</p>}
          </div>
        </Card>
      </div>

      <div className="grid grid-cols-12 gap-5">
        <Card title={useAutomaticRecommendation ? "Automatic allocation" : "Manual allocation preview"} delay={0.04} className="col-span-12 lg:col-span-6">
          <div className="grid items-center gap-3 sm:grid-cols-[190px_1fr]">
            <div className="h-[210px]">
              <AllocationPieChart data={weights} />
            </div>
            <div className="space-y-3">
              {weights.map((item, index) => <div key={item.name}><div className="mb-1 flex justify-between text-xs"><span className="text-slate-500 dark:text-slate-400">{item.name}</span><span className="font-mono font-semibold text-slate-800 dark:text-slate-100">{(item.weight * 100).toFixed(1)}%</span></div><div className="h-1.5 rounded-full bg-slate-100 dark:bg-slate-800"><div className="h-full rounded-full" style={{ width: `${Math.min(item.weight * 100, 100)}%`, background: COLORS[index % COLORS.length] }} /></div></div>)}
            </div>
          </div>
        </Card>
        <Card title={useAutomaticRecommendation ? "Recommended weights" : "What-if weights"} delay={0.1} className="col-span-12 lg:col-span-6">
          <div className="h-[250px] text-slate-400">
            <AllocationBarChart data={weights} />
          </div>
          <div className="mt-2 flex items-center gap-2 text-xs text-slate-400"><SlidersHorizontal size={13} /> {holdCash ? "This is a simple 50% cash overlay, not a Kelly calculation." : "All funds are invested. Short selling is not allowed."}</div>
        </Card>
      </div>

      <div className="flex items-start gap-2 rounded-xl border border-sky-300/10 bg-sky-300/[0.04] px-4 py-3 text-xs leading-relaxed text-sky-100"><Activity size={14} className="mt-0.5 shrink-0" /><p>These estimates use historical prices for your selected assets. They describe possible long-term risk and return, not guaranteed results. Expected returns are backward-looking estimates.</p></div>
    </div>
  );
}
