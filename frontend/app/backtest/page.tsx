"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AlertCircle, Play, RefreshCw } from "lucide-react";
import dynamic from "next/dynamic";
import Card from "@/components/Card";
import PageHeader from "@/components/PageHeader";
import MetricCard from "@/components/quant/MetricCard";
import { api, type ExecutionCostData, type RebalanceSimulationData } from "@/lib/api";
import { usePortfolios } from "@/components/PortfolioProvider";
import { loadAlignedPortfolioHistory, monthlyPricePanel } from "@/lib/portfolio-data";

import { drawdownPath } from "@/lib/portfolio-math";

const COST_BPS = 8;
const INITIAL_WEALTH = 1_000_000;
const chartLoading = () => <div className="h-full w-full animate-pulse rounded-xl bg-white/[0.025]" aria-hidden="true" />;
const PortfolioValueChart = dynamic(() => import("@/components/charts/BacktestCharts").then((module) => module.PortfolioValueChart), { ssr: false, loading: chartLoading });
const PortfolioDrawdownChart = dynamic(() => import("@/components/charts/BacktestCharts").then((module) => module.PortfolioDrawdownChart), { ssr: false, loading: chartLoading });
const TradingCostsChart = dynamic(() => import("@/components/charts/BacktestCharts").then((module) => module.TradingCostsChart), { ssr: false, loading: chartLoading });

function recordEntries(values: Record<string, number> | undefined, dates: string[]) {
  if (!values) return [];
  return Object.entries(values).sort(([left], [right]) => Number(left) - Number(right)).map(([period, value]) => {
    const index = Number(period);
    return { period: index < 0 ? dates[0] ?? "Start" : dates[index + 1] ?? `Period ${index + 1}`, value, rawPeriod: period };
  });
}

export default function BacktestPage() {
  const { activePortfolio, loading: portfoliosLoading } = usePortfolios();
  const [result, setResult] = useState<RebalanceSimulationData | null>(null);
  const [costs, setCosts] = useState<ExecutionCostData | null>(null);
  const [runMeta, setRunMeta] = useState<{ name: string; tickers: string[]; dates: string[]; observations: number } | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [costNote, setCostNote] = useState<string | null>(null);
  const [latency, setLatency] = useState<number | null>(null);
  const requestGeneration = useRef(0);
  const activeRequest = useRef<AbortController | null>(null);

  const run = useCallback(async () => {
    activeRequest.current?.abort();
    const controller = new AbortController();
    activeRequest.current = controller;
    const generation = ++requestGeneration.current;
    if (!activePortfolio) { setResult(null); setCosts(null); setRunMeta(null); setLoading(false); return; }
    setLoading(true); setError(null); setCostNote(null); setResult(null); setCosts(null); setRunMeta(null);
    try {
      const market = await loadAlignedPortfolioHistory(activePortfolio.tickers, controller.signal);
      if (generation !== requestGeneration.current) return;
      const sampledPrices = monthlyPricePanel(market.prices, market.periodsPerYear);
      const sampledDates = market.dates.filter((_, index) => index % Math.max(1, Math.round(market.periodsPerYear / 12)) === 0);
      const dates = [...sampledDates];
      if (dates.at(-1) !== market.dates.at(-1)) dates.push(market.dates.at(-1) as string);
      if (sampledPrices.length < 3) throw new Error("At least two monthly return periods are required for a meaningful backtest.");
      const initialWeights = activePortfolio.tickers.map((ticker) => activePortfolio.target_weights[ticker] ?? 0);
      const targets = Array.from({ length: sampledPrices.length - 1 }, () => [...initialWeights]);
      const request = {
        tickers: activePortfolio.tickers, initial_weights: initialWeights, target_weights: targets, prices: sampledPrices,
        initial_wealth: INITIAL_WEALTH, execution_cost_bps: COST_BPS, max_turnover: 0.35, periods_per_year: 12,
      };
      setRunMeta({ name: activePortfolio.name, tickers: activePortfolio.tickers, dates, observations: sampledPrices.length - 1 });
      const simulationResponse = await api.rebalanceSimulation(request, controller.signal);
      if (generation !== requestGeneration.current) return;
      setResult(simulationResponse.data);
      setLatency(simulationResponse.meta.execution_time_ms);

      try {
        const detailsByTicker = new Map((await api.stockDetails(activePortfolio.tickers, controller.signal)).map((detail) => [detail.stock.ticker, detail]));
        const details = activePortfolio.tickers.map((ticker) => detailsByTicker.get(ticker));
        if (details.some((detail) => !detail)) throw new Error("Some portfolio assets have no market metadata.");
        const completeDetails = details.filter((detail): detail is NonNullable<typeof detail> => Boolean(detail));
        const avgDailyValues = completeDetails.map((detail) => {
          const bars = detail.recent_prices.filter((bar) => bar.volume != null && bar.volume > 0 && bar.close > 0).slice(-20);
          return bars.length
            ? bars.reduce((sum, bar) => sum + bar.close * (bar.volume ?? 0), 0) / bars.length / (252 / market.periodsPerYear)
            : 0;
        });
        if (!avgDailyValues.every((value: any) => value > 0)) {
          setCostNote("The portfolio history is available, but trading volume is missing for one or more assets. Detailed spread and impact estimates are hidden.");
        } else {
          const currentWeights = [...initialWeights];
          for (let period = 0; period < sampledPrices.length - 2; period += 1) {
            const desired = initialWeights;
            const delta = desired.map((weight, index) => weight - currentWeights[index]);
            const grossTurnover = delta.reduce((sum, value) => sum + Math.abs(value), 0);
            const executed = grossTurnover > 0.35
              ? currentWeights.map((weight, index) => weight + delta[index] * 0.35 / grossTurnover)
              : [...desired];
            const assetReturns = sampledPrices[period + 1].map((price, index) => price / sampledPrices[period][index] - 1);
            const portfolioReturn = executed.reduce((sum, weight, index) => sum + weight * assetReturns[index], 0);
            for (let index = 0; index < currentWeights.length; index += 1) {
              currentWeights[index] = executed[index] * (1 + assetReturns[index]) / (1 + portfolioReturn);
            }
            const total = currentWeights.reduce((sum, weight) => sum + weight, 0);
            for (let index = 0; index < currentWeights.length; index += 1) currentWeights[index] /= total;
          }
          const delta = initialWeights.map((weight, index) => weight - currentWeights[index]);
          const lastCashFlow = simulationResponse.data.cash_flows.at(-1);
          const startingWealth = Number(lastCashFlow?.starting_wealth ?? INITIAL_WEALTH);
          const tradeSizes = delta.map((weight) => weight * startingWealth);
          const orderPrices = sampledPrices[sampledPrices.length - 2];
          const dailyVolatility = market.tickers.map((_, index) => Math.sqrt(Math.max(market.covariance[index][index], 0) / 252));
          const costResponse = await api.executionCosts({
            trade_sizes: tradeSizes,
            prices: orderPrices,
            advs: avgDailyValues,
            volatilities: dailyVolatility,
            spreads: market.tickers.map(() => 0.001),
            fixed_fee_per_trade: 1,
            pct_broker_fee: 0.0005,
          }, controller.signal);
          if (generation !== requestGeneration.current) return;
          setCosts(costResponse.data);
          setLatency(Math.max(simulationResponse.meta.execution_time_ms, costResponse.meta.execution_time_ms));
          setCostNote("Detailed friction breakdown estimates the final rebalance using recent trading value and a 10 bps spread assumption.");
        }
      } catch {
        if (generation === requestGeneration.current) setCostNote("The portfolio backtest ran, but recent trading volume was unavailable for the detailed market-impact estimate.");
      }
    } catch (cause) {
      if (generation === requestGeneration.current && !controller.signal.aborted) {
        setResult(null); setCosts(null);
        setError(cause instanceof Error ? cause.message : "Could not run a backtest for the selected portfolio.");
      }
    } finally { if (generation === requestGeneration.current && !controller.signal.aborted) setLoading(false); }
  }, [activePortfolio]);

  useEffect(() => {
    void run();
    return () => activeRequest.current?.abort();
  }, [run]);
  const dates = runMeta?.dates ?? [];
  const wealth = useMemo(() => recordEntries(result?.wealth_path, dates), [result, dates]);
  const pnl = useMemo(() => {
    let peak = INITIAL_WEALTH;
    return wealth.map((point) => {
      peak = Math.max(peak, point.value);
      return { period: point.period, pnl: point.value - INITIAL_WEALTH, wealth: point.value, drawdown: point.value / peak - 1 };
    });
  }, [wealth]);
  const cashflow = result?.cash_flows ?? [];
  const tradedNotional = cashflow.reduce((total, row) => total + Number(row.turnover ?? 0) * Number(row.starting_wealth ?? 0), 0);
  const totalCosts = Number(result?.total_transaction_cost ?? 0);
  const finalWealth = wealth.at(-1)?.value ?? INITIAL_WEALTH;
  const netReturn = finalWealth / INITIAL_WEALTH - 1;
  const maxDrawdown = Math.min(0, ...drawdownPath(wealth.map((point) => point.value), INITIAL_WEALTH));
  const periodReturns = wealth.slice(1).map((point, i) => point.value / wealth[i].value - 1);
  const average = periodReturns.length ? periodReturns.reduce((sum, value) => sum + value, 0) / periodReturns.length : 0;
  const std = periodReturns.length > 1 ? Math.sqrt(periodReturns.reduce((sum, value) => sum + (value - average) ** 2, 0) / (periodReturns.length - 1)) : 0;
  const sharpe = std ? (average / std) * Math.sqrt(12) : null;
  const downside = periodReturns.length ? Math.sqrt(periodReturns.reduce((sum, value) => sum + Math.min(value, 0) ** 2, 0) / periodReturns.length) * Math.sqrt(12) : 0;
  const sortino = downside ? (average * 12) / downside : null;
  const annualizedReturn = periodReturns.length ? Math.pow(Math.max(finalWealth / INITIAL_WEALTH, 0.000001), 12 / periodReturns.length) - 1 : 0;
  const calmar = maxDrawdown < 0 ? annualizedReturn / Math.abs(maxDrawdown) : null;
  const costBreakdown = costs?.cost_breakdown ?? {};
  const costBars = Object.entries(costBreakdown).map(([source, amount]) => ({ source: source.replaceAll("_", " "), amount }));

  return <div className="space-y-6">
    <PageHeader title="Backtest Tear Sheet" subtitle="Review how your selected portfolio's historical prices performed under a monthly rebalance-to-target rule." />
    <Card>
      <div className="flex flex-wrap items-center justify-between gap-4"><div><p className="text-sm font-semibold text-slate-100">{activePortfolio?.name ?? "No portfolio selected"}</p><p className="mt-1 text-xs text-slate-400">{runMeta ? `${runMeta.tickers.join(", ")} · ${runMeta.dates[0]} to ${runMeta.dates.at(-1)} · ${runMeta.observations} monthly periods` : "Uses stored prices and the saved target allocation"} · €1,000,000 normalized starting value · {COST_BPS} bps estimated rebalance cost</p></div><button onClick={() => void run()} disabled={loading || !activePortfolio} className="inline-flex items-center gap-2 rounded-lg bg-slate-900 px-4 py-2.5 text-sm font-semibold text-white hover:bg-slate-700 disabled:opacity-50 dark:bg-emerald-500 dark:text-slate-950"><Play size={14} />{loading ? "Running…" : "Run simulation"}</button></div>
      {costNote && <p className="mt-3 text-xs text-slate-400">{costNote}</p>}
      {!activePortfolio && !portfoliosLoading && <p className="mt-4 rounded-lg bg-amber-300/[0.05] p-3 text-xs text-amber-100">Create or select a saved portfolio before running a backtest.</p>}
      {error && <p role="alert" className="mt-4 flex items-start gap-2 rounded-lg bg-rose-50 p-3 text-xs text-rose-700 dark:bg-rose-500/10 dark:text-rose-300"><AlertCircle size={14} className="mt-0.5 shrink-0" />Backtest could not be completed: {error}</p>}
    </Card>
    <div className="grid grid-cols-12 gap-3">
      <div className="col-span-12 sm:col-span-6 xl:col-span-2"><MetricCard delay={0} label="Net return" value={result ? `${(netReturn * 100).toFixed(2)}%` : "N/A"} detail="After transaction costs" tone={netReturn >= 0 ? "positive" : "negative"} /></div>
      <div className="col-span-12 sm:col-span-6 xl:col-span-2"><MetricCard delay={0.04} label="Sharpe" value={result ? (sharpe?.toFixed(2) ?? "N/A") : "N/A"} detail="Annual figure from monthly returns" /></div>
      <div className="col-span-12 sm:col-span-6 xl:col-span-2"><MetricCard delay={0.08} label="Sortino" value={result ? (sortino?.toFixed(2) ?? "N/A") : "N/A"} detail="Focuses on negative returns" /></div>
      <div className="col-span-12 sm:col-span-6 xl:col-span-2"><MetricCard delay={0.12} label="Calmar" value={result ? (calmar?.toFixed(2) ?? "N/A") : "N/A"} detail="Annual growth vs. largest fall" /></div>
      <div className="col-span-12 sm:col-span-6 xl:col-span-4"><MetricCard delay={0.16} label="Max drawdown" value={result ? `${(maxDrawdown * 100).toFixed(2)}%` : "N/A"} detail={latency == null ? "Largest fall from a peak" : `Calculation took ${latency.toFixed(0)} ms`} tone="negative" /></div>
    </div>
    <div className="grid grid-cols-12 gap-5">
      <Card title="Portfolio value after costs" delay={0.05} className="col-span-12 xl:col-span-8">
        <div className="h-[320px]">{pnl.length ? <PortfolioValueChart data={pnl} /> : <EmptyPlot loading={loading} />}</div>
      </Card>
      <Card title="Portfolio drawdown" delay={0.11} className="col-span-12 xl:col-span-4">
        <div className="h-[320px]">{pnl.length ? <PortfolioDrawdownChart data={pnl} /> : <EmptyPlot loading={loading} />}</div>
      </Card>
    </div>
    <div className="grid grid-cols-12 gap-5">
      <Card title="Trading costs" delay={0.05} className="col-span-12 xl:col-span-7">
        <div className="grid gap-3 sm:grid-cols-3"><MetricCard label="Rebalancing costs" value={`€${totalCosts.toLocaleString("en-US", { maximumFractionDigits: 0 })}`} detail={`${(result?.total_transaction_drag_bps ?? 0).toFixed(1)} basis points`} tone="negative" /><MetricCard label="Estimated cost per trade" value={costs ? `€${costs.total_execution_drag.toLocaleString("en-US", { maximumFractionDigits: 0 })}` : "N/A"} detail={costs ? `${costs.total_friction_bps.toFixed(1)} basis points on traded amount` : "Cost estimate"} /><MetricCard label="Average yearly turnover" value={result ? `${(result.average_annual_turnover * 100).toFixed(0)}%` : "N/A"} detail={`${(result?.total_turnover ?? 0).toFixed(2)} total turnover`} tone="accent" /></div>
        <div className="mt-5 h-[190px]">{costBars.length ? <TradingCostsChart data={costBars} /> : <p className="flex h-full items-center justify-center text-sm text-slate-400">Run simulation for cost breakdown</p>}</div>
      </Card>
      <Card title="Simulation summary" delay={0.11} className="col-span-12 xl:col-span-5">
        <div className="space-y-3 text-sm">{[
          ["Starting value", "€1,000,000"], ["Ending value after costs", `€${finalWealth.toLocaleString("en-US", { maximumFractionDigits: 0 })}`], ["Total rebalancing costs", `€${totalCosts.toLocaleString("en-US", { maximumFractionDigits: 0 })}`], ["Total amount traded", `€${tradedNotional.toLocaleString("en-US", { maximumFractionDigits: 0 })}`], ["Months simulated", `${cashflow.length} months`], ["Calculation time", latency == null ? "N/A" : `${latency.toFixed(2)} ms`],
        ].map(([label, value]) => <div key={label} className="flex items-center justify-between border-b border-slate-100 pb-2.5 last:border-0 dark:border-slate-800"><span className="text-slate-500 dark:text-slate-400">{label}</span><span className="font-mono text-xs font-semibold text-slate-800 dark:text-slate-100">{value}</span></div>)}</div>
        <p className="mt-3 text-xs leading-relaxed text-slate-400">This is a historical simulation using stored prices, your saved target weights, and an 8 bps transaction-cost assumption. It is not a forecast and does not place trades.</p>
      </Card>
    </div>
  </div>;
}

function EmptyPlot({ loading }: { loading: boolean }) {
  return <div className="flex h-full items-center justify-center gap-2 text-sm text-slate-400"><RefreshCw size={14} className={loading ? "animate-spin" : ""} />{loading ? "Running backtest…" : "Run simulation to load chart"}</div>;
}
