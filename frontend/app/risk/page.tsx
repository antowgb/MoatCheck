"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Activity, AlertTriangle, RefreshCw } from "lucide-react";
import dynamic from "next/dynamic";
import Card from "@/components/Card";
import PageHeader from "@/components/PageHeader";
import MetricCard from "@/components/quant/MetricCard";
import { api, type CrisisScenarioCatalogEntry, type HistoricalCrisisReplayData, type PortfolioRiskData } from "@/lib/api";
import { usePortfolios } from "@/components/PortfolioProvider";
import { loadAlignedPortfolioHistory, type AlignedPortfolioHistory } from "@/lib/portfolio-data";

const chartLoading = () => <div className="h-full w-full animate-pulse rounded-xl bg-white/[0.025]" aria-hidden="true" />;
const ReturnDistributionChart = dynamic(() => import("@/components/charts/RiskCharts").then((module) => module.ReturnDistributionChart), { ssr: false, loading: chartLoading });
const CrisisDrawdownChart = dynamic(() => import("@/components/charts/RiskCharts").then((module) => module.CrisisDrawdownChart), { ssr: false, loading: chartLoading });
const HistoricalVolatilityChart = dynamic(() => import("@/components/charts/RiskCharts").then((module) => module.HistoricalVolatilityChart), { ssr: false, loading: chartLoading });

function correlationMatrix(tickers: string[], returns: number[][]): { tickers: string[]; values: number[][] } {
  if (!tickers.length || returns.length < 2) return { tickers: [], values: [] };
  const columns = tickers.map((_, column) => returns.map((row) => row[column]));
  const means = columns.map((values) => values.reduce((sum, value) => sum + value, 0) / values.length);
  const deviations = columns.map((values, i) => Math.sqrt(values.reduce((sum, value) => sum + (value - means[i]) ** 2, 0)));
  return {
    tickers,
    values: tickers.map((_, i) => tickers.map((__, j) => {
      if (deviations[i] === 0 || deviations[j] === 0) return 0;
      const covariance = columns[i].reduce((sum, value, k) => sum + (value - means[i]) * (columns[j][k] - means[j]), 0);
      return covariance / (deviations[i] * deviations[j]);
    })),
  };
}

export default function RiskPage() {
  const { activePortfolio, loading: portfoliosLoading } = usePortfolios();
  const replayGeneration = useRef(0);
  const replayRequest = useRef<AbortController | null>(null);
  const [catalog, setCatalog] = useState<CrisisScenarioCatalogEntry[]>([]);
  const [scenarioId, setScenarioId] = useState("");
  const [replay, setReplay] = useState<HistoricalCrisisReplayData | null>(null);
  const [portfolioRisk, setPortfolioRisk] = useState<PortfolioRiskData | null>(null);
  const [portfolioHistory, setPortfolioHistory] = useState<AlignedPortfolioHistory | null>(null);
  const [portfolioLoading, setPortfolioLoading] = useState(false);
  const [portfolioError, setPortfolioError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [latency, setLatency] = useState<number | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    let active = true;
    api.historicalScenarioCatalog(controller.signal).then((response) => {
      if (!active) return;
      setCatalog(response.data.scenarios);
      setScenarioId(response.data.scenarios.find((item) => item.id === "covid-2020")?.id ?? response.data.scenarios[0]?.id ?? "");
    }).catch((cause) => active && setError(cause instanceof Error ? cause.message : "Unable to load historical scenarios"));
    return () => { active = false; controller.abort(); };
  }, []);

  const evaluate = useCallback(async (id = scenarioId) => {
    if (!id) return;
    replayRequest.current?.abort();
    const controller = new AbortController();
    replayRequest.current = controller;
    const generation = ++replayGeneration.current;
    setBusy(true); setError(null); setReplay(null); setLatency(null);
    try {
      const historical = await api.replayHistoricalCrisis(id, undefined, controller.signal);
      if (generation !== replayGeneration.current || controller.signal.aborted) return;
      const panel = historical.data;
      setReplay(panel);
      setLatency(historical.meta.execution_time_ms);
    } catch (cause) {
      if (generation !== replayGeneration.current || controller.signal.aborted) return;
      setError(cause instanceof Error ? cause.message : "Historical replay unavailable");
      setReplay(null);
    } finally { if (generation === replayGeneration.current && !controller.signal.aborted) setBusy(false); }
  }, [scenarioId]);

  useEffect(() => { if (scenarioId) void evaluate(scenarioId); }, [evaluate, scenarioId]);
  useEffect(() => {
    let cancelled = false;
    const controller = new AbortController();
    setPortfolioRisk(null); setPortfolioHistory(null); setPortfolioError(null); setPortfolioLoading(false);
    if (!activePortfolio) return () => { cancelled = true; controller.abort(); };
    setPortfolioLoading(true);
    void loadAlignedPortfolioHistory(activePortfolio.tickers, controller.signal)
      .then(async (history) => {
        const riskResponse = await api.portfolioRisk({
          tickers: history.tickers,
          weights: history.tickers.map((ticker) => activePortfolio.target_weights[ticker] ?? 0),
          returns: history.returns,
          confidence_level: 0.95,
          method: "historical",
        }, controller.signal);
        if (!cancelled) { setPortfolioHistory(history); setPortfolioRisk(riskResponse.data); }
      })
      .catch((cause) => { if (!cancelled) setPortfolioError(cause instanceof Error ? cause.message : "Could not load this portfolio's market history."); })
      .finally(() => { if (!cancelled) setPortfolioLoading(false); });
    return () => { cancelled = true; controller.abort(); };
  }, [activePortfolio]);
  const correlation = useMemo(() => portfolioHistory
    ? correlationMatrix(portfolioHistory.tickers, portfolioHistory.returns)
    : { tickers: [], values: [] }, [portfolioHistory]);
  const scenario = replay?.scenario ?? catalog.find((item) => item.id === scenarioId) ?? null;
  const stressPath = useMemo(() => replay ? Object.entries(replay.replay.drawdown_path).map(([date, drawdown]) => ({ date: date.slice(5), drawdown: drawdown * 100 })) : [], [replay]);
  const volatilityPath = useMemo(() => replay?.rolling_volatility_path.map((row) => ({
    date: row.date.slice(5), portfolio: row.portfolio_annualized_volatility == null ? null : row.portfolio_annualized_volatility * 100,
  })) ?? [], [replay]);
  const distribution = useMemo(() => {
    if (!replay) return [];
    const daily = replay.daily_returns.map((row) => replay.tickers.reduce((sum, ticker, i) => sum + row.returns[ticker] * replay.weights[i], 0) * 100);
    const min = Math.min(...daily), max = Math.max(...daily);
    const low = Math.floor(min), high = Math.ceil(max) || 1;
    const step = Math.max(1, Math.ceil((high - low) / 10));
    const bins = Array.from({ length: Math.ceil((high - low) / step) }, (_, i) => low + i * step);
    return bins.map((lower) => ({ range: `${lower} to ${lower + step}%`, count: daily.filter((value: any) => value >= lower && value < lower + step).length, tail: lower < -2 }));
  }, [replay]);
  const portfolioReturns = useMemo(() => portfolioHistory?.returns.map((row) => row.reduce((sum, value, index) =>
    sum + value * (activePortfolio?.target_weights[portfolioHistory.tickers[index]] ?? 0), 0,
  )) ?? [], [activePortfolio, portfolioHistory]);
  const portfolioVolatility = useMemo(() => {
    if (portfolioReturns.length < 2 || !portfolioHistory) return null;
    const mean = portfolioReturns.reduce((sum, value) => sum + value, 0) / portfolioReturns.length;
    const variance = portfolioReturns.reduce((sum, value) => sum + (value - mean) ** 2, 0) / (portfolioReturns.length - 1);
    return Math.sqrt(variance * portfolioHistory.periodsPerYear);
  }, [portfolioHistory, portfolioReturns]);

  return <div className="space-y-6">
    <PageHeader title="Risk & Stress Lab" subtitle="Review risk for your selected portfolio, then compare it with documented historical market crises." />
    <div className="flex flex-wrap items-center justify-between gap-2 rounded-2xl border border-white/[0.07] bg-white/[0.025] px-4 py-3 text-xs text-slate-400">
      <span>Active portfolio: <b className="text-slate-200">{activePortfolio?.name ?? (portfoliosLoading ? "Loading…" : "None selected")}</b>{activePortfolio ? ` · ${activePortfolio.tickers.join(", ")}` : ""}</span>
      <span>{portfolioHistory ? `${portfolioHistory.dates[0]} to ${portfolioHistory.dates.at(-1)} · ${portfolioHistory.returns.length} observations` : "Uses stored prices for your selected assets"}</span>
    </div>
    {!activePortfolio && !portfoliosLoading && <p className="rounded-xl border border-amber-300/10 bg-amber-300/[0.04] p-4 text-sm text-amber-100">Select or create a portfolio to calculate its risk. Historical crisis replays below are benchmark scenarios and do not use your holdings.</p>}
    {portfolioError && <p role="alert" className="rounded-xl border border-rose-300/10 bg-rose-300/[0.04] p-4 text-sm text-rose-200">Could not calculate selected portfolio risk: {portfolioError}</p>}
    <div className="grid grid-cols-12 gap-3">
      <div className="col-span-12 sm:col-span-6 xl:col-span-3"><MetricCard delay={0} label="Portfolio VaR · 95%" value={portfolioRisk ? `${(portfolioRisk.var * 100).toFixed(2)}%` : portfolioLoading ? "Loading…" : "N/A"} detail={`Per observation · ${portfolioHistory?.periodsPerYear ?? "N/A"} periods/year`} tone="negative" /></div>
      <div className="col-span-12 sm:col-span-6 xl:col-span-3"><MetricCard delay={0.05} label="Portfolio expected shortfall" value={portfolioRisk ? `${(portfolioRisk.cvar * 100).toFixed(2)}%` : "N/A"} detail="Average loss in the worst 5% of observations" tone="negative" /></div>
      <div className="col-span-12 sm:col-span-6 xl:col-span-3"><MetricCard delay={0.1} label="Annualized volatility" value={portfolioVolatility == null ? "N/A" : `${(portfolioVolatility * 100).toFixed(2)}%`} detail="Calculated from selected portfolio weights" tone="accent" /></div>
      <div className="col-span-12 sm:col-span-6 xl:col-span-3"><MetricCard delay={0.15} label="Historical benchmark drawdown" value={replay ? `${(replay.replay.worst_drawdown * 100).toFixed(2)}%` : "N/A"} detail={scenario?.title ?? "Select a crisis replay"} tone="negative" /></div>
    </div>
    <div className="grid grid-cols-12 gap-5">
      <Card title="How your portfolio assets moved together" delay={0.05} className="col-span-12 xl:col-span-7">
        {correlation.tickers.length ? <div className="overflow-x-auto"><table className="w-full min-w-[340px] border-separate border-spacing-1 text-sm"><thead><tr><th />{correlation.tickers.map((ticker) => <th key={ticker} className="pb-2 text-xs font-medium text-slate-400">{ticker}</th>)}</tr></thead><tbody>{correlation.values.map((row, i) => <tr key={correlation.tickers[i]}><th className="pr-2 text-left text-xs font-medium text-slate-400">{correlation.tickers[i]}</th>{row.map((value, j) => <td key={j} className="rounded-lg px-3 py-3 text-center font-mono text-xs font-semibold text-slate-900 dark:text-slate-100" style={{ backgroundColor: `rgba(${value < 0 ? "244,63,94" : "16,185,129"},${0.12 + Math.abs(value) * 0.5})` }}>{value.toFixed(2)}</td>)}</tr>)}</tbody></table></div> : <p className="text-sm text-slate-400">{portfolioLoading ? "Loading selected portfolio price history…" : "No shared price history is available for this portfolio."}</p>}
        <p className="mt-3 text-xs text-slate-400">Based on aligned historical price observations for the active portfolio.</p>
      </Card>
      <Card title="Benchmark returns during the crisis" delay={0.11} className="col-span-12 xl:col-span-5">
        <div className="h-[230px]">{distribution.length ? <ReturnDistributionChart data={distribution} /> : <div className="flex h-full items-center justify-center text-sm text-slate-400">No historical observations loaded</div>}</div>
        <p className="mt-2 text-xs text-slate-400">Equal-weighted benchmark series from the bundled dataset. This chart does not represent your selected holdings.</p>
      </Card>
    </div>
    <div className="grid grid-cols-12 gap-5">
      <Card title="Historical benchmark crisis replay" delay={0.05} className="col-span-12 xl:col-span-5">
        <div className="flex flex-wrap items-center gap-3"><select value={scenarioId} onChange={(event) => setScenarioId(event.target.value)} className="max-w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm dark:border-slate-700 dark:bg-slate-950">{catalog.map((item) => <option key={item.id} value={item.id}>{item.title}</option>)}</select><button onClick={() => void evaluate()} disabled={busy || !scenarioId} className="inline-flex items-center gap-2 rounded-lg bg-slate-900 px-3 py-2 text-sm font-medium text-white hover:bg-slate-700 disabled:opacity-50 dark:bg-emerald-500 dark:text-slate-950"><RefreshCw size={14} className={busy ? "animate-spin" : ""} />{busy ? "Evaluating" : "Replay scenario"}</button></div>
        {replay && <div className="mt-5 space-y-2 text-sm"><p className="flex justify-between"><span className="text-slate-500">Dates</span><b>{replay.scenario.start_date} to {replay.scenario.end_date}</b></p><p className="flex justify-between"><span className="text-slate-500">Trading days</span><b>{replay.replay.n_observations}</b></p><p className="flex justify-between"><span className="text-slate-500">Benchmark return</span><b className={replay.replay.cumulative_return < 0 ? "text-rose-500" : "text-emerald-500"}>{(replay.replay.cumulative_return * 100).toFixed(2)}%</b></p><p className="flex justify-between"><span className="text-slate-500">Peak and lowest point</span><b className="font-mono text-xs">{replay.replay.peak_date} to {replay.replay.trough_date}</b></p></div>}
        {scenario && <div className="mt-4 rounded-lg bg-slate-50 p-3 text-xs leading-relaxed text-slate-600 dark:bg-slate-800 dark:text-slate-300"><p><b>Data source:</b> {scenario.source} · updated {scenario.retrieved_on}</p><p className="mt-1">{scenario.calendar_note}</p><p className="mt-2"><b>Assets and data used</b></p><ul className="mt-1 list-disc space-y-1 pl-4">{scenario.assets.map((ticker) => { const asset = scenario.asset_metadata[ticker]; return <li key={ticker}><a href={asset.source_url} target="_blank" rel="noreferrer" className="underline">{asset.label}</a> · {asset.field}{asset.note ? ` · ${asset.note}` : ""}</li>; })}</ul><p className="mt-2"><b>Portfolio setup:</b> Assets in this period have equal weights.</p><ul className="mt-1 list-disc space-y-1 pl-4">{scenario.limitations.map((note) => <li key={note}>{note}</li>)}</ul></div>}
        {error && <p role="alert" className="mt-3 text-xs text-rose-500">Risk API error: {error}</p>}
      </Card>
      <Card title="Benchmark portfolio fall from its peak" delay={0.11} className="col-span-12 xl:col-span-7"><div className="h-[250px]">{stressPath.length ? <CrisisDrawdownChart data={stressPath} /> : <div className="flex h-full items-center justify-center text-sm text-slate-400"><Activity size={15} className="mr-2" />{busy ? "Loading past prices…" : "Run the replay to see the chart"}</div>}</div></Card>
    </div>
    <Card title="Historical benchmark volatility · annualized 20-session window"><div className="h-[240px]">{volatilityPath.length ? <HistoricalVolatilityChart data={volatilityPath} /> : <div className="flex h-full items-center justify-center text-sm text-slate-400">Volatility data will appear after the replay.</div>}</div><p className="mt-2 flex gap-2 text-xs text-slate-400"><AlertTriangle size={13} className="shrink-0" />Volatility uses the bundled benchmark returns. The chart starts after five days of data.</p></Card>
  </div>;
}
