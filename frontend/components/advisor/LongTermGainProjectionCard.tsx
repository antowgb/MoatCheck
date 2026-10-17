"use client";

import { useMemo, useState } from "react";
import { Area, AreaChart, CartesianGrid, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import MetricInfoTooltip from "@/components/explainability/MetricInfoTooltip";
import Card from "@/components/Card";
import type { AdvisorRecommendationResult } from "@/lib/api";

type Scenario = "pessimistic" | "median" | "optimistic";
const COLORS: Record<Scenario, string> = { pessimistic: "#f59e0b", median: "#22c55e", optimistic: "#22d3ee" };
const money = (value: number, currency: string) => new Intl.NumberFormat("en-GB", { style: "currency", currency, maximumFractionDigits: 0 }).format(value);

export default function LongTermGainProjectionCard({ recommendation }: { recommendation?: AdvisorRecommendationResult | null }) {
  const [scenario, setScenario] = useState<Scenario>("median");
  const projection = recommendation?.projections;
  const chartData = useMemo(() => (projection?.years ?? []).map((point) => ({
    year: point.year, capital: point.total_capital_deposited,
    wealth: scenario === "pessimistic" ? point.pessimistic_value : scenario === "optimistic" ? point.optimistic_value : point.median_value,
    gain: scenario === "pessimistic" ? point.pessimistic_net_gain : scenario === "optimistic" ? point.optimistic_net_gain : point.median_net_gain,
  })), [projection, scenario]);
  const last = chartData[chartData.length - 1];
  const currency = projection?.currency ?? "EUR";
  return <Card title="Long-term estimate" className="col-span-12">
    <div className="flex flex-wrap items-start justify-between gap-3"><div><p className="text-xs text-slate-500">Illustrative portfolio value with monthly contributions</p><p className="mt-1 text-[10px] text-slate-600">Model scenarios are not promises or personalized return guarantees.</p></div><div className="flex rounded-xl border border-white/[0.07] bg-black/15 p-1">{(["pessimistic", "median", "optimistic"] as Scenario[]).map((item) => <button type="button" key={item} onClick={() => setScenario(item)} className={`rounded-lg px-2.5 py-1.5 text-[10px] capitalize transition ${item === scenario ? "bg-white/[0.08] text-slate-100" : "text-slate-500 hover:text-slate-300"}`}>{item}</button>)}</div></div>
    {!projection || !chartData.length ? <div className="mt-6 rounded-xl border border-dashed border-white/[0.08] p-6 text-center text-xs text-slate-500">Preview the portfolio to build its 1 to 20 year estimate.</div> : <>
      <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-4"><div><p className="text-[10px] uppercase tracking-wider text-slate-500">Expected annual return<MetricInfoTooltip label="Expected annual return" description="Historical return estimate used as the model drift, before taxes and changes in valuation." /></p><p className="mt-1 font-mono text-sm text-slate-200">{(projection.annualized_expected_return * 100).toFixed(1)}%</p></div><div><p className="text-[10px] uppercase tracking-wider text-slate-500">Annual volatility<MetricInfoTooltip label="Annual volatility" description="Estimated yearly return standard deviation. Future volatility can differ." /></p><p className="mt-1 font-mono text-sm text-slate-200">{(projection.annualized_volatility * 100).toFixed(1)}%</p></div><div><p className="text-[10px] uppercase tracking-wider text-slate-500">Deposited capital<MetricInfoTooltip label="Deposited capital" description="Starting capital plus monthly deposits. It excludes investment gains." /></p><p className="mt-1 font-mono text-sm text-slate-200">{money(last?.capital ?? 0, currency)}</p></div><div><p className="text-[10px] uppercase tracking-wider text-slate-500">Estimated net gain · {projection.target_horizon_years}y<MetricInfoTooltip label="Estimated net gain" description="Scenario wealth minus all deposited capital. The selected scenario is a modeled percentile, before taxes, fees, and inflation." /></p><p className="mt-1 font-mono text-sm" style={{ color: COLORS[scenario] }}>{money(last?.gain ?? 0, currency)}</p></div></div>
      <div className="mt-4 h-64 w-full"><ResponsiveContainer width="100%" height="100%"><AreaChart data={chartData} margin={{ top: 10, right: 12, bottom: 0, left: 0 }}><defs><linearGradient id="projectedWealthFill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor={COLORS[scenario]} stopOpacity={0.28} /><stop offset="95%" stopColor={COLORS[scenario]} stopOpacity={0.01} /></linearGradient></defs><CartesianGrid stroke="rgba(255,255,255,.06)" vertical={false} /><XAxis dataKey="year" tickFormatter={(year: number) => `${year}y`} tick={{ fill: "#64748b", fontSize: 10 }} axisLine={false} tickLine={false} /><YAxis tickFormatter={(value: number) => new Intl.NumberFormat("en", { notation: "compact", maximumFractionDigits: 1 }).format(value)} tick={{ fill: "#64748b", fontSize: 10 }} axisLine={false} tickLine={false} width={44} /><Tooltip contentStyle={{ background: "#111720", border: "1px solid rgba(255,255,255,.1)", borderRadius: 12, fontSize: 11 }} formatter={(value: number, name: string) => [money(value, currency), name === "wealth" ? `${scenario} estimate` : "Capital deposited"]} labelFormatter={(year) => `Year ${year}`} /><ReferenceLine y={last?.capital} stroke="rgba(203,213,225,.35)" strokeDasharray="4 4" /><Area type="monotone" dataKey="wealth" stroke={COLORS[scenario]} strokeWidth={2.4} fill="url(#projectedWealthFill)" /><Area type="monotone" dataKey="capital" stroke="#94a3b8" strokeWidth={1.5} strokeDasharray="5 4" fill="transparent" /></AreaChart></ResponsiveContainer></div>
      <p className="mt-2 text-[10px] leading-relaxed text-slate-600">10,000 simulated monthly paths. Deposits are added at each month end. Scenarios use historical expected return and volatility estimates, which may differ from future results.</p>
    </>}
  </Card>;
}
