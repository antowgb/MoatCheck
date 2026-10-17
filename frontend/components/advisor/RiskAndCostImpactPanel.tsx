"use client";

import Card from "@/components/Card";
import MetricInfoTooltip from "@/components/explainability/MetricInfoTooltip";
import type { AdvisorRecommendationResult } from "@/lib/api";

type RiskNumbers = { expectedReturn: number; volatility: number; var: number; lvar: number | null };

export default function RiskAndCostImpactPanel({ before, after, recommendation }: {
  before: RiskNumbers;
  after: RiskNumbers;
  recommendation: AdvisorRecommendationResult;
}) {
  const rows = [
    { label: "Expected return", help: "Expected yearly return based on the same past data.", higherIsBetter: true, a: before.expectedReturn, b: after.expectedReturn, percent: true },
    { label: "Volatility", help: "Expected yearly price movement based on past data.", a: before.volatility, b: after.volatility, percent: true },
    { label: `VaR · ${recommendation.holding_period_days} days`, help: `Loss estimate over ${recommendation.holding_period_days} calendar days at ${(recommendation.confidence_level * 100).toFixed(0)}% confidence. A higher number means more risk.`, a: before.var, b: after.var, percent: true },
    { label: "L-VaR · liquidation window", help: "Loss estimate over the modeled liquidation window, including spread and market impact. Its horizon differs from holding-period VaR above.", a: before.lvar, b: after.lvar, percent: true },
  ];
  const currency = recommendation.valuation.currency;
  const money = (value: number) => new Intl.NumberFormat("en-GB", { style: "currency", currency, maximumFractionDigits: 0 }).format(value);
  const format = (value: number | null) => value == null || !Number.isFinite(value) ? "N/A" : `${(value * 100).toFixed(2)}%`;

  return <Card title="Risk and trading costs" delay={0.08} className="min-w-0">
    <div className="overflow-x-auto"><table className="w-full min-w-[440px] text-sm"><thead><tr className="border-b border-white/[0.08] text-[10px] uppercase tracking-wider text-slate-500"><th className="pb-2 text-left font-medium">Metric</th><th className="pb-2 text-right font-medium">Current</th><th className="pb-2 text-right font-medium">Proposed</th><th className="pb-2 text-right font-medium">Change</th></tr></thead><tbody>{rows.map((row) => <tr key={row.label} className="border-b border-white/[0.045] last:border-0"><th className="py-3 text-left font-medium text-slate-300"><span className="inline-flex items-center">{row.label}<MetricInfoTooltip label={row.label} description={row.help} /></span></th><td className="py-3 text-right font-mono text-slate-400">{format(row.a)}</td><td className="py-3 text-right font-mono text-slate-100">{format(row.b)}</td><td className={`py-3 text-right font-mono ${row.a == null || row.b == null ? "text-slate-400" : (row.higherIsBetter ? row.b >= row.a : row.b <= row.a) ? "text-emerald-300" : "text-amber-300"}`}>{row.a == null || row.b == null ? "N/A" : `${row.b - row.a > 0 ? "+" : ""}${format(row.b - row.a)}`}</td></tr>)}</tbody></table></div>
    <div className="mt-5 border-t border-white/[0.08] pt-4"><div className="flex items-center justify-between gap-3"><div><h3 className="text-xs font-semibold text-slate-200">Estimated trading cost</h3><p className="mt-1 text-[10px] text-slate-500">Spread and market impact estimate · {recommendation.execution_cost.total_friction_bps.toFixed(1)} basis points</p></div><p className="font-mono text-lg font-semibold text-amber-200">{money(recommendation.execution_cost.total_execution_drag)}</p></div>
      <div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-4">{Object.entries(recommendation.execution_cost.cost_breakdown).map(([source, amount]) => <div key={source} className="rounded-lg border border-white/[0.06] bg-black/15 p-2.5"><p className="text-[9px] uppercase leading-relaxed tracking-wider text-slate-500">{source.replaceAll("_", " ")}</p><p className="mt-1 font-mono text-xs text-slate-200">{money(amount)}</p></div>)}</div>
    </div>
  </Card>;
}
