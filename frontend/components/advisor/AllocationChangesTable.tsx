"use client";

import { ArrowDownRight, ArrowRight, ArrowUpRight } from "lucide-react";
import Card from "@/components/Card";
import MetricInfoTooltip from "@/components/explainability/MetricInfoTooltip";
import type { AdvisorRecommendationResult } from "@/lib/api";

const pct = (value: number) => `${(value * 100).toFixed(2)}%`;

export default function AllocationChangesTable({ recommendation, capital }: { recommendation: AdvisorRecommendationResult; capital: number }) {
  const formatter = new Intl.NumberFormat("en-GB", { style: "currency", currency: recommendation.valuation.currency, maximumFractionDigits: 0 });
  const valuedCapital = recommendation.valuation.total_market_value ?? capital;
  const costs = Array.isArray(recommendation.execution_cost.per_order_costs) ? recommendation.execution_cost.per_order_costs as number[] : [];
  const rows = [...recommendation.tickers, "CASH"].map((ticker, index) => {
    const current = recommendation.current_weights[ticker] ?? 0;
    const target = recommendation.target_weights[ticker] ?? 0;
    const delta = target - current;
    const direction = Math.abs(delta) < 0.0001 ? "Hold" : delta > 0 ? "Buy" : "Sell";
    const notional = Math.abs(delta) * valuedCapital;
    const impact = ticker === "CASH" ? 0 : Math.max(0, notional - (costs[index] ?? 0));
    return { ticker, current, target, delta, direction, notional, impact };
  });

  return <Card title="Allocation changes" delay={0.08} className="min-w-0">
    <div className="mb-3 flex flex-wrap items-center justify-between gap-3"><p className="text-xs text-slate-500">Suggested changes · buys and sells {pct(recommendation.gross_turnover)} · one-way change {pct(recommendation.one_way_turnover)}</p><MetricInfoTooltip label="One-way turnover" description="The share of the portfolio value that would change hands once. Total buys plus sells are counted separately in gross turnover." /></div>
    <div className="overflow-x-auto"><table className="w-full min-w-[620px] text-left text-xs"><thead><tr className="border-b border-white/[0.08] text-[10px] uppercase tracking-wider text-slate-500"><th className="pb-2 font-medium">Asset</th><th className="pb-2 text-right font-medium">Current</th><th className="pb-2 text-right font-medium">Suggested</th><th className="pb-2 text-right font-medium">Weight change</th><th className="pb-2 text-right font-medium">Action</th><th className="pb-2 text-right font-medium">Trade value</th><th className="pb-2 text-right font-medium">After costs*</th></tr></thead><tbody>{rows.map((row) => {
      const Icon = row.direction === "Buy" ? ArrowUpRight : row.direction === "Sell" ? ArrowDownRight : ArrowRight;
      const tone = row.direction === "Buy" ? "text-emerald-300" : row.direction === "Sell" ? "text-amber-300" : "text-slate-500";
      return <tr key={row.ticker} className="border-b border-white/[0.045] last:border-0"><th className="py-3 font-medium text-slate-200">{row.ticker}</th><td className="py-3 text-right font-mono text-slate-400">{pct(row.current)}</td><td className="py-3 text-right font-mono text-slate-200">{pct(row.target)}</td><td className={`py-3 text-right font-mono ${row.delta >= 0 ? "text-emerald-300" : "text-amber-300"}`}>{row.delta > 0 ? "+" : ""}{pct(row.delta)}</td><td className={`py-3 text-right ${tone}`}><span className="inline-flex items-center gap-1"><Icon size={13} />{row.direction}</span></td><td className="py-3 text-right font-mono text-slate-300">{formatter.format(row.notional)}</td><td className="py-3 text-right font-mono text-slate-300">{formatter.format(row.impact)}</td></tr>;
    })}</tbody></table></div>
    <p className="mt-3 text-[10px] leading-relaxed text-slate-600">* Estimated trade value after trading costs. This is not a forecast of investment profit.</p>
  </Card>;
}
