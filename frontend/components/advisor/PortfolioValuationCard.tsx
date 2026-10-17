"use client";

import { BriefcaseBusiness, TrendingDown, TrendingUp } from "lucide-react";
import Card from "@/components/Card";
import MetricInfoTooltip from "@/components/explainability/MetricInfoTooltip";

const money = (value: number, currency: string) => new Intl.NumberFormat("en-GB", { style: "currency", currency, maximumFractionDigits: 0 }).format(value);

export interface PortfolioValuation {
  available: boolean;
  currency: string;
  total_market_value: number | null;
  total_cost_basis: number | null;
  unrealized_pnl: number | null;
  unrealized_pnl_fraction: number | null;
}

export default function PortfolioValuationCard({ valuation }: { valuation?: PortfolioValuation }) {
  const pnl = valuation?.unrealized_pnl;
  const positive = (pnl ?? 0) >= 0;
  const Icon = positive ? TrendingUp : TrendingDown;
  return <Card title="Portfolio value" className="col-span-12 xl:col-span-6">
    {valuation?.available ? <>
      <div className="flex items-center gap-3"><span className="rounded-xl border border-emerald-300/15 bg-emerald-300/[0.08] p-2.5 text-emerald-200"><BriefcaseBusiness size={17} /></span><div><p className="text-[10px] uppercase tracking-wider text-slate-500">Current market value<MetricInfoTooltip label="Market value" description="Tracked share quantities multiplied by the latest stored close. This is not an intraday account balance." /></p><p className="mt-1 font-mono text-2xl font-semibold text-slate-100">{money(valuation.total_market_value ?? 0, valuation.currency)}</p></div></div>
      <div className="mt-5 grid grid-cols-2 gap-3 border-t border-white/[0.06] pt-4"><div><p className="flex items-center text-[10px] uppercase tracking-wider text-slate-500">Total cost basis<MetricInfoTooltip label="Cost basis" description="Your tracked quantities multiplied by their average purchase prices. It excludes fees and taxes." /></p><p className="mt-1 font-mono text-sm text-slate-200">{money(valuation.total_cost_basis ?? 0, valuation.currency)}</p></div><div><p className="text-[10px] uppercase tracking-wider text-slate-500">Unrealized PnL<MetricInfoTooltip label="Unrealized PnL" description="Current market value minus purchase cost. The percentage divides this gain or loss by total purchase cost, before taxes and fees." /></p><p className={`mt-1 inline-flex items-center gap-1 font-mono text-sm ${positive ? "text-emerald-300" : "text-rose-300"}`}><Icon size={14} />{money(pnl ?? 0, valuation.currency)} ({((valuation.unrealized_pnl_fraction ?? 0) * 100).toFixed(1)}%)</p></div></div>
      <p className="mt-3 text-[10px] leading-relaxed text-slate-600">Estimate from your saved quantities and average buy prices, marked to the latest aligned close.</p>
    </> : <div className="py-2"><p className="text-sm font-medium text-slate-200">Add your holdings to track value</p><p className="mt-1 text-xs leading-relaxed text-slate-500">Without tracked positions, the Advisor starts from cash and uses your monthly contribution for the buy plan. It cannot calculate your actual portfolio value or unrealized PnL.</p></div>}
  </Card>;
}
