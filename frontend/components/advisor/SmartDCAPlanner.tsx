"use client";

import { ArrowDownRight, ArrowUpRight, Save } from "lucide-react";
import Card from "@/components/Card";
import type { AdvisorRecommendationResult } from "@/lib/api";

const money = (value: number, currency: string) => new Intl.NumberFormat("en-GB", { style: "currency", currency, maximumFractionDigits: 0 }).format(value);

export default function SmartDCAPlanner({
  amount, years, recommendation, busy, onAmountChange, onYearsChange, onSave,
}: {
  amount: number; years: number; recommendation?: AdvisorRecommendationResult | null; busy?: boolean;
  onAmountChange: (value: number) => void; onYearsChange: (value: number) => void; onSave: () => void;
}) {
  const plan = recommendation?.dca_plan;
  const currency = plan?.currency ?? recommendation?.projections.currency ?? "EUR";
  const startsWithCash = recommendation != null && !recommendation.valuation.available;
  return <Card title="Monthly investing plan" className="col-span-12 xl:col-span-6">
    <p className="text-xs leading-relaxed text-slate-500">{startsWithCash
      ? "No positions are tracked, so your monthly contribution is the available cash budget. The plan applies the suggested target weights and leaves any remainder in cash; it does not assume an existing lump sum."
      : "Choose an amount you can keep investing. The plan directs new money toward assets below their target weight. It never recommends a sale."}</p>
    <div className="mt-4 grid grid-cols-[1fr_130px] gap-3">
      <label className="block"><span className="text-[10px] uppercase tracking-wider text-slate-500">Monthly contribution</span><div className="mt-1 flex items-center rounded-xl border border-white/[0.08] bg-black/20 px-3"><span className="text-xs text-slate-500">{currency}</span><input type="number" min={0} step={25} value={amount} onChange={(event) => onAmountChange(Math.max(0, Number(event.target.value) || 0))} className="w-full bg-transparent px-2 py-2.5 font-mono text-sm text-slate-100 outline-none" aria-label="Monthly contribution amount" /></div></label>
      <label className="block"><span className="text-[10px] uppercase tracking-wider text-slate-500">Projection years</span><select value={years} onChange={(event) => onYearsChange(Number(event.target.value))} className="mt-1 w-full rounded-xl border border-white/[0.08] bg-[#111720] px-3 py-2.5 text-xs text-slate-100 outline-none">{Array.from({ length: 20 }, (_, index) => index + 1).map((year) => <option key={year} value={year}>{year} {year === 1 ? "year" : "years"}</option>)}</select></label>
    </div>
    <button type="button" onClick={onSave} disabled={busy} className="mt-3 inline-flex items-center gap-1.5 rounded-lg border border-cyan-300/15 px-3 py-2 text-[11px] font-medium text-cyan-200 hover:bg-cyan-300/[0.06] disabled:opacity-50"><Save size={13} />Save plan</button>
    {plan && <div className="mt-4 border-t border-white/[0.06] pt-3"><p className="mb-2 text-[10px] uppercase tracking-wider text-slate-500">Suggested monthly buys</p>{plan.orders.length ? <div className="space-y-2">{plan.orders.map((order) => <div key={order.ticker} className="flex items-center justify-between text-xs"><span className="inline-flex items-center gap-1.5 text-slate-300"><ArrowUpRight size={13} className="text-emerald-300" />{order.ticker}</span><span className="font-mono text-slate-100">{money(order.amount, currency)} <span className="text-slate-600">({(order.share_of_monthly_dca * 100).toFixed(0)}%)</span></span></div>)}</div> : <p className="text-xs text-slate-500"><ArrowDownRight className="mr-1 inline text-amber-300" size={13} />Set a contribution above zero to see a buy-only plan.</p>}{plan.unallocated_amount > 0 && <div className="mt-3 flex items-center justify-between border-t border-white/[0.06] pt-2 text-xs"><span className="text-slate-500">Keep in cash</span><span className="font-mono text-slate-300">{money(plan.unallocated_amount, currency)}</span></div>}</div>}
  </Card>;
}
