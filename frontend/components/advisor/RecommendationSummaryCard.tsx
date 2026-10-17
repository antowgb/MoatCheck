"use client";

import { BadgeCheck, CircleCheck, Clock3, ShieldCheck } from "lucide-react";
import Card from "@/components/Card";
import MetricInfoTooltip from "@/components/explainability/MetricInfoTooltip";
import type { AdvisorRecommendationResult, PersistedAdvisorRecommendation } from "@/lib/api";

export default function RecommendationSummaryCard({
  recommendation,
  persisted,
  busy,
  onAcknowledge,
}: {
  recommendation: AdvisorRecommendationResult;
  persisted: PersistedAdvisorRecommendation | null;
  busy: boolean;
  onAcknowledge: () => void;
}) {
  const thesis = recommendation.rationale.find((item) => item.code === "risk_regime_transition_alert")?.explanation
    ?? recommendation.rationale.find((item) => item.code === "allocation_method")?.explanation
    ?? "No model thesis available in this snapshot.";

  return <Card delay={0.04} className="relative overflow-hidden border-emerald-300/15 bg-[linear-gradient(120deg,rgba(34,197,94,.08),rgba(23,32,44,.65)_45%,rgba(8,145,178,.06))]">
    <div className="absolute right-0 top-0 h-40 w-40 rounded-full bg-emerald-400/[0.07] blur-3xl" />
    <div className="relative flex flex-col gap-5 xl:flex-row xl:items-center xl:justify-between">
      <div className="max-w-3xl">
        <div className="flex flex-wrap items-center gap-2"><span className="inline-flex items-center gap-1.5 rounded-full border border-emerald-300/20 bg-emerald-300/[0.08] px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.13em] text-emerald-200"><ShieldCheck size={12} /> Medium / long-horizon thesis</span>{persisted && <span className="inline-flex items-center gap-1 rounded-full border border-white/10 px-2.5 py-1 text-[10px] text-slate-300">{persisted.status === "acknowledged" ? <><CircleCheck size={12} className="text-emerald-300" /> Human reviewed</> : <><Clock3 size={12} className="text-amber-300" /> Pending review</>}</span>}</div>
        <p className="mt-4 text-xl font-semibold tracking-tight text-white sm:text-2xl">{thesis}</p>
        <p className="mt-2 max-w-2xl text-sm leading-relaxed text-slate-300">{recommendation.portfolio_name} · {recommendation.allocation_method.replaceAll("_", " ")} · {recommendation.tickers.length} assets. The proposal is sized and turnover-capped for a {recommendation.holding_period_days}-day investment horizon.</p>
      </div>
      <div className="flex flex-wrap items-center gap-4 xl:justify-end">
        <div className="min-w-36 rounded-xl border border-white/[0.08] bg-black/20 p-3"><p className="flex items-center text-[10px] uppercase tracking-wider text-slate-500">Evidence coverage<MetricInfoTooltip label="Evidence coverage" description={recommendation.confidence_interpretation} /></p><p className="mt-1 font-mono text-xl font-semibold text-cyan-200">{recommendation.confidence_score.toFixed(0)}<span className="text-sm">%</span></p><p className="text-[10px] text-slate-500">heuristic · not success probability</p></div>
        <div className="min-w-28 rounded-xl border border-white/[0.08] bg-black/20 p-3"><p className="text-[10px] uppercase tracking-wider text-slate-500">Holding period</p><p className="mt-1 font-mono text-xl font-semibold text-slate-100">{recommendation.holding_period_days}<span className="ml-1 text-xs text-slate-500">days</span></p><p className="text-[10px] text-slate-500">Plans from 30 to 365 days</p></div>
        <button type="button" onClick={onAcknowledge} disabled={!persisted?.id || persisted.status === "acknowledged" || busy} className="inline-flex items-center justify-center gap-2 rounded-lg bg-emerald-300 px-4 py-2.5 text-xs font-semibold text-[#062019] shadow-[0_0_24px_rgba(52,211,153,.12)] transition hover:bg-emerald-200 disabled:cursor-not-allowed disabled:opacity-40"><BadgeCheck size={15} />{persisted?.status === "acknowledged" ? "Reviewed" : "Mark as reviewed"}</button>
      </div>
    </div>
  </Card>;
}
