"use client";

import { Activity } from "lucide-react";
import Card from "@/components/Card";
import MetricInfoTooltip from "@/components/explainability/MetricInfoTooltip";
import type { AdvisorRecommendationResult } from "@/lib/api";

const labelText: Record<string, string> = {
  low_volatility: "Low volatility",
  normal_volatility: "Normal volatility",
  high_volatility: "High volatility",
};

export default function MacroRegimeCard({ recommendation }: { recommendation: AdvisorRecommendationResult }) {
  const regime = recommendation.regime;
  const entries = Object.entries(regime.probability_next_period ?? {});
  const thesis = regime.high_volatility_transition_probability >= 0.25
    ? "Recent data points to a greater chance of high volatility continuing. Consider the risk of a larger drop before increasing your exposure."
    : "Recent data does not point to a higher chance of high volatility. This estimate depends on the amount and quality of past data.";
  return <Card title="Market conditions" delay={0.14} className="h-full">
    <div className="flex items-center justify-between gap-3"><div><p className="text-[10px] uppercase tracking-wider text-slate-500">Recent price movement</p><p className="mt-1 text-lg font-semibold text-cyan-100">{labelText[regime.current_state] ?? regime.current_state.replaceAll("_", " ")}</p></div><div className="rounded-xl border border-cyan-300/15 bg-cyan-300/[0.06] p-2.5 text-cyan-200"><Activity size={18} /></div></div>
    <div className="mt-4 space-y-3">{entries.map(([state, probability]) => <div key={state}><div className="mb-1 flex justify-between text-xs"><span className="text-slate-400">{labelText[state] ?? state.replaceAll("_", " ")}</span><span className="font-mono text-slate-200">{(probability * 100).toFixed(1)}%</span></div><div className="h-1.5 overflow-hidden rounded-full bg-white/[0.07]"><div className={`h-full rounded-full ${state === "high_volatility" ? "bg-amber-300" : "bg-cyan-300"}`} style={{ width: `${probability * 100}%` }} /></div></div>)}</div>
    <div className="mt-4 rounded-lg border border-white/[0.07] bg-black/15 p-3"><p className="text-xs leading-relaxed text-slate-300">{thesis}</p><p className="mt-2 text-[10px] text-slate-600">Based on {regime.observations} past readings over {regime.lookback_months} months.</p></div>
    <div className="mt-3 flex items-start gap-1 text-[10px] leading-relaxed text-slate-600"><MetricInfoTooltip label="About this estimate" description="The model looks at past changes in volatility. It does not predict the economy or guarantee future market conditions." />This describes past price movement. It does not label the economy as being in a bull or bear market.</div>
  </Card>;
}
