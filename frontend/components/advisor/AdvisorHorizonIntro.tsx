import { CalendarRange, ShieldCheck, Sparkles } from "lucide-react";

/** Compact onboarding explaining the advisor's multi-week/month decision horizon and review process. */
export default function AdvisorHorizonIntro() {
  return <section aria-label="How the portfolio advisor works" className="bento-panel grid gap-4 p-4 sm:grid-cols-[minmax(0,1.5fr)_repeat(3,minmax(0,1fr))] sm:items-center sm:p-5">
    <div><p className="text-[9px] font-semibold uppercase tracking-[0.17em] text-emerald-300">Long-term guidance · no day trading</p><h2 className="mt-1 text-sm font-semibold text-slate-100">Built for decisions held for weeks or months</h2><p className="mt-1 max-w-xl text-[11px] leading-relaxed text-slate-400">The Advisor uses daily or weekly price data, cautious position sizes, waiting periods, trading limits, and estimated costs. These estimates can be wrong and cannot guarantee results. MoatCheck never sends trades to a broker.</p></div>
    <div className="flex items-start gap-2.5"><CalendarRange size={15} className="mt-0.5 shrink-0 text-cyan-300" /><div><p className="text-[10px] font-semibold text-slate-200">30 to 365 days</p><p className="mt-1 text-[9px] leading-relaxed text-slate-500">This time frame helps avoid reacting to short-term price swings.</p></div></div>
    <div className="flex items-start gap-2.5"><ShieldCheck size={15} className="mt-0.5 shrink-0 text-amber-300" /><div><p className="text-[10px] font-semibold text-slate-200">Risk and trading costs</p><p className="mt-1 text-[9px] leading-relaxed text-slate-500">Position limits, past losses, and estimated trading costs are included.</p></div></div>
    <div className="flex items-start gap-2.5"><Sparkles size={15} className="mt-0.5 shrink-0 text-violet-300" /><div><p className="text-[10px] font-semibold text-slate-200">You stay in control</p><p className="mt-1 text-[9px] leading-relaxed text-slate-500">Each formal proposal is saved for your review; inspect assumptions before acting.</p></div></div>
  </section>;
}
