"use client";

import { useEffect, useState } from "react";
import { ArrowRight, Clock3, Database, X } from "lucide-react";
import Link from "next/link";
import Card from "@/components/Card";

const STORAGE_KEY = "moatcheck_workspace_intro_seen";
const steps = [
  { number: "01", title: "Set your allocation", body: "Choose assets and save target weights to your account.", href: "/portfolio", action: "Manage portfolio" },
  { number: "02", title: "Review the evidence", body: "Inspect inputs, risk estimates, costs, and model rationale.", href: "/methodology#data-sources", action: "Data methodology" },
  { number: "03", title: "Choose your time frame", body: "Use the Advisor for plans that last weeks or months. It never places trades.", href: "/advisor", action: "Open Advisor" },
];

/** A dismissible first-visit guide styled with the site's shared glass card and typography. */
export default function WorkspaceOnboarding() {
  const [visible, setVisible] = useState(false);
  useEffect(() => {
    try { setVisible(window.localStorage.getItem(STORAGE_KEY) !== "1"); }
    catch { setVisible(true); }
  }, []);
  if (!visible) return null;

  const dismiss = () => {
    try { window.localStorage.setItem(STORAGE_KEY, "1"); }
    catch { /* Dismissal still works for this render when storage is blocked. */ }
    setVisible(false);
  };

  return (
    <Card className="relative mb-1 overflow-hidden border-emerald-300/[0.1] bg-emerald-300/[0.018]">
      <button type="button" aria-label="Dismiss getting started guide" onClick={dismiss} className="absolute right-3 top-3 grid h-8 w-8 place-items-center rounded-xl text-slate-500 transition hover:bg-white/[0.06] hover:text-slate-200 focus-visible:ring-2 focus-visible:ring-emerald-300 sm:right-4 sm:top-4"><X size={15} /></button>
      <div className="flex items-start gap-3 pr-8">
        <span className="grid h-9 w-9 shrink-0 place-items-center rounded-xl border border-emerald-300/10 bg-emerald-300/[0.05] text-emerald-200"><Clock3 size={16} /></span>
        <div>
          <p className="text-[9px] font-semibold uppercase tracking-[0.15em] text-emerald-200/80">Getting started</p>
          <h2 className="mt-1 text-sm font-semibold text-slate-100">Portfolio research for decisions held over weeks or months</h2>
          <p className="mt-1 max-w-3xl text-xs leading-relaxed text-slate-400">MoatCheck helps you review portfolio choices and risks. It does not place trades or guarantee results.</p>
        </div>
      </div>
      <div className="mt-4 grid gap-0 border-t border-white/[0.06] sm:grid-cols-3 sm:divide-x sm:divide-white/[0.06]">
        {steps.map((step) => (
          <div key={step.number} className="border-b border-white/[0.05] py-3 first:sm:pr-4 sm:border-b-0 sm:px-4 sm:first:pl-0 sm:last:pr-0">
            <div className="flex items-center gap-2"><span className="font-mono text-[10px] text-emerald-300/70">{step.number}</span><h3 className="text-[11px] font-semibold text-slate-200">{step.title}</h3></div>
            <p className="mt-1 pl-7 text-[10px] leading-relaxed text-slate-500">{step.body}</p>
            <Link href={step.href} className="mt-2 inline-flex items-center gap-1 pl-7 text-[10px] font-medium text-slate-300 transition hover:text-emerald-200">{step.action}<ArrowRight size={12} /></Link>
          </div>
        ))}
      </div>
      <p className="mt-3 flex items-center gap-1.5 border-t border-white/[0.06] pt-3 text-[9px] leading-relaxed text-slate-600"><Database size={11} className="shrink-0" />Price and company data come from outside providers. Coverage and update times vary. <Link href="/methodology#data-sources" className="text-slate-400 underline decoration-white/15 underline-offset-2 hover:text-emerald-200">See data sources</Link>.</p>
    </Card>
  );
}
