"use client";

import { useEffect, useId, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { BookOpenCheck, X } from "lucide-react";

export type CalculationKind = "markowitz" | "fractional-kelly" | "liquidity-var" | "markov-transition";

const FORMULAS: Record<CalculationKind, { title: string; latex: string }> = {
  markowitz: { title: "Constrained Markowitz allocation", latex: String.raw`w^* = \arg\min_w\; w^T\Sigma w\quad\text{s.t.}\quad \mathbf{1}^T w=1,\;0\leq w_i\leq1` },
  "fractional-kelly": { title: "Fractional multivariate Kelly sizing", latex: String.raw`f = \lambda\,\Sigma^{-1}(\mu-r_f\mathbf{1}),\quad 0.15\leq\lambda\leq0.25` },
  "liquidity-var": { title: "Liquidity-adjusted VaR", latex: String.raw`LVaR_H = V\,VaR_\alpha\sqrt{H}+C_{spread}+C_{impact}` },
  "markov-transition": { title: "Volatility regime transition", latex: String.raw`P_{ij}=\frac{N_{ij}+\frac12}{\sum_j(N_{ij}+\frac12)},\quad \pi P=\pi` },
};

function FormulaView({ kind }: { kind: CalculationKind }) {
  const common = "rounded bg-black/30 px-3 py-4 font-mono text-center text-sm text-cyan-100 sm:text-base";
  if (kind === "markowitz") return <div className={common} aria-label={FORMULAS[kind].latex}><span>w<sup>*</sup></span><span className="mx-2">=</span><span className="text-emerald-300">arg min</span><sub>w</sub><span className="mx-2">w<sup>T</sup>Σw</span><span className="mx-2 text-slate-500">s.t.</span><span>1<sup>T</sup>w = 1, 0 ≤ w<sub>i</sub> ≤ 1</span></div>;
  if (kind === "fractional-kelly") return <div className={common} aria-label={FORMULAS[kind].latex}>f = <span className="text-amber-300">λ</span> Σ<sup>−1</sup>(μ − r<sub>f</sub>1), <span className="text-slate-400">λ ∈ [0.15, 0.25]</span></div>;
  if (kind === "liquidity-var") return <div className={common} aria-label={FORMULAS[kind].latex}>LVaR<sub>H</sub> = V · VaR<sub>α</sub>√H + C<sub>spread</sub> + C<sub>impact</sub></div>;
  return <div className={common} aria-label={FORMULAS[kind].latex}>P<sub>ij</sub> = (N<sub>ij</sub> + ½) / Σ<sub>j</sub>(N<sub>ij</sub> + ½), <span className="ml-2">πP = π</span></div>;
}

export default function CalculationDrawer({
  kind,
  inputs,
  steps,
  caveats,
  triggerLabel = "How is this calculated?",
}: {
  kind: CalculationKind;
  inputs: Record<string, string | number>;
  steps: string[];
  caveats: string[];
  triggerLabel?: string;
}) {
  const [open, setOpen] = useState(false);
  const dialogRef = useRef<HTMLDialogElement>(null);
  const titleId = useId();
  useEffect(() => {
    if (!open) return;
    const dialog = dialogRef.current;
    const previousFocus = document.activeElement as HTMLElement | null;
    const previousOverflow = document.body.style.overflow;
    dialog?.showModal();
    document.body.style.overflow = "hidden";
    return () => {
      dialog?.close();
      document.body.style.overflow = previousOverflow;
      previousFocus?.focus();
    };
  }, [open]);

  return <>
    <button type="button" onClick={() => setOpen(true)} className="inline-flex items-center gap-1.5 rounded-md text-xs font-medium text-cyan-300 transition hover:text-cyan-100"><BookOpenCheck size={14} />{triggerLabel}</button>
    {open && createPortal(<dialog ref={dialogRef} aria-labelledby={titleId} className="calculation-dialog" onCancel={() => setOpen(false)} onClick={(event) => { if (event.target === event.currentTarget) setOpen(false); }}>
      <aside className="absolute inset-y-0 right-0 flex w-full max-w-xl flex-col border-l border-white/10 bg-[#0a1017] shadow-2xl">
        <header className="flex items-start justify-between gap-4 border-b border-white/[0.08] px-5 py-4 sm:px-7"><div><p className="text-[10px] font-semibold uppercase tracking-[0.17em] text-cyan-300">How this works</p><h2 id={titleId} className="mt-1 text-base font-semibold text-slate-100">{FORMULAS[kind].title}</h2></div><button type="button" onClick={() => setOpen(false)} className="rounded-lg border border-white/10 p-2 text-slate-400 hover:text-white" aria-label="Close calculation details"><X size={16} /></button></header>
        <div className="flex-1 space-y-6 overflow-y-auto px-5 py-6 sm:px-7">
          <section><h3 className="mb-2 text-xs font-semibold uppercase tracking-wider text-slate-400">Model formula</h3><FormulaView kind={kind} /><details className="mt-2 text-[10px] text-slate-500"><summary className="cursor-pointer">Formula text</summary><code className="mt-2 block break-all rounded bg-black/25 p-2">{FORMULAS[kind].latex}</code></details></section>
          <section><h3 className="mb-2 text-xs font-semibold uppercase tracking-wider text-slate-400">Your inputs</h3><dl className="divide-y divide-white/[0.06] rounded-lg border border-white/[0.08]">{Object.entries(inputs).map(([name, value]) => <div key={name} className="flex items-start justify-between gap-4 px-3 py-2.5"><dt className="text-xs text-slate-400">{name}</dt><dd className="text-right font-mono text-xs text-slate-200">{typeof value === "number" ? value.toLocaleString(undefined, { maximumFractionDigits: 5 }) : value}</dd></div>)}</dl></section>
          <section><h3 className="mb-2 text-xs font-semibold uppercase tracking-wider text-slate-400">How the result is calculated</h3><ol className="space-y-3">{steps.map((step, index) => <li key={`${index}-${step}`} className="flex gap-3 text-sm leading-relaxed text-slate-300"><span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full border border-cyan-400/30 bg-cyan-400/10 font-mono text-[10px] text-cyan-200">{index + 1}</span><span>{step}</span></li>)}</ol></section>
          <section className="rounded-lg border border-amber-300/15 bg-amber-300/[0.05] p-3"><h3 className="text-xs font-semibold text-amber-200">Things to keep in mind</h3><ul className="mt-2 list-disc space-y-1.5 pl-4 text-xs leading-relaxed text-amber-100/70">{caveats.map((item) => <li key={item}>{item}</li>)}</ul></section>
        </div>
      </aside>
    </dialog>, document.body)}
  </>;
}
