"use client";

import { useEffect, useId, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { Info } from "lucide-react";

/** Inline-safe help with keyboard support, outside dismissal, and unclipped content. */
export default function MetricInfoTooltip({ label, description }: { label: string; description: string }) {
  const [open, setOpen] = useState(false);
  const [position, setPosition] = useState({ left: 0, top: 0 });
  const buttonRef = useRef<HTMLButtonElement>(null);
  const id = useId();
  useEffect(() => {
    if (!open) return;
    const reposition = () => {
      const rect = buttonRef.current?.getBoundingClientRect();
      if (rect) setPosition({ left: Math.max(12, Math.min(rect.left, window.innerWidth - 268)), top: rect.bottom + 8 });
    };
    const dismiss = (event: PointerEvent) => { if (!buttonRef.current?.contains(event.target as Node)) setOpen(false); };
    const escape = (event: KeyboardEvent) => { if (event.key === "Escape") setOpen(false); };
    reposition();
    window.addEventListener("resize", reposition);
    window.addEventListener("scroll", reposition, true);
    document.addEventListener("pointerdown", dismiss);
    document.addEventListener("keydown", escape);
    return () => {
      window.removeEventListener("resize", reposition);
      window.removeEventListener("scroll", reposition, true);
      document.removeEventListener("pointerdown", dismiss);
      document.removeEventListener("keydown", escape);
    };
  }, [open]);
  return <span className="inline-flex align-middle">
    <button ref={buttonRef} type="button" aria-label={`Explain ${label}`} aria-describedby={open ? id : undefined}
      onClick={() => setOpen((value) => !value)} onBlur={() => setOpen(false)}
      className="ml-1 inline-flex h-5 w-5 shrink-0 items-center justify-center rounded-full text-slate-400 hover:text-cyan-200">
      <Info size={13} aria-hidden="true" />
    </button>
    {open && createPortal(<span id={id} role="tooltip" style={position} className="pointer-events-none fixed z-[100] block w-64 max-w-[calc(100vw-24px)] rounded-xl border border-white/10 bg-[#17202C] p-3 text-left text-[11px] font-normal normal-case leading-relaxed tracking-normal text-slate-300 shadow-2xl">
      <span className="mb-1 block font-semibold text-cyan-200">{label}</span>{description}
    </span>, document.body)}
  </span>;
}
