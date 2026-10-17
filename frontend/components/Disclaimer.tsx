"use client";

import { X } from "lucide-react";
import Link from "next/link";
import { useEffect, useState } from "react";

const STORAGE_KEY = "moatcheck_disclaimer_dismissed";

export default function Disclaimer() {
  const [dismissed, setDismissed] = useState(true);

  useEffect(() => {
    setDismissed(localStorage.getItem(STORAGE_KEY) === "1");
  }, []);

  if (dismissed) return null;

  return (
    <div className="relative mb-5 rounded-2xl border border-amber-300/15 bg-amber-300/[0.035] py-3 pl-4 pr-12 text-xs leading-relaxed text-amber-100/90">
      Personal learning tool for quant finance. This is not investment advice.
      The data and calculations may contain errors, verify independently
      before making any decision.{" "}
        <Link href="/methodology" className="font-medium text-amber-100 underline decoration-amber-100/30 underline-offset-2 hover:text-white">
        How these scores are calculated
      </Link>
      .
      <button
        onClick={() => {
          localStorage.setItem(STORAGE_KEY, "1");
          setDismissed(true);
        }}
        aria-label="Dismiss"
        title="Hide this message on this browser"
        className="absolute right-2 top-2 grid h-8 w-8 place-items-center rounded-xl text-amber-200/60 transition-colors hover:bg-amber-300/[0.08] hover:text-amber-100 focus-visible:ring-2 focus-visible:ring-amber-300"
      >
        <X size={14} />
      </button>
    </div>
  );
}
