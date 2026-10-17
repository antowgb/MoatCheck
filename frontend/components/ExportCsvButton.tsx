"use client";

import { Download } from "lucide-react";
import { downloadCsv } from "@/lib/csv";

export default function ExportCsvButton<T extends object>({
  data,
  filename,
  columns,
  label = "Export CSV",
  className = "",
}: {
  data: T[];
  filename: string;
  columns?: string[];
  label?: string;
  className?: string;
}) {
  return (
    <button
      onClick={() => downloadCsv(filename, data as unknown as Record<string, unknown>[], columns)}
      disabled={data.length === 0}
      className={`inline-flex items-center gap-1.5 rounded-xl border border-white/[0.07] px-3 py-2 text-xs font-medium text-slate-400 transition-colors hover:border-white/[0.12] hover:bg-white/[0.04] hover:text-slate-100 disabled:cursor-not-allowed disabled:opacity-40 ${className}`}
    >
      <Download size={13} />
      {label}
    </button>
  );
}
