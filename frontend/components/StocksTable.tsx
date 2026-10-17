"use client";

import { motion } from "framer-motion";
import { ArrowDown, ArrowUp, ArrowUpDown } from "lucide-react";
import Link from "next/link";
import { useMemo, useState } from "react";
import ExportCsvButton from "./ExportCsvButton";
import HeaderTooltip from "./HeaderTooltip";
import { TallyBadges } from "./QualitativeBadges";
import ScoreBadge from "./ScoreBadge";
import type { Stock } from "@/lib/api";
import { formatDate, isStale } from "@/lib/date";

type SortKey = "ticker" | "name" | "sector" | "composite_score" | "updated_at";

export default function StocksTable({ stocks }: { stocks: Stock[] }) {
  const [sortKey, setSortKey] = useState<SortKey>("composite_score");
  const [asc, setAsc] = useState(false);
  const [sectorFilter, setSectorFilter] = useState("");

  const sectors = useMemo(
    () => Array.from(new Set(stocks.map((s) => s.sector).filter(Boolean))).sort() as string[],
    [stocks]
  );

  const rows = useMemo(() => {
    const filtered = sectorFilter ? stocks.filter((s) => s.sector === sectorFilter) : stocks;
    return [...filtered].sort((a, b) => {
      const va = a[sortKey];
      const vb = b[sortKey];
      if (va === null || va === undefined) return 1;
      if (vb === null || vb === undefined) return -1;
      const cmp = typeof va === "number" ? va - (vb as number) : String(va).localeCompare(String(vb));
      return asc ? cmp : -cmp;
    });
  }, [stocks, sortKey, asc, sectorFilter]);

  const COLUMN_TOOLTIPS: Record<SortKey, string> = {
    ticker: "Stock ticker symbol.",
    name: "Company name.",
    sector: "GICS sector, as reported by the data provider.",
    composite_score: "Weighted blend of the fundamental (60%) and risk (40%) scores, 0-100.",
    updated_at: "Date this ticker's data was last refreshed.",
  };

  const header = (key: SortKey, label: string) => (
    <th
      className="cursor-pointer select-none px-3 py-3 text-left text-[9px] font-semibold uppercase tracking-[0.12em] text-slate-500 transition-colors hover:text-slate-200"
      onClick={() => {
        if (sortKey === key) setAsc(!asc);
        else {
          setSortKey(key);
          setAsc(key !== "composite_score");
        }
      }}
    >
      <span className="inline-flex items-center gap-1">
        <HeaderTooltip label={label} tooltip={COLUMN_TOOLTIPS[key]} />
        {sortKey === key ? (
          asc ? <ArrowUp size={12} /> : <ArrowDown size={12} />
        ) : (
          <ArrowUpDown size={12} className="opacity-30" />
        )}
      </span>
    </th>
  );

  return (
    <div>
      <div className="mb-3 flex items-center justify-between gap-3">
        <select
          className="quant-control min-w-36 px-3 py-2 text-xs"
          value={sectorFilter}
          onChange={(e) => setSectorFilter(e.target.value)}
        >
          <option value="">All sectors</option>
          {sectors.map((s) => (
            <option key={s} value={s}>
              {s}
            </option>
          ))}
        </select>
        <ExportCsvButton
          data={rows}
          filename="stocks"
          columns={["ticker", "name", "sector", "industry", "composite_score", "status", "updated_at"]}
        />
      </div>
      <div className="overflow-x-auto rounded-2xl border border-white/[0.07] bg-[#0e141c]/70">
        <table className="min-w-full divide-y divide-white/[0.055]">
          <thead className="bg-white/[0.025]">
            <tr>
              {header("ticker", "Ticker")}
              {header("name", "Name")}
              {header("sector", "Sector")}
              {header("composite_score", "Composite score")}
              <th className="px-3 py-3 text-left text-[9px] font-semibold uppercase tracking-[0.12em] text-slate-500">
                <HeaderTooltip
                  label="Recent events"
                  tooltip="Count of AI-classified qualitative events over 90 days (positive, negative, neutral). Indicative, not a score."
                />
              </th>
              {header("updated_at", "Updated")}
            </tr>
          </thead>
          <tbody className="divide-y divide-white/[0.045]">
            {rows.map((s, i) => (
              <motion.tr
                key={s.ticker}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.2, delay: Math.min(i, 20) * 0.015 }}
                className={
                  s.status === "pending_refresh"
                ? "bg-amber-300/[0.035]"
                : "transition-colors hover:bg-white/[0.025]"
                }
              >
                <td className="px-3 py-2 font-medium font-mono">
                  <Link href={`/stock/?ticker=${s.ticker}`} className="text-emerald-200 hover:text-emerald-100 hover:underline">
                    {s.ticker}
                  </Link>
                </td>
                <td className="px-3 py-2 text-sm text-slate-300">{s.name ?? "N/A"}</td>
                <td className="px-3 py-2 text-sm text-slate-500">{s.sector ?? "N/A"}</td>
                <td className="px-3 py-2">
                  {s.status === "pending_refresh" ? (
                    <span className="rounded-lg border border-amber-300/15 bg-amber-300/[0.07] px-2 py-1 text-[10px] font-medium text-amber-200">
                      Data pending
                    </span>
                  ) : (
                    <ScoreBadge score={s.composite_score} />
                  )}
                </td>
                <td className="px-3 py-2">
                  <TallyBadges tally={s.qualitative_tally} ticker={s.ticker} linkToTimeline />
                </td>
                <td className="px-3 py-2 text-sm">
                  {s.status === "pending_refresh" ? (
                    <span className="text-slate-600">N/A</span>
                  ) : (
                    <span
                      className={
                        s.updated_at ? (isStale(s.updated_at)
                          ? "text-slate-600"
                          : "text-slate-500") : "text-slate-500"
                      }
                    >
                      {s.updated_at ? formatDate(s.updated_at) : "Not available"}
                    </span>
                  )}
                </td>
              </motion.tr>
            ))}
            {rows.length === 0 && (
              <tr>
                <td colSpan={6} className="px-3 py-8 text-center text-sm text-slate-500">
                  No stocks yet. Trigger a data refresh to populate the list.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
