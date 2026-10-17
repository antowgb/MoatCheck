"use client";

import { motion } from "framer-motion";
import dynamic from "next/dynamic";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { Suspense, useEffect, useState } from "react";
import Card from "@/components/Card";
import Disclaimer from "@/components/Disclaimer";
import ExportCsvButton from "@/components/ExportCsvButton";
import { QualitativeEventItem } from "@/components/QualitativeBadges";
import ScoreBadge from "@/components/ScoreBadge";
import Skeleton from "@/components/Skeleton";
import { api, type PricePoint, type QualitativeEvent, type StockDetail } from "@/lib/api";
import { formatDate, isStale } from "@/lib/date";

const PriceChart = dynamic(() => import("@/components/PriceChart"), { ssr: false, loading: () => <div className="h-96 w-full animate-pulse rounded-2xl bg-white/[0.035]" aria-hidden="true" /> });

const COMPONENT_LABELS: Record<string, string> = {
  revenue_growth_yoy: "Revenue growth (YoY)",
  operating_margin: "Operating margin",
  roe_or_roic: "ROE / ROIC",
  debt_to_ebitda: "Debt/EBITDA (inverted)",
  fcf_positive: "FCF positive",
  volatility: "Volatility",
  sharpe: "Sharpe ratio",
  max_drawdown: "Max drawdown",
  fundamental: "Fundamental",
  risk: "Risk",
};
const label = (k: string) => COMPONENT_LABELS[k] ?? k;

function fmt(
  v: number | string | null | undefined,
  opts?: { pct?: boolean; big?: boolean; currency?: boolean }
): string {
  if (v === null || v === undefined) return "N/A";
  const n = Number(v);
  if (!Number.isFinite(n)) return "N/A";
  if (opts?.pct) return `${(n * 100).toFixed(1)} %`;

  const prefix = opts?.currency ? "$" : "";
  const unitSep = opts?.currency ? "" : " ";
  if (opts?.big) {
    if (Math.abs(n) >= 1e12) return `${prefix}${(n / 1e12).toFixed(2)}${unitSep}T`;
    if (Math.abs(n) >= 1e9) return `${prefix}${(n / 1e9).toFixed(2)}${unitSep}B`;
    if (Math.abs(n) >= 1e6) return `${prefix}${(n / 1e6).toFixed(2)}${unitSep}M`;
  }
  return `${prefix}${n.toFixed(2)}`;
}

function StockDetailView() {
  const ticker = useSearchParams().get("ticker")?.toUpperCase() ?? "";
  const [detail, setDetail] = useState<StockDetail | null>(null);
  const [history, setHistory] = useState<PricePoint[]>([]);
  const [qualitative, setQualitative] = useState<QualitativeEvent[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    setDetail(null); setHistory([]); setQualitative(null); setError(null);
    if (!ticker) return;
    api.stockDetail(ticker).then((value) => { if (active) setDetail(value); })
      .catch((cause) => { if (active) setError(String(cause)); });
    api.priceHistory(ticker).then((value) => { if (active) setHistory(value); }).catch(() => {});
    api.stockQualitative(ticker).then((value) => { if (active) setQualitative(value.events); })
      .catch(() => { if (active) setQualitative([]); });
    return () => { active = false; };
  }, [ticker]);

  if (!ticker)
    return (
      <p className="text-slate-500 dark:text-slate-400">No ticker provided (e.g. /stock/?ticker=NVDA).</p>
    );
  if (error) return <p className="text-red-600 dark:text-rose-400 text-sm">Error: {error}</p>;
  if (!detail)
    return (
      <div className="space-y-6">
        <Skeleton className="h-12 w-80 max-w-full" />
        <div className="grid grid-cols-1 gap-5 md:grid-cols-3">
          <Skeleton className="h-40" />
          <Skeleton className="h-40" />
          <Skeleton className="h-40" />
        </div>
        <Skeleton className="h-96" />
      </div>
    );

  if (detail.stock.asset_type === "etf")
    return (
      <div className="space-y-6">
        <Disclaimer />
        <p className="bento-panel p-6 text-base leading-relaxed text-slate-300 sm:p-8">
          <span className="font-mono text-2xl font-semibold text-slate-50">{detail.stock.ticker}</span>
          {" - "}
          {detail.stock.name}. This is a comparison benchmark, not part of the investable universe tracked by this
          tool.
        </p>
      </div>
    );

  const f = detail.fundamentals;
  const s = detail.score;

  const pending = detail.stock.status === "pending_refresh";

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25 }}
      className="space-y-7"
    >
      <Disclaimer />
      <div className="border-b border-white/[0.07] pb-6">
        <h1 className="text-3xl font-semibold tracking-tight text-slate-50 sm:text-4xl">
          <span className="font-mono">{detail.stock.ticker}</span>{" "}
          <span className="font-normal text-slate-300">{detail.stock.name}</span>
        </h1>
        <p className="mt-2 text-base text-slate-400">
          {detail.stock.sector} · {detail.stock.industry} · {detail.stock.currency}
        </p>
        <p className="mt-2 text-sm text-slate-400">
          Fundamentals, risk metrics, and the full score breakdown for this ticker.
        </p>
        {!pending && (
          <p
            className={`text-xs mt-2 ${
              detail.stock.updated_at ? isStale(detail.stock.updated_at) : false
                ? "text-slate-400 dark:text-slate-600"
                : "text-slate-500 dark:text-slate-400"
            }`}
          >
            Data last updated: {detail.stock.updated_at ? formatDate(detail.stock.updated_at) : "Not available"}
          </p>
        )}
      </div>

      {pending && (
        <div className="rounded-2xl border border-amber-300/15 bg-amber-300/[0.05] px-5 py-4 text-sm leading-relaxed text-amber-100">
          Data pending for this ticker: it was just added and hasn&apos;t
          been processed by a refresh yet (depends on the remaining Alpha Vantage quota, ~3 tickers/day).
          Scores and fundamentals aren&apos;t available yet.
        </div>
      )}

      <div className="grid grid-cols-1 gap-5 md:grid-cols-3">
        <Card title="Composite score">
          <div className="text-4xl font-bold">
            <ScoreBadge score={s?.composite_score ?? null} />
          </div>
          <p className="mt-3 font-mono text-sm text-slate-300">
            Fundamental: {fmt(s?.fundamental_score)} · Risk: {fmt(s?.risk_score)}
          </p>
          <Link
            href="/methodology#composite"
            className="mt-4 inline-block text-sm font-medium text-emerald-300 hover:text-emerald-200 hover:underline"
          >
            How is this score calculated?
          </Link>
        </Card>
        <Card title="Risk">
          <ul className="space-y-3 text-sm text-slate-400">
            <li className="flex items-center justify-between gap-4"><span>Annualized volatility</span><b className="font-mono text-slate-100">{fmt(s?.volatility_annualized, { pct: true })}</b></li>
            <li className="flex items-center justify-between gap-4"><span>Sharpe</span><b className="font-mono text-slate-100">{fmt(s?.sharpe_ratio)}</b></li>
            <li className="flex items-center justify-between gap-4"><span>Max drawdown</span><b className="font-mono text-slate-100">{fmt(s?.max_drawdown, { pct: true })}</b></li>
          </ul>
        </Card>
        <Card title="Valuation">
          <ul className="space-y-3 text-sm text-slate-400">
            <li className="flex items-center justify-between gap-4"><span>Trailing P/E</span><b className="font-mono text-slate-100">{fmt(f?.pe_trailing)}</b></li>
            <li className="flex items-center justify-between gap-4"><span>Forward P/E</span><b className="font-mono text-slate-100">{fmt(f?.pe_forward)}</b></li>
            <li className="flex items-center justify-between gap-4"><span>Market cap</span><b className="font-mono text-slate-100">{fmt(f?.market_cap, { big: true, currency: true })}</b></li>
          </ul>
        </Card>
      </div>

      <Card title="Fundamentals">
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-4">
          {[
            ["Revenue", fmt(f?.revenue, { big: true })],
            ["Growth YoY", fmt(f?.revenue_growth_yoy, { pct: true })],
            ["Operating margin", fmt(f?.operating_margin, { pct: true })],
            ["Net margin", fmt(f?.net_margin, { pct: true })],
            ["ROE", fmt(f?.roe, { pct: true })],
            ["Net debt / EBITDA", fmt(f?.debt_to_ebitda)],
            ["Free cash flow", fmt(f?.free_cash_flow, { big: true })],
          ].map(([metric, value]) => (
            <div key={metric} className="rounded-2xl border border-white/[0.055] bg-white/[0.02] p-4">
              <p className="text-xs font-medium uppercase tracking-wide text-slate-400">{metric}</p>
              <p className="mt-2 break-words font-mono text-xl font-semibold tabular-nums text-slate-100">{value}</p>
            </div>
          ))}
        </div>
      </Card>

      <Card title="Price (5 years)">
        {history.length > 0 ? (
          <>
            <div className="flex justify-end mb-2">
              <ExportCsvButton data={history} filename={`${ticker}_price_history`} columns={["date", "close"]} />
            </div>
            <PriceChart data={history} lines={[{ dataKey: "close", color: "#0284c7", name: ticker }]} />
          </>
        ) : (
          <p className="text-sm text-slate-400 dark:text-slate-600">No price history.</p>
        )}
      </Card>

      <Card title="Qualitative signals" id="qualitative" className="scroll-mt-4">
        <p className="mb-4 text-sm leading-relaxed text-slate-400">
          Recent AI-classified events (contracts, regulatory, technical moat, M&amp;A). This is an indicative{" "}
          <strong>count, not a score</strong>, and is not weighted into the composite. Summaries may contain
          extraction errors. Always verify against the source.{" "}
          <Link href="/methodology#qualitative" className="text-emerald-600 dark:text-emerald-400 hover:underline">
            Learn more
          </Link>
        </p>
        {qualitative === null ? (
          <Skeleton className="h-24" />
        ) : qualitative.length === 0 ? (
          <p className="text-sm text-slate-400 dark:text-slate-600">
            No qualitative events detected for this ticker.
          </p>
        ) : (
          <ul className="mt-1">
            {qualitative.map((ev) => (
              <QualitativeEventItem key={ev.id} event={ev} />
            ))}
          </ul>
        )}
      </Card>

      {s?.score_breakdown != null && (
        <Card title="Score breakdown">
          <div className="grid grid-cols-1 gap-8 xl:grid-cols-2">
            {s.score_breakdown.fundamental && (
              <div>
                <h3 className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase mb-2">
                  Fundamental components
                </h3>
                <table className="w-full text-sm">
                  <tbody>
                    {Object.entries(s.score_breakdown.fundamental.components).map(([k, c]) => (
                      <tr key={k} className="border-b border-slate-100 dark:border-slate-800 last:border-0">
                        <td className="py-1 pr-2 text-slate-700 dark:text-slate-300">{label(k)}</td>
                        <td className="py-1 pr-2 text-slate-500 dark:text-slate-500 font-mono">
                          weight {(c.weight * 100).toFixed(0)}%
                        </td>
                        <td className="py-1 text-right font-medium font-mono text-slate-900 dark:text-slate-100">
                          {(c.normalized * 100).toFixed(0)}/100
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                {s.score_breakdown.fundamental.missing.length > 0 && (
                  <p className="text-xs text-slate-400 dark:text-slate-600 mt-2">
                    Excluded (missing data): {s.score_breakdown.fundamental.missing.map(label).join(", ")}
                  </p>
                )}
              </div>
            )}
            {s.score_breakdown.risk && (
              <div>
                <h3 className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase mb-2">
                  Risk components
                </h3>
                <table className="w-full text-sm">
                  <tbody>
                    {Object.entries(s.score_breakdown.risk.components).map(([k, v]) => (
                      <tr key={k} className="border-b border-slate-100 dark:border-slate-800 last:border-0">
                        <td className="py-1 pr-2 text-slate-700 dark:text-slate-300">{label(k)}</td>
                        <td className="py-1 text-right font-medium font-mono text-slate-900 dark:text-slate-100">
                          {v.toFixed(1)}/100
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                {s.score_breakdown.risk.missing.length > 0 && (
                  <p className="text-xs text-slate-400 dark:text-slate-600 mt-2">
                    Excluded (missing data): {s.score_breakdown.risk.missing.map(label).join(", ")}
                  </p>
                )}
              </div>
            )}
          </div>
          {s.score_breakdown.composite && (
            <p className="text-xs text-slate-400 dark:text-slate-600 mt-4 font-mono">
              Composite weights:{" "}
              {Object.entries(s.score_breakdown.composite.weights)
                .map(([k, w]) => `${label(k)} ${(w * 100).toFixed(0)}%`)
                .join(" + ")}
            </p>
          )}
        </Card>
      )}
    </motion.div>
  );
}

export default function StockPage() {
  return (
    <Suspense fallback={<p className="text-slate-500 dark:text-slate-400">Loading…</p>}>
      <StockDetailView />
    </Suspense>
  );
}
