"use client";

import { ArrowDown, ArrowUp, ArrowUpDown, RotateCcw, SlidersHorizontal } from "lucide-react";
import Link from "next/link";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { Suspense, useCallback, useEffect, useRef, useState } from "react";
import Card from "@/components/Card";
import CorrelationHeatmap from "@/components/CorrelationHeatmap";
import Disclaimer from "@/components/Disclaimer";
import ExportCsvButton from "@/components/ExportCsvButton";
import HeaderTooltip from "@/components/HeaderTooltip";
import PageHeader from "@/components/PageHeader";
import { TallyBadges } from "@/components/QualitativeBadges";
import ScoreBadge from "@/components/ScoreBadge";
import Skeleton from "@/components/Skeleton";
import { api, backendErrorMessage, type CorrelationResult, type ScreenerRow } from "@/lib/api";
import { formatDate, isStale } from "@/lib/date";

type SortKey =
  | "composite_score"
  | "fundamental_score"
  | "risk_score"
  | "revenue_growth_yoy"
  | "market_cap"
  | "pe_trailing"
  | "debt_to_ebitda";
type SortDir = "asc" | "desc";

const SORT_COLUMNS: { key: SortKey; label: string; tooltip: string }[] = [
  { key: "composite_score", label: "Composite", tooltip: "Weighted blend of the fundamental (60%) and risk (40%) scores, 0-100." },
  { key: "fundamental_score", label: "Fundamental", tooltip: "Revenue growth, margins, ROE, leverage, and free cash flow, blended into a 0-100 score." },
  { key: "risk_score", label: "Risk", tooltip: "Volatility, Sharpe, Sortino, and max drawdown, blended into a 0-100 score." },
  { key: "revenue_growth_yoy", label: "Growth YoY", tooltip: "Revenue growth over the trailing 12 months vs. the same period a year ago." },
  { key: "market_cap", label: "Market cap", tooltip: "Shares outstanding times the latest closing price." },
  { key: "pe_trailing", label: "P/E", tooltip: "Price divided by trailing 12-month earnings per share." },
  { key: "debt_to_ebitda", label: "Debt/EBITDA", tooltip: "Net debt divided by EBITDA (TTM); lower is less leveraged. Not scored for Financial Services." },
];

const TICKER_SECTOR_TOOLTIPS: Record<"ticker" | "sector", string> = {
  ticker: "Stock ticker symbol.",
  sector: "GICS sector, as reported by the data provider.",
};

const inputCls = "quant-control px-3 py-2 text-xs outline-none focus:border-emerald-300/50";
const VIRTUAL_ROW_HEIGHT = 56;
const VIRTUAL_OVERSCAN = 8;
const SCREENER_VIEWPORT_HEIGHT = "min(72vh, 760px)";

function fmtPct(v: number | null): string {
  return v != null ? `${(v * 100).toFixed(1)} %` : "N/A";
}
// Market cap only (the sole caller of this formatter) , always shown as a dollar amount.
function fmtMarketCap(v: number | null): string {
  if (v == null) return "N/A";
  if (Math.abs(v) >= 1e12) return `$${(v / 1e12).toFixed(2)}T`;
  if (Math.abs(v) >= 1e9) return `$${(v / 1e9).toFixed(2)}B`;
  if (Math.abs(v) >= 1e6) return `$${(v / 1e6).toFixed(2)}M`;
  return `$${v.toFixed(0)}`;
}
function fmtNum(v: number | null): string {
  return v != null ? v.toFixed(2) : "N/A";
}

interface Filters {
  sector: string;
  minScore: string;
  minGrowth: string;
  minRiskScore: string;
  marketCapMin: string;
  marketCapMax: string;
  peMax: string;
  debtToEbitdaMax: string;
}

const EMPTY_FILTERS: Filters = {
  sector: "",
  minScore: "",
  minGrowth: "",
  minRiskScore: "",
  marketCapMin: "",
  marketCapMax: "",
  peMax: "",
  debtToEbitdaMax: "",
};

// Mapping between filter key (React state) <-> query string parameter (URL + API).
const FILTER_PARAMS: Record<keyof Filters, string> = {
  sector: "sector",
  minScore: "min_score",
  minGrowth: "min_growth",
  minRiskScore: "min_risk_score",
  marketCapMin: "market_cap_min",
  marketCapMax: "market_cap_max",
  peMax: "pe_max",
  debtToEbitdaMax: "debt_to_ebitda_max",
};

function filtersFromSearchParams(params: URLSearchParams): Filters {
  const f = { ...EMPTY_FILTERS };
  (Object.keys(FILTER_PARAMS) as (keyof Filters)[]).forEach((k) => {
    const v = params.get(FILTER_PARAMS[k]);
    if (v !== null) f[k] = v;
  });
  return f;
}

function ScreenerView() {
  const requestGeneration = useRef(0);
  const activeListRequest = useRef<AbortController | null>(null);
  const appliedQuery = useRef<Parameters<typeof api.screener>[0] | null>(null);
  const correlationGeneration = useRef(0);
  const activeCorrelationRequest = useRef<AbortController | null>(null);
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();

  const [filters, setFilters] = useState<Filters>(() => filtersFromSearchParams(searchParams));
  const [rows, setRows] = useState<ScreenerRow[]>([]);
  const [nextCursor, setNextCursor] = useState<string | null>(null);
  const [hasMore, setHasMore] = useState(false);
  const [loadingMore, setLoadingMore] = useState(false);
  const [includeEtfs, setIncludeEtfs] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [sortKey, setSortKey] = useState<SortKey>(
    () => (searchParams.get("sort") as SortKey | null) ?? "composite_score"
  );
  const [sortDir, setSortDir] = useState<SortDir>(() => (searchParams.get("dir") as SortDir | null) ?? "desc");

  const [selectedTickers, setSelectedTickers] = useState<Set<string>>(new Set());
  const [correlation, setCorrelation] = useState<CorrelationResult | null>(null);
  const [correlationError, setCorrelationError] = useState<string | null>(null);
  const [correlationLoading, setCorrelationLoading] = useState(false);
  const [tableScrollTop, setTableScrollTop] = useState(0);

  const toggleTicker = (ticker: string) => {
    ++correlationGeneration.current; setCorrelationLoading(false);
    activeCorrelationRequest.current?.abort();
    setSelectedTickers((prev) => {
      const next = new Set(prev);
      if (next.has(ticker)) next.delete(ticker);
      else next.add(ticker);
      return next;
    });
    setCorrelation(null);
    setCorrelationError(null);
  };

  const handleCompareCorrelation = () => {
    const generation = ++correlationGeneration.current;
    activeCorrelationRequest.current?.abort();
    const controller = new AbortController();
    activeCorrelationRequest.current = controller;
    setCorrelationLoading(true);
    setCorrelationError(null);
    api
      .correlation([...selectedTickers], controller.signal)
      .then((res) => { if (generation === correlationGeneration.current) setCorrelation(res); })
      .catch((e) => { if (generation === correlationGeneration.current && !controller.signal.aborted) setCorrelationError(backendErrorMessage(e)); })
      .finally(() => { if (generation === correlationGeneration.current) setCorrelationLoading(false); });
  };

  // Filters and sort order are evaluated in PostgreSQL before cursor paging.
  const applyFilters = useCallback(
    (next: Filters, nextSortKey: SortKey = sortKey, nextSortDir: SortDir = sortDir) => {
      activeListRequest.current?.abort();
      const controller = new AbortController();
      activeListRequest.current = controller;
      const generation = ++requestGeneration.current;
      setLoading(true);
      setLoadingMore(false);
      setError(null);
      setNextCursor(null);
      setHasMore(false);

      const qs = new URLSearchParams();
      (Object.keys(FILTER_PARAMS) as (keyof Filters)[]).forEach((k) => {
        if (next[k]) qs.set(FILTER_PARAMS[k], next[k]);
      });
      qs.set("sort", nextSortKey);
      qs.set("dir", nextSortDir);
      router.replace(`${pathname}?${qs.toString()}`, { scroll: false });

      const query: Parameters<typeof api.screener>[0] = {
          sector: next.sector || undefined,
          min_score: next.minScore ? Number(next.minScore) : undefined,
          min_growth: next.minGrowth ? Number(next.minGrowth) / 100 : undefined,
          min_risk_score: next.minRiskScore ? Number(next.minRiskScore) : undefined,
          market_cap_min: next.marketCapMin ? Number(next.marketCapMin) : undefined,
          market_cap_max: next.marketCapMax ? Number(next.marketCapMax) : undefined,
          pe_max: next.peMax ? Number(next.peMax) : undefined,
          debt_to_ebitda_max: next.debtToEbitdaMax ? Number(next.debtToEbitdaMax) : undefined,
          include_etfs: includeEtfs,
          sort_by: nextSortKey,
          sort_dir: nextSortDir,
          limit: 100,
        };
      appliedQuery.current = query;
      api.screener(query, controller.signal)
        .then((res) => {
          if (generation !== requestGeneration.current) return;
          setRows(res.rows);
          setNextCursor(res.next_cursor);
          setHasMore(res.has_more);
        })
        .catch((e) => { if (generation === requestGeneration.current && !controller.signal.aborted) setError(String(e)); })
        .finally(() => { if (generation === requestGeneration.current && !controller.signal.aborted) setLoading(false); });
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [pathname, router, includeEtfs, sortKey, sortDir]
  );

  // Initial load: picks up filters/sort already present in the URL
  // (link sharing / reloading a filtered view).
  useEffect(() => {
    applyFilters(filters, sortKey, sortDir);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => () => {
    activeListRequest.current?.abort();
    activeCorrelationRequest.current?.abort();
  }, []);

  const handleFilterChange = (key: keyof Filters, value: string) => {
    setFilters((f) => ({ ...f, [key]: value }));
  };

  const handleReset = () => {
    setFilters(EMPTY_FILTERS);
    applyFilters(EMPTY_FILTERS);
  };

  // Sorting runs before cursor pagination on the server.
  const handleSort = (key: SortKey) => {
    const nextDir: SortDir = key === sortKey && sortDir === "desc" ? "asc" : "desc";
    setSortKey(key);
    setSortDir(nextDir);
    applyFilters(filters, key, nextDir);
  };

  const loadMore = () => {
    if (!nextCursor || loadingMore || loading) return;
    activeListRequest.current?.abort();
    const controller = new AbortController();
    activeListRequest.current = controller;
    const generation = requestGeneration.current;
    setLoadingMore(true);
    api.screener({ ...appliedQuery.current, cursor: nextCursor }, controller.signal).then((res) => {
      if (generation !== requestGeneration.current) return;
      setRows((current) => [...new Map([...current, ...res.rows].map((row) => [row.ticker, row])).values()]);
      setNextCursor(res.next_cursor);
      setHasMore(res.has_more);
    }).catch((e) => { if (generation === requestGeneration.current && !controller.signal.aborted) setError(String(e)); })
      .finally(() => { if (generation === requestGeneration.current && !controller.signal.aborted) setLoadingMore(false); });
  };

  const sortedRows = rows;
  const visibleRowCount = Math.ceil(760 / VIRTUAL_ROW_HEIGHT) + VIRTUAL_OVERSCAN * 2;
  const virtualStart = Math.max(0, Math.floor(tableScrollTop / VIRTUAL_ROW_HEIGHT) - VIRTUAL_OVERSCAN);
  const virtualEnd = Math.min(sortedRows.length, virtualStart + visibleRowCount);
  const visibleRows = sortedRows.slice(virtualStart, virtualEnd);
  const virtualTopHeight = virtualStart * VIRTUAL_ROW_HEIGHT;
  const virtualBottomHeight = Math.max(0, (sortedRows.length - virtualEnd) * VIRTUAL_ROW_HEIGHT);

  const sortIcon = (key: SortKey) =>
    key === sortKey ? (
      sortDir === "asc" ? (
        <ArrowUp size={12} />
      ) : (
        <ArrowDown size={12} />
      )
    ) : (
      <ArrowUpDown size={12} className="opacity-30" />
    );

  return (
    <div>
      <PageHeader
        title="Screener"
        subtitle="Filter companies by score, growth, price, and debt to find ones to review."
      />
      <Disclaimer />

      <Card className="mb-6">
        <div className="flex items-center gap-2 mb-3 text-slate-500 dark:text-slate-400">
          <SlidersHorizontal size={14} />
          <span className="text-xs font-semibold uppercase tracking-wide">Filters</span>
        </div>
        <div className="flex flex-wrap gap-3 items-end">
          <label className="text-sm">
            <span className="block text-slate-500 dark:text-slate-400 mb-1">Sector</span>
            <input
              className={inputCls}
              value={filters.sector}
              onChange={(e) => handleFilterChange("sector", e.target.value)}
              placeholder="e.g. Technology"
            />
          </label>
          <label className="inline-flex items-center gap-2 pb-2 text-xs text-slate-400">
            <input type="checkbox" checked={includeEtfs} onChange={(event) => setIncludeEtfs(event.target.checked)} />
            Include ETFs
          </label>
          <label className="text-sm">
            <span className="block text-slate-500 dark:text-slate-400 mb-1">Min composite score</span>
            <input
              type="number"
              className={`${inputCls} w-28`}
              value={filters.minScore}
              onChange={(e) => handleFilterChange("minScore", e.target.value)}
              placeholder="0-100"
            />
          </label>
          <label className="text-sm">
            <span className="block text-slate-500 dark:text-slate-400 mb-1">Min risk score</span>
            <input
              type="number"
              className={`${inputCls} w-28`}
              value={filters.minRiskScore}
              onChange={(e) => handleFilterChange("minRiskScore", e.target.value)}
              placeholder="0-100"
            />
          </label>
          <label className="text-sm">
            <span className="block text-slate-500 dark:text-slate-400 mb-1">Min growth (%)</span>
            <input
              type="number"
              className={`${inputCls} w-28`}
              value={filters.minGrowth}
              onChange={(e) => handleFilterChange("minGrowth", e.target.value)}
              placeholder="e.g. 10"
            />
          </label>
          <label className="text-sm">
            <span className="block text-slate-500 dark:text-slate-400 mb-1">Min market cap</span>
            <input
              type="number"
              className={`${inputCls} w-32`}
              value={filters.marketCapMin}
              onChange={(e) => handleFilterChange("marketCapMin", e.target.value)}
              placeholder="e.g. 1e9"
            />
          </label>
          <label className="text-sm">
            <span className="block text-slate-500 dark:text-slate-400 mb-1">Max market cap</span>
            <input
              type="number"
              className={`${inputCls} w-32`}
              value={filters.marketCapMax}
              onChange={(e) => handleFilterChange("marketCapMax", e.target.value)}
              placeholder="e.g. 1e12"
            />
          </label>
          <label className="text-sm">
            <span className="block text-slate-500 dark:text-slate-400 mb-1">Max P/E</span>
            <input
              type="number"
              className={`${inputCls} w-24`}
              value={filters.peMax}
              onChange={(e) => handleFilterChange("peMax", e.target.value)}
              placeholder="e.g. 30"
            />
          </label>
          <label className="text-sm">
            <span className="block text-slate-500 dark:text-slate-400 mb-1">Max Debt/EBITDA</span>
            <input
              type="number"
              className={`${inputCls} w-24`}
              value={filters.debtToEbitdaMax}
              onChange={(e) => handleFilterChange("debtToEbitdaMax", e.target.value)}
              placeholder="e.g. 3"
            />
          </label>
          <button
            onClick={() => applyFilters(filters)}
            className="rounded-xl bg-emerald-300 px-4 py-2 text-xs font-semibold text-[#08110d] transition-colors hover:bg-emerald-200"
          >
            Filter
          </button>
          <button
            onClick={handleReset}
            className="inline-flex items-center gap-1.5 rounded-xl border border-white/[0.07] px-3 py-2 text-xs text-slate-400 transition-colors hover:bg-white/[0.045] hover:text-slate-100"
          >
            <RotateCcw size={13} />
            Reset
          </button>
        </div>
      </Card>

      {error && <p className="text-red-600 dark:text-rose-400 text-sm mb-4">Error: {error}</p>}

      <div className="flex justify-end mb-2">
        <ExportCsvButton
          data={sortedRows}
          filename="screener"
          columns={[
            "ticker",
            "sector",
            "composite_score",
            "fundamental_score",
            "risk_score",
            "revenue_growth_yoy",
            "market_cap",
            "pe_trailing",
            "debt_to_ebitda",
          ]}
        />
      </div>

      <Card className="mb-6">
        <div className="flex flex-wrap items-center gap-3">
          <button
            onClick={handleCompareCorrelation}
            disabled={selectedTickers.size < 2 || correlationLoading}
            className="rounded-xl bg-emerald-300 px-4 py-2 text-xs font-semibold text-[#08110d] transition-colors hover:bg-emerald-200 disabled:cursor-not-allowed disabled:bg-slate-700 disabled:text-slate-400"
          >
            {correlationLoading ? "Comparing…" : "Compare correlation"}
          </button>
          <span className="text-sm text-slate-500 dark:text-slate-400">
            {selectedTickers.size < 2
              ? "Select at least 2 assets to compare how they move together."
              : `${selectedTickers.size} tickers selected: ${[...selectedTickers].join(", ")}`}
          </span>
        </div>

        {correlationError && (
          <p className="text-red-600 dark:text-rose-400 text-sm mt-3">Error: {correlationError}</p>
        )}

        {correlation && (
          <div className="mt-4">
            <CorrelationHeatmap tickers={correlation.tickers} matrix={correlation.matrix} />
          </div>
        )}
      </Card>

      {loading ? (
        <div className="space-y-2" style={{ height: SCREENER_VIEWPORT_HEIGHT, minHeight: 420 }}>
          <Skeleton className="h-full" />
        </div>
      ) : (
        <div className="overflow-auto rounded-[24px] border border-white/[0.07] bg-[#0e141c]/70" style={{ height: SCREENER_VIEWPORT_HEIGHT, minHeight: 420 }} onScroll={(event) => setTableScrollTop(Math.floor(event.currentTarget.scrollTop / VIRTUAL_ROW_HEIGHT) * VIRTUAL_ROW_HEIGHT)}>
          <table className="min-w-full divide-y divide-white/[0.055]">
            <thead className="sticky top-0 z-10 bg-[#111720]">
              <tr className="text-left text-xs font-semibold uppercase tracking-wide text-slate-300">
                <th className="px-3 py-2 w-8" />
                <th className="px-3 py-2">
                  <HeaderTooltip label="Ticker" tooltip={TICKER_SECTOR_TOOLTIPS.ticker} />
                </th>
                <th className="px-3 py-2">
                  <HeaderTooltip label="Sector" tooltip={TICKER_SECTOR_TOOLTIPS.sector} />
                </th>
                {SORT_COLUMNS.map((c) => (
                  <th
                    key={c.key}
                    className="cursor-pointer select-none px-4 py-4 transition-colors hover:text-slate-100"
                    onClick={() => handleSort(c.key)}
                  >
                    <span className="inline-flex items-center gap-1">
                      <HeaderTooltip label={c.label} tooltip={c.tooltip} />
                      {sortIcon(c.key)}
                    </span>
                  </th>
                ))}
                <th className="px-3 py-2 text-left">
                  <HeaderTooltip label="Updated" tooltip="Last successful stock-data refresh recorded by MoatCheck. This is not a live quote timestamp." />
                </th>
                <th className="px-3 py-2">
                  <HeaderTooltip
                    label="Recent events"
                    tooltip="Counts classified SEC and RSS events from the past 90 days, excluding low-confidence items. This is an indicator, not part of the company score."
                  />
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/[0.045]">
              {virtualTopHeight > 0 && <tr aria-hidden="true"><td colSpan={12} style={{ height: virtualTopHeight, padding: 0, border: 0 }} /></tr>}
              {visibleRows.map((r) => {
                return r.status === "pending_refresh" ? (
                  <tr
                    key={r.ticker}
                    className="h-14 bg-amber-300/[0.035]"
                  >
                    <td className="px-3 py-2">
                      <input
                        type="checkbox"
                        checked={selectedTickers.has(r.ticker)}
                        onChange={() => toggleTicker(r.ticker)}
                        aria-label={`Select ${r.ticker}`}
                      />
                    </td>
                    <td className="px-3 py-2 font-medium font-mono">
                      <Link href={`/stock/?ticker=${r.ticker}`} className="text-emerald-600 dark:text-emerald-400 hover:underline">
                        {r.ticker}
                      </Link>
                    </td>
                    <td className="px-3 py-2 text-sm text-slate-500">{r.sector ?? "N/A"}</td>
                    <td className="px-3 py-2" colSpan={SORT_COLUMNS.length + 2}>
                      <span className="rounded-lg border border-amber-300/15 bg-amber-300/[0.07] px-2 py-1 text-[10px] font-medium text-amber-200">
                        Data pending
                      </span>
                    </td>
                  </tr>
                ) : (
                  <tr
                    key={r.ticker}
                    className="h-14 transition-colors hover:bg-white/[0.035]"
                  >
                    <td className="px-3 py-2">
                      <input
                        type="checkbox"
                        checked={selectedTickers.has(r.ticker)}
                        onChange={() => toggleTicker(r.ticker)}
                        aria-label={`Select ${r.ticker}`}
                      />
                    </td>
                    <td className="px-3 py-2 font-medium font-mono">
                      <Link href={`/stock/?ticker=${r.ticker}`} className="text-emerald-600 dark:text-emerald-400 hover:underline">
                        {r.ticker}
                      </Link>
                    </td>
                    <td className="px-3 py-2 text-sm text-slate-500 dark:text-slate-400">{r.sector ?? "N/A"}</td>
                    <td className="px-3 py-2">
                      <ScoreBadge score={r.composite_score} />
                    </td>
                    <td className="px-3 py-2">
                      <ScoreBadge score={r.fundamental_score} />
                    </td>
                    <td className="px-3 py-2">
                      <ScoreBadge score={r.risk_score} />
                    </td>
                    <td className="px-3 py-2 text-sm font-mono text-slate-700 dark:text-slate-300">
                      {fmtPct(r.revenue_growth_yoy)}
                    </td>
                    <td className="px-3 py-2 text-sm font-mono text-slate-700 dark:text-slate-300">
                      {fmtMarketCap(r.market_cap)}
                    </td>
                    <td className="px-3 py-2 text-sm font-mono text-slate-700 dark:text-slate-300">
                      {fmtNum(r.pe_trailing)}
                    </td>
                    <td className="px-3 py-2 text-sm font-mono text-slate-700 dark:text-slate-300">
                      {fmtNum(r.debt_to_ebitda)}
                    </td>
                    <td className="px-3 py-2 text-xs tabular-nums text-slate-500">
                      {r.updated_at ? (
                        <time dateTime={r.updated_at} className={isStale(r.updated_at) ? "text-amber-300" : ""}>
                          {formatDate(r.updated_at)}
                        </time>
                      ) : "N/A"}
                    </td>
                    <td className="px-3 py-2">
                      <TallyBadges tally={r.qualitative_tally} ticker={r.ticker} linkToTimeline />
                    </td>
                  </tr>
                );
              })}
              {sortedRows.length === 0 && (
                <tr>
                  <td colSpan={12} className="px-3 py-6 text-center text-slate-400 dark:text-slate-600 text-sm">
                    No results.
                  </td>
                </tr>
              )}
              {virtualBottomHeight > 0 && <tr aria-hidden="true"><td colSpan={12} style={{ height: virtualBottomHeight, padding: 0, border: 0 }} /></tr>}
            </tbody>
          </table>
        </div>
      )}
      {!loading && hasMore && (
        <div className="mt-4 flex justify-center">
          <button onClick={loadMore} disabled={loadingMore} className="rounded-xl border border-white/10 bg-white/[0.04] px-5 py-2 text-sm text-slate-200 hover:bg-white/[0.08] disabled:opacity-50">
            {loadingMore ? "Loading…" : "Load more stocks"}
          </button>
        </div>
      )}
    </div>
  );
}

export default function ScreenerPage() {
  return (
    <Suspense fallback={<p className="text-slate-500 dark:text-slate-400">Loading…</p>}>
      <ScreenerView />
    </Suspense>
  );
}
