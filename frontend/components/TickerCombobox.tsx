"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { api, backendErrorMessage } from "@/lib/api";

export interface TickerComboboxOption {
  ticker: string;
  name: string | null;
  sector: string | null;
}

interface Props {
  label: string;
  options: TickerComboboxOption[];
  selected: string[];
  onToggle: (ticker: string) => void;
  onClear: () => void;
  emptyLabel?: string;
  loading?: boolean;
}

const inputCls =
  "glass-control rounded-xl px-3 py-2 text-sm text-slate-100 placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-emerald-400/20 focus:border-emerald-300/40 transition-colors";

export default function TickerCombobox({
  label,
  options,
  selected,
  onToggle,
  onClear,
  emptyLabel = "No ticker tracked yet.",
  loading = false,
}: Props) {
  const [query, setQuery] = useState("");
  const [sector, setSector] = useState("");
  const [open, setOpen] = useState(false);
  const [ingesting, setIngesting] = useState(false);
  const [ingestError, setIngestError] = useState<string | null>(null);
  const [remoteMatch, setRemoteMatch] = useState<{ ticker: string; name: string | null } | null>(null);
  const [remoteSearching, setRemoteSearching] = useState(false);
  const [addedOptions, setAddedOptions] = useState<TickerComboboxOption[]>([]);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function handlePointerDownOutside(e: PointerEvent) {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) setOpen(false);
    }
    function handleKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape") setOpen(false);
    }
    document.addEventListener("pointerdown", handlePointerDownOutside, true);
    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("pointerdown", handlePointerDownOutside, true);
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, []);

  const sectors = useMemo(
    () => Array.from(new Set(options.map((o) => o.sector).filter((s): s is string => !!s))).sort(),
    [options]
  );

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    return [...options, ...addedOptions].filter((o) => {
      if (sector && o.sector !== sector) return false;
      if (!q) return true;
      return o.ticker.toLowerCase().includes(q) || (o.name ?? "").toLowerCase().includes(q);
    });
  }, [options, addedOptions, query, sector]);

  const canAddTicker = /^[A-Za-z0-9.^= -]{1,16}$/.test(query.trim()) && filtered.length === 0;

  useEffect(() => {
    setRemoteMatch(null);
    setIngestError(null);
    const candidate = query.trim();
    if (!candidate || filtered.length > 0 || !/^[A-Za-z0-9.^= -]{1,16}$/.test(candidate)) return;
    const controller = new AbortController();
    const timer = window.setTimeout(() => {
      setRemoteSearching(true);
      void api.searchTicker(candidate, controller.signal)
        .then((result) => {
          if (result.found) setRemoteMatch({ ticker: result.ticker, name: result.name });
        })
        .catch(() => { /* The manual add action still reports the exact error. */ })
        .finally(() => { if (!controller.signal.aborted) setRemoteSearching(false); });
    }, 350);
    return () => { window.clearTimeout(timer); controller.abort(); };
  }, [filtered.length, query]);

  const addTicker = async () => {
    const ticker = query.trim().toUpperCase();
    setIngesting(true);
    setIngestError(null);
    try {
      const result = await api.ingestTicker(ticker);
      const option = { ticker: result.ticker, name: result.name, sector: null };
      setAddedOptions((items) => [...items.filter((item) => item.ticker !== option.ticker), option]);
      onToggle(option.ticker);
      setQuery("");
      setOpen(false);
    } catch (error) {
      setIngestError(backendErrorMessage(error));
    } finally {
      setIngesting(false);
    }
  };

  return (
    <div ref={containerRef} className="relative">
      <div className="flex items-center justify-between mb-2">
        <span className="text-sm font-medium text-slate-300">{label}</span>
        {selected.length > 0 && (
          <button
            onClick={onClear}
            className="text-xs text-slate-500 underline underline-offset-2 transition hover:text-slate-200"
          >
            Clear ({selected.length} selected)
          </button>
        )}
      </div>

      {selected.length > 0 && (
        <div className="flex flex-wrap gap-1.5 mb-2">
          {selected.map((t) => (
            <span
              key={t}
              className="inline-flex items-center gap-1 rounded-lg border border-emerald-300/15 bg-emerald-400/[0.08] px-2 py-1 text-xs font-mono text-emerald-200"
            >
              {t}
              <button
                onClick={() => onToggle(t)}
                className="text-emerald-300 transition hover:text-white"
                aria-label={`Remove ${t}`}
              >
                ×
              </button>
            </span>
          ))}
        </div>
      )}

      <div className="flex gap-2">
        <input
          className={`${inputCls} min-w-0 flex-1`}
          placeholder={loading ? "Loading tickers…" : "Search ticker or name…"}
          value={query}
          disabled={loading}
          onFocus={() => setOpen(true)}
          onKeyDown={(event) => {
            if (event.key !== "Enter" || !canAddTicker || ingesting) return;
            event.preventDefault();
            void addTicker();
          }}
          onChange={(e) => {
            setQuery(e.target.value);
            setOpen(true);
          }}
        />
        {sectors.length > 0 && (
          <select className={`${inputCls} w-44`} value={sector} onChange={(e) => setSector(e.target.value)}>
            <option value="">All sectors</option>
            {sectors.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
        )}
      </div>

      {open && (
        <div className="absolute z-50 mt-2 max-h-64 w-full overflow-y-auto rounded-xl border border-white/[0.1] bg-[#111720] p-1.5 shadow-[0_20px_60px_-18px_rgba(0,0,0,.9)] backdrop-blur-xl">
          {filtered.length === 0 && (
            <p className="px-3 py-2 text-sm text-slate-500">
              {loading ? "Loading tickers…" : remoteSearching ? "Checking the market symbol…" : remoteMatch ? `${remoteMatch.ticker} found outside the local universe.` : options.length === 0 ? emptyLabel : "No match."}
            </p>
          )}
          {canAddTicker && (
            <div className="border-t border-white/[0.06] p-2">
              <button type="button" onClick={addTicker} disabled={ingesting} className="w-full rounded-lg bg-emerald-400/10 px-3 py-2 text-left text-sm text-emerald-200 hover:bg-emerald-400/15 disabled:opacity-50">
                {ingesting ? "Adding symbol…" : remoteMatch ? `Add ${remoteMatch.ticker}${remoteMatch.name ? ` · ${remoteMatch.name}` : ""}` : `Check and add ${query.trim().toUpperCase()}`}
              </button>
              {!ingesting && !ingestError && <p className="px-1 pt-2 text-[10px] text-slate-500">Press Enter or click to validate this symbol.</p>}
              {ingestError && <p role="alert" className="px-1 pt-2 text-xs leading-relaxed text-rose-300">{ingestError}</p>}
            </div>
          )}
          {filtered.map((o) => (
            <label
              key={o.ticker}
              className="flex cursor-pointer items-center gap-2 rounded-lg px-3 py-2 text-sm text-slate-200 transition-colors hover:bg-white/[0.06]"
            >
              <input type="checkbox" className="shrink-0" checked={selected.includes(o.ticker)} onChange={() => onToggle(o.ticker)} />
              <span className="font-mono font-medium text-emerald-200">{o.ticker}</span>
              {o.name && <span className="truncate text-slate-400">{o.name}</span>}
            </label>
          ))}
        </div>
      )}
    </div>
  );
}
