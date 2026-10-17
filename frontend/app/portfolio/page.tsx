"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { ArrowRight, LoaderCircle, Trash2 } from "lucide-react";
import Card from "@/components/Card";
import Disclaimer from "@/components/Disclaimer";
import PageHeader from "@/components/PageHeader";
import TickerCombobox, { type TickerComboboxOption } from "@/components/TickerCombobox";
import { usePortfolios } from "@/components/PortfolioProvider";
import { api } from "@/lib/api";

const STORAGE_KEY = "moatcheck_portfolio";
type DraftPortfolio = { tickers: string[]; weights: Record<string, number> };

function loadDraft(): DraftPortfolio {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return { tickers: [], weights: {} };
    const parsed = JSON.parse(raw) as Partial<DraftPortfolio>;
    return {
      tickers: Array.isArray(parsed.tickers) ? parsed.tickers : [],
      weights: parsed.weights && typeof parsed.weights === "object" ? parsed.weights : {},
    };
  } catch {
    return { tickers: [], weights: {} };
  }
}

/** Portfolio setup and saved-account handoff. Research charts live in their dedicated workspaces. */
export default function PortfolioPage() {
  const { portfolios, activePortfolio, refreshPortfolios, setActivePortfolioId } = usePortfolios();
  const [tickers, setTickers] = useState<string[]>([]);
  const [weights, setWeights] = useState<Record<string, number>>({});
  const [options, setOptions] = useState<TickerComboboxOption[]>([]);
  const [name, setName] = useState("My long-term portfolio");
  const [hydrated, setHydrated] = useState(false);
  const [optionsLoading, setOptionsLoading] = useState(true);
  const [weightsLoading, setWeightsLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [editingPortfolioId, setEditingPortfolioId] = useState<string | null>(activePortfolio?.id ?? null);
  const [error, setError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);

  const tickersKey = useMemo(() => [...tickers].sort().join(","), [tickers]);

  useEffect(() => {
    const draft = activePortfolio ? { tickers: activePortfolio.tickers, weights: activePortfolio.target_weights } : loadDraft();
    setName(activePortfolio?.name ?? "My long-term portfolio");
    setEditingPortfolioId(activePortfolio?.id ?? null);
    setTickers(draft.tickers);
    setWeights(draft.weights);
    setHydrated(true);
    api.listStocks()
      .then((stocks) => setOptions(stocks.map((stock) => ({ ticker: stock.ticker, name: stock.name, sector: stock.sector })).sort((a, b) => a.ticker.localeCompare(b.ticker))))
      .catch(() => setOptions([]))
      .finally(() => setOptionsLoading(false));
  }, [activePortfolio?.id]);

  useEffect(() => {
    if (!hydrated) return;
    try { window.localStorage.setItem(STORAGE_KEY, JSON.stringify({ tickers, weights })); } catch { /* Draft remains in memory when storage is unavailable. */ }
  }, [hydrated, tickers, weights]);

  useEffect(() => {
    if (activePortfolio && tickersKey === [...activePortfolio.tickers].sort().join(",")) {
      setWeights(activePortfolio.target_weights); setWeightsLoading(false);
      return;
    }
    if (tickers.length < 2) {
      setWeightsLoading(false);
      setWeights(tickers.length === 1 ? { [tickers[0]]: 1 } : {});
      return;
    }
    let cancelled = false;
    setWeightsLoading(true);
    api.portfolioWeights(tickers)
      .then((result) => { if (!cancelled) setWeights(result.weights); })
      .catch(() => { if (!cancelled) setWeights({}); })
      .finally(() => { if (!cancelled) setWeightsLoading(false); });
    return () => { cancelled = true; };
  // ticker identity, not array identity, drives the calculation.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tickersKey]);

  const toggleTicker = (ticker: string) => {
    setMessage(null);
    setTickers((current) => current.includes(ticker) ? current.filter((item) => item !== ticker) : [...current, ticker]);
  };

  const savePortfolio = async (asNew = false) => {
    setError(null);
    setMessage(null);
    const values = tickers.map((ticker) => [ticker, weights[ticker]] as const);
    if (!values.length || values.some(([, weight]) => !Number.isFinite(weight) || weight < 0)) {
      setError("Choose at least one ticker and wait for its suggested allocation to load.");
      return;
    }
    const total = values.reduce((sum, [, weight]) => sum + weight, 0);
    if (total <= 0) {
      setError("The suggested allocation has no investable weight.");
      return;
    }
    setSaving(true);
    try {
      const body = {
        name: name.trim() || "My long-term portfolio",
        tickers,
        target_weights: Object.fromEntries(values.map(([ticker, weight]) => [ticker, weight / total])),
        risk_constraints: { horizon: "medium_to_long_term" },
        optimization_parameters: { allocation_method: "inverse_volatility" },
        description: "Saved for medium- to long-term portfolio review.",
      };
      const shouldUpdate = editingPortfolioId !== null && !asNew;
      const response = shouldUpdate
        ? await api.updatePortfolioConfig(editingPortfolioId, body)
        : await api.createPortfolioConfig(body);
      await refreshPortfolios();
      setActivePortfolioId(response.data.id);
      setEditingPortfolioId(response.data.id);
      setMessage(shouldUpdate ? "Portfolio changes saved." : "Portfolio created and selected.");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not save the portfolio. Sign in and try again.");
    } finally {
      setSaving(false);
    }
  };

  const deletePortfolio = async () => {
    if (!activePortfolio) return;
    const confirmed = window.confirm(
      `Delete ${activePortfolio.name}? Its tracked quantities, DCA settings, and saved Advisor recommendations will also be deleted. Backtests and stress runs remain as unlinked results.`,
    );
    if (!confirmed) return;
    setSaving(true);
    setError(null);
    setMessage(null);
    try {
      await api.deletePortfolioConfig(activePortfolio.id);
      await refreshPortfolios();
      setMessage(`Deleted ${activePortfolio.name}.`);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not delete the portfolio.");
    } finally {
      setSaving(false);
    }
  };

  const startNewPortfolio = () => {
    setEditingPortfolioId(null);
    setName("My long-term portfolio");
    setTickers([]);
    setWeights({});
    setMessage(null);
    setError(null);
  };

  return <div className="space-y-5">
    <PageHeader title="My Portfolio" subtitle="Choose the assets you want to follow, review a starting allocation, and save it for the Advisor." />
    <Disclaimer />
    {portfolios.length > 0 && <Card title="Saved portfolios" className="relative z-10">
      <div className="flex flex-wrap gap-2">
        {portfolios.map((portfolio) => <button key={portfolio.id} type="button" onClick={() => { setActivePortfolioId(portfolio.id); setEditingPortfolioId(portfolio.id); }} className={`rounded-xl border px-3 py-2 text-left transition ${activePortfolio?.id === portfolio.id ? "border-emerald-300/25 bg-emerald-300/[0.08] text-emerald-100" : "border-white/[0.08] bg-white/[0.02] text-slate-300 hover:bg-white/[0.05]"}`}>
          <span className="block text-xs font-medium">{portfolio.name}</span><span className="mt-1 block text-[10px] text-slate-500">{portfolio.tickers.length} assets · updated {new Date(portfolio.updated_at).toLocaleDateString()}</span>
        </button>)}
        <button type="button" onClick={startNewPortfolio} className="rounded-xl border border-dashed border-white/[0.12] px-3 py-2 text-xs text-slate-400 transition hover:border-emerald-300/25 hover:text-emerald-200">Create another</button>
      </div>
      {activePortfolio && editingPortfolioId === activePortfolio.id && <p className="mt-3 text-[10px] text-slate-500">Editing the selected portfolio. Changes are saved only when you choose Save changes.</p>}
    </Card>}

    <Card className="relative z-30 overflow-visible">
      <TickerCombobox
        label="Choose assets"
        options={options}
        selected={tickers}
        onToggle={toggleTicker}
        onClear={() => { setTickers([]); setMessage(null); }}
        emptyLabel="No assets are available right now."
        loading={optionsLoading}
      />
    </Card>

    <div className="grid grid-cols-1 gap-5 xl:grid-cols-12">
      <Card title="Starting allocation" className="xl:col-span-7">
        {tickers.length === 0 ? <p className="text-sm text-slate-400">Add assets above to build a portfolio.</p> : <div className="divide-y divide-white/[0.06]">
          {tickers.map((ticker) => <div key={ticker} className="flex items-center justify-between gap-4 py-3 first:pt-0 last:pb-0">
            <div className="flex min-w-0 items-center gap-3"><span className="font-mono text-sm font-semibold text-slate-100">{ticker}</span><span className="truncate text-xs text-slate-500">{options.find((option) => option.ticker === ticker)?.name}</span></div>
            <span className="shrink-0 font-mono text-sm text-emerald-200">{weightsLoading ? "Loading…" : Number.isFinite(weights[ticker]) ? `${(weights[ticker] * 100).toFixed(1)}%` : "Pending"}</span>
          </div>)}
        </div>}
        <p className="mt-4 text-xs leading-relaxed text-slate-500">Suggested weights use inverse historical volatility as a starting point. Review the assumptions in the Optimizer before making decisions.</p>
        <Link href="/optimizer" className="mt-4 inline-flex items-center gap-1.5 text-xs font-medium text-emerald-200 hover:text-emerald-100">Open Portfolio Optimizer <ArrowRight size={13} /></Link>
      </Card>

      <Card title="Save for long-term review" className="xl:col-span-5">
        <p className="mb-4 text-xs leading-relaxed text-slate-400">MoatCheck is designed for plans held over weeks or months. It does not place trades. Saving changes updates this portfolio; Save as new keeps the current one and creates a separate copy.</p>
        <label className="block text-xs font-medium text-slate-300">Portfolio name
          <input className="quant-control mt-2 h-11 w-full px-3.5 text-sm placeholder:text-sm placeholder:text-slate-500" value={name} maxLength={120} onChange={(event) => setName(event.target.value)} placeholder="e.g. Long-term core" />
        </label>
        <div className="mt-4 grid gap-2">
          <button type="button" onClick={() => void savePortfolio()} disabled={saving || !tickers.length || weightsLoading || tickers.some((ticker) => !Number.isFinite(weights[ticker]))} className="inline-flex h-11 w-full items-center justify-center gap-2 rounded-xl bg-emerald-400 px-4 text-sm font-semibold text-slate-950 transition hover:bg-emerald-300 disabled:cursor-not-allowed disabled:opacity-50">
            {saving ? <LoaderCircle size={15} className="animate-spin" /> : null}{saving ? "Saving…" : editingPortfolioId ? "Save changes" : "Create and select portfolio"}
          </button>
          {editingPortfolioId && <button type="button" onClick={() => void savePortfolio(true)} disabled={saving || !tickers.length || weightsLoading || tickers.some((ticker) => !Number.isFinite(weights[ticker]))} className="inline-flex h-10 w-full items-center justify-center rounded-xl border border-white/[0.09] bg-white/[0.025] px-4 text-xs font-medium text-slate-300 transition hover:bg-white/[0.06] disabled:cursor-not-allowed disabled:opacity-50">Save as new portfolio</button>}
          {activePortfolio && editingPortfolioId === activePortfolio.id && <button type="button" onClick={() => void deletePortfolio()} disabled={saving} className="inline-flex h-10 w-full items-center justify-center gap-2 rounded-xl border border-rose-300/15 px-4 text-xs text-rose-200 transition hover:bg-rose-300/[0.06] disabled:opacity-50"><Trash2 size={14} />Delete selected portfolio</button>}
        </div>
        {error && <p role="alert" className="mt-3 text-xs text-rose-300">{error}</p>}
        {message && <p role="status" className="mt-3 text-xs text-emerald-200">{message} <Link href="/advisor" className="underline underline-offset-2">Open Advisor</Link></p>}
      </Card>
    </div>

    {tickers.length > 0 && <div className="flex justify-end"><button type="button" onClick={() => { setTickers([]); setMessage(null); }} className="inline-flex items-center gap-2 rounded-xl px-3 py-2 text-xs text-slate-500 transition hover:bg-rose-400/[0.08] hover:text-rose-300"><Trash2 size={14} />Clear this draft</button></div>}
  </div>;
}
