"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { ArrowRight, Check, Eye, LoaderCircle, Sparkles, Trash2, Zap } from "lucide-react";
import Card from "@/components/Card";
import Disclaimer from "@/components/Disclaimer";
import PageHeader from "@/components/PageHeader";
import TickerCombobox, { type TickerComboboxOption } from "@/components/TickerCombobox";
import { usePortfolios } from "@/components/PortfolioProvider";
import { api, backendErrorMessage } from "@/lib/api";

const STORAGE_KEY = "moatcheck_portfolio";
type DraftPortfolio = { tickers: string[]; weights: Record<string, number> };

type PortfolioIdea = {
  id: string;
  name: string;
  tagline: string;
  description: string;
  risk: "Modéré" | "Équilibré" | "Dynamique";
  horizon: string;
  tickers: string[];
  weights: Record<string, number>;
  accent: string;
};

const PORTFOLIO_IDEAS: PortfolioIdea[] = [
  {
    id: "quality-core",
    name: "Quality Core",
    tagline: "Un socle robuste pour commencer",
    description: "Un panier concentré d’entreprises rentables et très liquides, pensé pour une approche simple et de long terme.",
    risk: "Équilibré",
    horizon: "5 ans et plus",
    tickers: ["MSFT", "AAPL", "AMZN", "GOOGL", "COST"],
    weights: { MSFT: 0.24, AAPL: 0.22, AMZN: 0.2, GOOGL: 0.18, COST: 0.16 },
    accent: "emerald",
  },
  {
    id: "growth-engine",
    name: "Growth Engine",
    tagline: "Croissance et innovation",
    description: "Une idée plus offensive autour des grandes tendances technologiques. À confronter à votre tolérance au risque.",
    risk: "Dynamique",
    horizon: "7 ans et plus",
    tickers: ["NVDA", "MSFT", "AMZN", "AVGO", "META"],
    weights: { NVDA: 0.22, MSFT: 0.22, AMZN: 0.2, AVGO: 0.19, META: 0.17 },
    accent: "violet",
  },
  {
    id: "steady-income",
    name: "Steady & Income",
    tagline: "Plus défensif, plus régulier",
    description: "Une base diversifiée entre consommation, santé, énergie et télécoms pour réduire la dépendance à un seul secteur.",
    risk: "Modéré",
    horizon: "3 ans et plus",
    tickers: ["JNJ", "PG", "KO", "XOM", "DUK"],
    weights: { JNJ: 0.21, PG: 0.21, KO: 0.2, XOM: 0.19, DUK: 0.19 },
    accent: "cyan",
  },
];

function formatIdeaWeight(weight: number): string {
  return `${(weight * 100).toFixed(0)}%`;
}

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
  const { portfolios, activePortfolio, refreshPortfolios, setActivePortfolioId, error: portfolioError } = usePortfolios();
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
  const [previewIdea, setPreviewIdea] = useState<PortfolioIdea | null>(null);
  const [creatingIdeaId, setCreatingIdeaId] = useState<string | null>(null);

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
      setError(backendErrorMessage(cause).includes("API 401") ? "Connectez-vous pour créer et sauvegarder un portfolio." : backendErrorMessage(cause));
    } finally {
      setSaving(false);
    }
  };

  const createFromIdea = async (idea: PortfolioIdea) => {
    setCreatingIdeaId(idea.id);
    setError(null);
    setMessage(null);
    try {
      const response = await api.createPortfolioConfig({
        name: idea.name,
        tickers: idea.tickers,
        target_weights: idea.weights,
        risk_constraints: { horizon: idea.horizon, profile: idea.risk },
        optimization_parameters: { allocation_method: "curated_idea", idea_id: idea.id },
        description: idea.description,
      });
      await refreshPortfolios();
      setActivePortfolioId(response.data.id);
      setEditingPortfolioId(response.data.id);
      setName(response.data.name);
      setTickers(response.data.tickers);
      setWeights(response.data.target_weights);
      setMessage(`${idea.name} a été créé et sélectionné.`);
      setPreviewIdea(null);
    } catch (cause) {
      const raw = backendErrorMessage(cause);
      setError(raw.includes("API 401") ? "Connectez-vous pour créer et sauvegarder un portfolio." : raw);
    } finally {
      setCreatingIdeaId(null);
    }
  };

  const useIdeaAsDraft = (idea: PortfolioIdea) => {
    setName(idea.name);
    setTickers(idea.tickers);
    setWeights(idea.weights);
    setEditingPortfolioId(null);
    setMessage(`${idea.name} est prêt à être personnalisé.`);
    setPreviewIdea(null);
  };

  const suggestions = useMemo(() => {
    if (!tickers.length) return ["Ajoutez 3 à 6 actifs pour une première diversification.", "Combinez plusieurs secteurs plutôt que des entreprises très proches."];
    if (tickers.length === 1) return ["Ajoutez au moins 2 actifs pour éviter un portfolio mono-ligne.", "Un ETF large ou un secteur différent peut compléter cette position."];
    const selectedSectors = new Set(tickers.map((ticker) => options.find((option) => option.ticker === ticker)?.sector).filter(Boolean));
    return [
      tickers.length > 8 ? "Votre liste est large : l’optimiseur peut aider à éviter une allocation trop fragmentée." : "Votre taille de panier est adaptée à une revue rapide.",
      selectedSectors.size < 2 ? "Essayez d’ajouter un secteur différent pour améliorer la diversification." : `${selectedSectors.size} secteurs sont représentés dans votre sélection.`,
    ];
  }, [options, tickers]);

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
    {portfolioError && <p role="alert" className="rounded-xl border border-amber-300/15 bg-amber-300/[0.05] px-4 py-3 text-sm leading-relaxed text-amber-100">Connectez-vous pour sauvegarder vos portfolios. <Link href="/login?next=/portfolio" className="font-semibold underline underline-offset-2">Ouvrir la connexion</Link></p>}

    <section aria-labelledby="portfolio-ideas-title" className="space-y-3">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <div className="flex items-center gap-2"><Sparkles size={16} className="text-emerald-300" /><h2 id="portfolio-ideas-title" className="text-sm font-semibold uppercase tracking-[0.14em] text-slate-300">Portfolio ideas</h2></div>
          <p className="mt-1 text-sm text-slate-500">Des points de départ lisibles, prévisualisables et créables en un clic.</p>
        </div>
        <Link href="/optimizer" className="text-xs font-medium text-emerald-200 hover:text-emerald-100">Voir les calculs avancés <ArrowRight size={13} className="ml-1 inline" /></Link>
      </div>
      <div className="grid grid-cols-1 gap-3 lg:grid-cols-3">
        {PORTFOLIO_IDEAS.map((idea) => (
          <Card key={idea.id} className="flex flex-col !p-4 sm:!p-5">
            <div className="flex items-start justify-between gap-3"><div><h3 className="font-semibold text-slate-100">{idea.name}</h3><p className="mt-1 text-xs text-emerald-200">{idea.tagline}</p></div><span className={`rounded-full px-2 py-1 text-[10px] font-medium ${idea.accent === "violet" ? "bg-violet-300/10 text-violet-200" : idea.accent === "cyan" ? "bg-cyan-300/10 text-cyan-200" : "bg-emerald-300/10 text-emerald-200"}`}>{idea.risk}</span></div>
            <div className="mt-4 flex flex-wrap gap-1.5">{idea.tickers.map((ticker) => <span key={ticker} className="rounded-md bg-white/[0.05] px-2 py-1 font-mono text-[10px] text-slate-300">{ticker} {formatIdeaWeight(idea.weights[ticker])}</span>)}</div>
            <p className="mt-4 flex-1 text-xs leading-relaxed text-slate-400">{idea.description}</p>
            <div className="mt-4 flex gap-2"><button type="button" onClick={() => setPreviewIdea(idea)} className="inline-flex h-9 flex-1 items-center justify-center gap-1.5 rounded-lg border border-white/[0.09] bg-white/[0.025] px-2 text-xs font-medium text-slate-300 transition hover:bg-white/[0.07] hover:text-white"><Eye size={14} />Aperçu</button><button type="button" onClick={() => void createFromIdea(idea)} disabled={creatingIdeaId !== null} className="inline-flex h-9 flex-1 items-center justify-center gap-1.5 rounded-lg bg-emerald-400 px-2 text-xs font-semibold text-slate-950 transition hover:bg-emerald-300 disabled:opacity-50">{creatingIdeaId === idea.id ? <LoaderCircle size={14} className="animate-spin" /> : <Zap size={14} />}{creatingIdeaId === idea.id ? "Création…" : "Créer"}</button></div>
          </Card>
        ))}
      </div>
    </section>

    <Card title="Suggestions pour votre sélection" className="border-cyan-300/10 bg-cyan-300/[0.025]">
      <div className="grid gap-2 sm:grid-cols-2">{suggestions.map((suggestion) => <p key={suggestion} className="flex items-start gap-2 text-xs leading-relaxed text-slate-300"><Check size={14} className="mt-0.5 shrink-0 text-cyan-300" />{suggestion}</p>)}</div>
    </Card>

    {previewIdea && <div role="presentation" className="fixed inset-0 z-[100] grid place-items-center bg-black/70 p-4 backdrop-blur-sm" onMouseDown={(event) => { if (event.target === event.currentTarget) setPreviewIdea(null); }}>
      <div role="dialog" aria-modal="true" aria-labelledby="portfolio-preview-title" className="max-h-[90vh] w-full max-w-xl overflow-y-auto rounded-2xl border border-white/[0.1] bg-[#111720] p-5 shadow-2xl sm:p-6">
        <div className="flex items-start justify-between gap-4"><div><p className="text-[10px] font-semibold uppercase tracking-[0.16em] text-emerald-300">Aperçu de l’idée</p><h2 id="portfolio-preview-title" className="mt-1 text-2xl font-semibold text-slate-50">{previewIdea.name}</h2><p className="mt-2 text-sm leading-relaxed text-slate-400">{previewIdea.description}</p></div><button type="button" aria-label="Fermer l’aperçu" onClick={() => setPreviewIdea(null)} className="rounded-lg px-2 py-1 text-xl text-slate-500 hover:bg-white/[0.06] hover:text-white">×</button></div>
        <div className="mt-5 grid grid-cols-2 gap-3 sm:grid-cols-3"><div className="rounded-xl bg-white/[0.04] p-3"><p className="text-[10px] uppercase tracking-wider text-slate-500">Profil</p><p className="mt-1 text-sm font-medium text-slate-200">{previewIdea.risk}</p></div><div className="rounded-xl bg-white/[0.04] p-3"><p className="text-[10px] uppercase tracking-wider text-slate-500">Horizon</p><p className="mt-1 text-sm font-medium text-slate-200">{previewIdea.horizon}</p></div><div className="rounded-xl bg-white/[0.04] p-3"><p className="text-[10px] uppercase tracking-wider text-slate-500">Lignes</p><p className="mt-1 text-sm font-medium text-slate-200">{previewIdea.tickers.length} actifs</p></div></div>
        <div className="mt-5 space-y-3"><p className="text-xs font-semibold uppercase tracking-[0.14em] text-slate-400">Allocation de départ</p>{previewIdea.tickers.map((ticker) => <div key={ticker}><div className="mb-1 flex justify-between text-xs"><span className="font-mono font-medium text-slate-200">{ticker}</span><span className="font-mono text-emerald-200">{formatIdeaWeight(previewIdea.weights[ticker])}</span></div><div className="h-2 overflow-hidden rounded-full bg-white/[0.06]"><div className="h-full rounded-full bg-emerald-300" style={{ width: `${previewIdea.weights[ticker] * 100}%` }} /></div></div>)}</div>
        <p className="mt-5 rounded-xl border border-amber-300/10 bg-amber-300/[0.04] p-3 text-xs leading-relaxed text-amber-100/80">Cette allocation est un point de départ pédagogique, pas une recommandation personnalisée. Vérifiez les données et ajustez-la dans l’Optimizer.</p>
        <div className="mt-5 grid gap-2 sm:grid-cols-2"><button type="button" onClick={() => useIdeaAsDraft(previewIdea)} className="inline-flex h-11 items-center justify-center gap-2 rounded-xl border border-white/[0.09] bg-white/[0.025] px-4 text-sm font-medium text-slate-200 hover:bg-white/[0.07]">Personnaliser d’abord</button><button type="button" onClick={() => void createFromIdea(previewIdea)} disabled={creatingIdeaId !== null} className="inline-flex h-11 items-center justify-center gap-2 rounded-xl bg-emerald-400 px-4 text-sm font-semibold text-slate-950 hover:bg-emerald-300 disabled:opacity-50">{creatingIdeaId === previewIdea.id ? <LoaderCircle size={15} className="animate-spin" /> : <Zap size={15} />}{creatingIdeaId === previewIdea.id ? "Création…" : "Créer ce portfolio"}</button></div>
      </div>
    </div>}
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
