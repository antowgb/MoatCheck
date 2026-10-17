"use client";

import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from "react";
import { api, type SavedPortfolio } from "@/lib/api";

type PortfolioContextValue = {
  portfolios: SavedPortfolio[];
  activePortfolio: SavedPortfolio | null;
  activePortfolioId: string | null;
  setActivePortfolioId: (id: string) => void;
  refreshPortfolios: () => Promise<void>;
  clearPortfolios: () => void;
  loading: boolean;
  error: string | null;
};

const PortfolioContext = createContext<PortfolioContextValue | null>(null);
const STORAGE_KEY = "moatcheck_active_portfolio";

/** Shared portfolio query/cache for the app shell and all workspace screens. */
export function PortfolioProvider({ children }: { children: React.ReactNode }) {
  const generation = useRef(0);
  const [portfolios, setPortfolios] = useState<SavedPortfolio[]>([]);
  const [activePortfolioId, setActiveId] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const clearPortfolios = useCallback(() => {
    ++generation.current;
    setPortfolios([]); setActiveId(null); setLoading(false); setError(null);
    try { window.localStorage.removeItem(STORAGE_KEY); } catch { /* Selection can stay in memory. */ }
  }, []);

  const refreshPortfolios = useCallback(async () => {
    const request = ++generation.current;
    setLoading(true);
    setError(null);
    try {
      const response = await api.listPortfolios();
      if (request !== generation.current) return;
      setPortfolios(response.data);
      let stored: string | null = null;
      try { stored = window.localStorage.getItem(STORAGE_KEY); } catch { /* Storage may be disabled. */ }
      const nextId = response.data.some((item) => item.id === stored)
        ? stored
        : response.data[0]?.id ?? null;
      setActiveId(nextId);
      try {
        if (nextId) window.localStorage.setItem(STORAGE_KEY, nextId);
        else window.localStorage.removeItem(STORAGE_KEY);
      } catch { /* Selection remains usable without browser storage. */ }
    } catch (cause) {
      if (request !== generation.current) return;
      setPortfolios([]);
      setActiveId(null);
      setError(cause instanceof Error ? cause.message : "Unable to load saved portfolios");
    } finally {
      if (request === generation.current) setLoading(false);
    }
  }, []);

  useEffect(() => { void refreshPortfolios(); }, [refreshPortfolios]);

  const setActivePortfolioId = useCallback((id: string) => {
    setActiveId(id);
    try { window.localStorage.setItem(STORAGE_KEY, id); } catch { /* In-memory selection still works. */ }
  }, []);

  const value = useMemo<PortfolioContextValue>(() => ({
    portfolios,
    activePortfolio: portfolios.find((item) => item.id === activePortfolioId) ?? null,
    activePortfolioId,
    setActivePortfolioId,
    refreshPortfolios,
    clearPortfolios,
    loading,
    error,
  }), [portfolios, activePortfolioId, setActivePortfolioId, refreshPortfolios, clearPortfolios, loading, error]);

  return <PortfolioContext.Provider value={value}>{children}</PortfolioContext.Provider>;
}

export function usePortfolios(): PortfolioContextValue {
  const context = useContext(PortfolioContext);
  if (!context) throw new Error("usePortfolios must be used inside PortfolioProvider");
  return context;
}
