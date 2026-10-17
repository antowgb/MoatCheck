"use client";

import { CircleCheck, CircleX, LoaderCircle, LogIn, LogOut, UserRound } from "lucide-react";
import { useEffect, useState } from "react";
import { usePathname, useRouter } from "next/navigation";
import { api, clearApiSession, connectApiSession, getApiSession, type ClientTelemetry } from "@/lib/api";
import { usePortfolios } from "@/components/PortfolioProvider";
import Logo from "@/components/ui/Logo";
import { getSupabaseBrowserClient } from "@/lib/supabase-browser";
import type { Session } from "@supabase/supabase-js";

type Health = "checking" | "online" | "offline";

function freshnessLabel(value: string | null): string {
  if (!value) return "No market snapshot yet";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "Snapshot time unavailable" : `Snapshot ${date.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" })}`;
}

export default function QuantHeader() {
  const { portfolios, activePortfolioId, setActivePortfolioId, refreshPortfolios, clearPortfolios, loading: portfoliosLoading } = usePortfolios();
  const router = useRouter();
  const pathname = usePathname();
  const [health, setHealth] = useState<Health>("checking");
  const [telemetry, setTelemetry] = useState<ClientTelemetry | null>(null);
  const [sessionError, setSessionError] = useState<string | null>(null);
  const [sessionBusy, setSessionBusy] = useState(false);
  const [sessionEmail, setSessionEmail] = useState<string | null>(null);

  useEffect(() => { setTelemetry(null); }, [activePortfolioId]);

  const toggleUserSession = async () => {
    if (!sessionEmail) {
      const next = pathname.startsWith("/login") || pathname.startsWith("/auth/") ? "/advisor" : pathname;
      router.push(`/login?next=${encodeURIComponent(next)}`);
      return;
    }
    setSessionBusy(true); setSessionError(null);
    try {
      try { await getSupabaseBrowserClient().auth.signOut(); } catch { /* Clear the API session even if Auth is unreachable. */ }
      await clearApiSession();
      setSessionEmail(null); clearPortfolios(); setTelemetry(null);
    } catch {
      setSessionError("Could not finish signing out. Please try again.");
    } finally { setSessionBusy(false); }
  };

  useEffect(() => {
    try { window.localStorage.removeItem("moatcheck_access_token"); } catch { /* Storage may be disabled. */ }
    let active = true;
    const onTelemetry = (event: Event) => {
      const detail = (event as CustomEvent<ClientTelemetry>).detail;
      if (active) setTelemetry((previous) => ({
        roundTripLatencyMs: detail.roundTripLatencyMs ?? previous?.roundTripLatencyMs ?? 0,
        serverCalculationMs: detail.serverCalculationMs ?? previous?.serverCalculationMs,
        snapshotTimestamp: detail.snapshotTimestamp ?? previous?.snapshotTimestamp,
        modelVersion: detail.modelVersion ?? previous?.modelVersion,
      }));
    };
    window.addEventListener("moatcheck:telemetry", onTelemetry);
    const poll = async () => {
      try { const result = await api.health(); if (active) setHealth(result.status === "ok" ? "online" : "offline"); }
      catch { if (active) setHealth("offline"); }
    };
    void poll();
    const interval = window.setInterval(poll, 30_000);
    return () => { active = false; window.removeEventListener("moatcheck:telemetry", onTelemetry); window.clearInterval(interval); };
  }, []);

  useEffect(() => {
    let active = true;
    let lastExchangedToken: string | null = null;
    let exchangeController: AbortController | null = null;
    let signedOut = false;
    let unsubscribe = () => {};
    const exchangeSession = async (session: Session) => {
      if (!active || signedOut) return;
      setSessionEmail(session.user.email ?? null);
      if (!session.access_token || session.access_token === lastExchangedToken) return;
      lastExchangedToken = session.access_token;
      exchangeController?.abort();
      const controller = new AbortController();
      exchangeController = controller;
      try {
        await connectApiSession(session.access_token, controller.signal);
        if (active && !controller.signal.aborted) { setSessionError(null); await refreshPortfolios(); }
      } catch {
        if (active && !controller.signal.aborted) {
          lastExchangedToken = null;
          setSessionError("Could not connect your account to the API. Please sign in again.");
        }
      }
    };
    try {
      const auth = getSupabaseBrowserClient().auth;
      const { data } = auth.onAuthStateChange((event, session) => {
        if (!active) return;
        setSessionEmail(session?.user.email ?? null);
        if (event === "SIGNED_OUT") {
          signedOut = true; exchangeController?.abort();
          lastExchangedToken = null;
          clearPortfolios(); setTelemetry(null);
          void clearApiSession().catch(() => { if (active) setSessionError("Could not clear the API session. Please sign out again."); });
        }
        if (session && (event === "INITIAL_SESSION" || event === "SIGNED_IN" || event === "TOKEN_REFRESHED")) {
          signedOut = false;
          window.setTimeout(() => { void exchangeSession(session); }, 0);
        }
      });
      unsubscribe = () => data.subscription.unsubscribe();
      void auth.getSession().then(async ({ data: sessionData }) => {
        if (!active) return;
        if (sessionData.session) await exchangeSession(sessionData.session);
        else {
          const apiSession = await getApiSession();
          if (active) setSessionEmail(apiSession?.email ?? null);
        }
      }).catch(() => { if (active) setSessionEmail(null); });
    } catch { setSessionEmail(null); }
    return () => { active = false; exchangeController?.abort(); unsubscribe(); };
  }, [refreshPortfolios, clearPortfolios]);

  const apiState = {
    checking: { label: "Checking API", icon: LoaderCircle, color: "text-amber-300", animate: true },
    online: { label: "API healthy", icon: CircleCheck, color: "text-emerald-300", animate: false },
    offline: { label: "API offline", icon: CircleX, color: "text-rose-300", animate: false },
  }[health];
  const StatusIcon = apiState.icon;
  const telemetryTitle = [apiState.label, telemetry ? `Browser round trip ${telemetry.roundTripLatencyMs.toFixed(0)} ms` : "No recent API response",
    telemetry?.serverCalculationMs == null ? null : `Server calculation ${telemetry.serverCalculationMs.toFixed(1)} ms`,
    freshnessLabel(telemetry?.snapshotTimestamp ?? null), telemetry?.modelVersion ? `Model ${telemetry.modelVersion}` : null].filter(Boolean).join(" · ");

  return (
    <header className="sticky top-2 z-20 mx-1 flex min-h-[64px] flex-wrap items-center justify-between gap-3 rounded-[22px] border border-white/[0.07] bg-[#111720]/95 px-3 py-2 shadow-[0_18px_55px_-40px_rgba(0,0,0,.9)] backdrop-blur-2xl sm:px-4 xl:mx-2">
      <div className="flex min-w-0 items-center gap-2.5">
        <span className="grid h-9 w-9 shrink-0 place-items-center rounded-[14px] bg-[#0B0F14]"><Logo size={27} /></span>
        <div className="min-w-0"><p className="truncate text-sm font-semibold tracking-tight text-slate-100">MoatCheck</p><p className="hidden text-[9px] uppercase tracking-[0.15em] text-slate-500 sm:block">Research workspace</p></div>
      </div>
      {sessionError && <p role="alert" className="text-xs text-rose-300">{sessionError}</p>}
      <div className="flex min-w-0 flex-1 items-center justify-end gap-2 sm:gap-3">
        <label className="relative min-w-0 max-w-[230px] flex-1 sm:flex-none">
          <span className="sr-only">Active portfolio</span>
          <select aria-label="Active portfolio" value={activePortfolioId ?? ""} onChange={(event) => setActivePortfolioId(event.target.value)} disabled={portfoliosLoading || portfolios.length === 0} className="quant-control h-9 w-full truncate py-1.5 pl-3 pr-7 text-[11px] font-semibold sm:w-[190px] sm:text-xs">
            {portfolios.length ? portfolios.map((item) => <option key={item.id} value={item.id}>{item.name}</option>) : <option value="">{portfoliosLoading ? "Loading portfolios…" : "No saved portfolio"}</option>}
          </select>
        </label>
        <div title={telemetryTitle} className="inline-flex h-9 shrink-0 items-center gap-1.5 rounded-full border border-white/[0.07] bg-white/[0.025] px-2.5 text-[10px] text-slate-400 sm:px-3">
          <StatusIcon size={13} className={`${apiState.color} ${apiState.animate ? "animate-spin" : ""}`} />
          <span className="hidden sm:inline">{apiState.label}</span>
          <span className="font-mono text-cyan-200">{telemetry ? `${telemetry.roundTripLatencyMs.toFixed(0)} ms` : "N/A"}</span>
        </div>
        <button type="button" disabled={sessionBusy} onClick={() => { void toggleUserSession(); }} className="inline-flex h-9 max-w-[160px] shrink-0 items-center gap-1.5 rounded-full border border-white/[0.08] bg-white/[0.03] px-2.5 text-slate-300 transition hover:border-cyan-300/25 hover:text-white disabled:opacity-60" aria-label={sessionEmail ? `Sign out ${sessionEmail}` : "Sign in to MoatCheck"} title={sessionEmail ?? "Sign in to MoatCheck"}>
          <span className="grid h-6 w-6 shrink-0 place-items-center rounded-full border border-cyan-300/15 bg-cyan-300/[0.08] text-cyan-200">{sessionEmail ? <UserRound size={13} /> : <LogIn size={13} />}</span>
          <span className="hidden truncate text-[10px] font-medium sm:block">{sessionBusy ? "Working…" : sessionEmail ?? "Sign in"}</span>
          {sessionEmail && <LogOut size={12} className="hidden shrink-0 text-slate-500 sm:block" />}
        </button>
      </div>
    </header>
  );
}
