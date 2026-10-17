"use client";

import { useState } from "react";
import Card from "@/components/Card";
import { deleteMyPersistedData } from "@/lib/api";

export default function PrivacyDataControls() {
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function removeSavedData() {
    const confirmation = window.prompt(
      'This permanently deletes your saved portfolios and linked quant results. Type DELETE to continue.',
    );
    if (confirmation !== "DELETE") return;

    setBusy(true);
    setMessage(null);
    setError(null);
    try {
      const result = await deleteMyPersistedData();
      const counts = result.deleted;
      setMessage(
        `Deleted ${counts.portfolios_deleted} portfolios, ${counts.backtest_runs_deleted} backtests, and ${counts.stress_tests_deleted} stress runs.`,
      );
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Saved data could not be deleted.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <Card>
      <h2 className="text-base font-semibold">Delete saved quant data</h2>
      <p className="mt-2 text-sm leading-relaxed text-slate-600 dark:text-slate-400">
        This removes saved portfolios, their linked backtests and stress runs, and advisor recommendations. It does not delete your Supabase login account. This action cannot be undone.
      </p>
      <button
        type="button"
        onClick={() => void removeSavedData()}
        disabled={busy}
        className="mt-4 rounded-lg border border-rose-500/40 px-3 py-2 text-sm font-medium text-rose-700 hover:bg-rose-500/10 disabled:cursor-wait disabled:opacity-50 dark:text-rose-300"
      >
        {busy ? "Deleting saved data…" : "Delete my saved quant data"}
      </button>
      {message && <p role="status" className="mt-3 text-sm text-emerald-700 dark:text-emerald-300">{message}</p>}
      {error && <p role="alert" className="mt-3 text-sm text-rose-700 dark:text-rose-300">{error}</p>}
    </Card>
  );
}
