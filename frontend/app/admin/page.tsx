"use client";

import { Check, KeyRound } from "lucide-react";
import { useEffect, useState } from "react";
import AddStockForm from "@/components/AddStockForm";
import AdminQualitativePanels from "@/components/AdminQualitativePanels";
import Card from "@/components/Card";
import PageHeader from "@/components/PageHeader";

const STORAGE_KEY = "moatcheck_admin_key";

export default function AdminPage() {
  const [adminKeyInput, setAdminKeyInput] = useState("");
  const [savedKey, setSavedKey] = useState<string | null>(null);

  useEffect(() => {
    try {
      localStorage.removeItem("moatcheck_admin_key");
      setSavedKey(sessionStorage.getItem(STORAGE_KEY));
    } catch {
      setSavedKey(null);
    }
  }, []);

  const save = (e: React.FormEvent) => {
    e.preventDefault();
    const k = adminKeyInput.trim();
    if (!k) return;
    try {
      sessionStorage.setItem(STORAGE_KEY, k);
    } catch {
      return;
    }
    setSavedKey(k);
    setAdminKeyInput("");
  };

  const forget = () => {
    try { sessionStorage.removeItem(STORAGE_KEY); } catch { /* storage may be disabled */ }
    setSavedKey(null);
  };

  return (
    <div className="w-full">
      <PageHeader
        title="Admin"
        subtitle="Use this page to add companies to track. Your admin key stays in this browser tab and is sent only to the MoatCheck server."
      />

      <Card className="mb-6">
        {savedKey ? (
          <div className="flex items-center justify-between">
            <p className="text-sm text-emerald-700 dark:text-emerald-400 inline-flex items-center gap-1.5">
              <Check size={15} />
              Admin key saved in this tab.
            </p>
            <button
              onClick={forget}
              className="text-sm text-slate-500 dark:text-slate-400 hover:text-red-600 dark:hover:text-rose-400 underline"
            >
              Forget the key
            </button>
          </div>
        ) : (
          <form onSubmit={save} className="flex gap-3 items-end">
            <label className="text-sm flex-1">
              <span className="block text-slate-500 dark:text-slate-400 mb-1 inline-flex items-center gap-1.5">
                <KeyRound size={13} />
                Admin key
              </span>
              <input
                type="password"
                className="border border-slate-200 dark:border-slate-800 rounded-lg px-2 py-1.5 bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100 w-full text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500/30 focus:border-emerald-400 dark:focus:border-emerald-500 transition-colors"
                value={adminKeyInput}
                onChange={(e) => setAdminKeyInput(e.target.value)}
                placeholder="Enter the admin key"
              />
            </label>
            <button
              type="submit"
              className="bg-emerald-600 hover:bg-emerald-700 text-white px-4 py-1.5 rounded-lg text-sm font-medium transition-colors"
            >
              Save
            </button>
          </form>
        )}
      </Card>

      {savedKey ? (
        <>
          <AddStockForm adminKey={savedKey} />
          <AdminQualitativePanels adminKey={savedKey} />
        </>
      ) : (
        <p className="text-sm text-slate-400 dark:text-slate-600">Enter the key above to add a ticker.</p>
      )}
    </div>
  );
}
