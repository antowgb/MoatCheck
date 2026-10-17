"use client";

import { FormEvent, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { LoaderCircle, ShieldCheck } from "lucide-react";
import Card from "@/components/Card";
import { usePortfolios } from "@/components/PortfolioProvider";
import { connectApiSession } from "@/lib/api";
import { getSupabaseBrowserClient } from "@/lib/supabase-browser";

export default function ResetPasswordPage() {
  const router = useRouter();
  const { refreshPortfolios } = usePortfolios();
  const [password, setPassword] = useState("");
  const [confirmedPassword, setConfirmedPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [hasRecoverySession, setHasRecoverySession] = useState<boolean | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    try {
      void getSupabaseBrowserClient().auth.getSession().then(({ data, error: sessionError }) => {
        setHasRecoverySession(!sessionError && Boolean(data.session));
      }).catch(() => setHasRecoverySession(false));
    } catch {
      setHasRecoverySession(false);
    }
  }, []);

  async function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError(null);
    if (password !== confirmedPassword) {
      setError("The passwords do not match.");
      return;
    }
    setBusy(true);
    try {
      const auth = getSupabaseBrowserClient().auth;
      const { data, error: updateError } = await auth.updateUser({ password });
      if (updateError) throw updateError;
      if (!data.user) throw new Error("The password was not updated.");
      const { data: sessionData, error: sessionError } = await auth.getSession();
      if (sessionError || !sessionData.session) throw new Error("Your recovery session expired. Request a new password reset link.");
      await connectApiSession(sessionData.session.access_token);
      await refreshPortfolios();
      router.replace("/advisor");
      router.refresh();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not update your password.");
    } finally { setBusy(false); }
  }

  return <div className="mx-auto flex min-h-[60vh] max-w-lg items-center justify-center py-8">
    <Card className="w-full p-6 sm:p-8"><div className="flex items-center gap-2 text-cyan-200"><ShieldCheck size={17} /><span className="text-xs font-medium">Password recovery</span></div><h1 className="mt-4 text-2xl font-semibold text-white">Choose a new password</h1>
      {hasRecoverySession === null ? <p className="mt-4 text-sm text-slate-400">Checking your recovery link…</p> : !hasRecoverySession ? <div><p className="mt-3 text-sm text-slate-400">This recovery link is invalid or has expired.</p><a href="/login" className="mt-5 inline-flex rounded-xl bg-[#22C55E] px-4 py-2.5 text-xs font-semibold text-[#07120B]">Request another link</a></div> : <form onSubmit={(event) => void onSubmit(event)} className="mt-5 space-y-4"><label className="block text-xs font-medium text-slate-300">New password<input autoComplete="new-password" type="password" minLength={8} required value={password} onChange={(event) => setPassword(event.target.value)} className="mt-2 h-11 w-full rounded-xl border border-white/[0.08] bg-[#0B0F14]/75 px-3.5 text-sm text-slate-100 outline-none focus:border-emerald-400/40" /></label><label className="block text-xs font-medium text-slate-300">Confirm new password<input autoComplete="new-password" type="password" minLength={8} required value={confirmedPassword} onChange={(event) => setConfirmedPassword(event.target.value)} className="mt-2 h-11 w-full rounded-xl border border-white/[0.08] bg-[#0B0F14]/75 px-3.5 text-sm text-slate-100 outline-none focus:border-emerald-400/40" /></label>{error && <p role="alert" className="text-xs text-rose-200">{error}</p>}<button disabled={busy} className="flex h-11 w-full items-center justify-center gap-2 rounded-xl bg-[#22C55E] text-sm font-semibold text-[#07120B] disabled:opacity-60">{busy && <LoaderCircle size={15} className="animate-spin" />}{busy ? "Updating…" : "Save new password"}</button></form>}
    </Card>
  </div>;
}
