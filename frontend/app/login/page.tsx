"use client";

import { FormEvent, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { ArrowRight, KeyRound, LoaderCircle, Mail, ShieldCheck } from "lucide-react";
import Card from "@/components/Card";
import Logo from "@/components/ui/Logo";
import { usePortfolios } from "@/components/PortfolioProvider";
import { connectApiSession } from "@/lib/api";
import { getSupabaseBrowserClient } from "@/lib/supabase-browser";

type AuthView = "login" | "signup" | "forgot";

function authRedirectUrl(path: string): string {
  // Set NEXT_PUBLIC_SITE_URL when local development should send auth emails
  // back to the deployed application instead of localhost.
  const siteUrl = process.env.NEXT_PUBLIC_SITE_URL?.trim();
  return new URL(path, siteUrl || window.location.origin).toString();
}

function destinationAfterLogin(): string {
  const candidate = new URLSearchParams(window.location.search).get("next");
  return candidate?.startsWith("/") && !candidate.startsWith("//") ? candidate : "/advisor";
}

export default function LoginPage() {
  const router = useRouter();
  const { refreshPortfolios } = usePortfolios();
  const [view, setView] = useState<AuthView>("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  async function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      const auth = getSupabaseBrowserClient().auth;
      if (view === "forgot") {
        const redirectTo = authRedirectUrl("/auth/callback?next=/auth/reset-password");
        const { error: resetError } = await auth.resetPasswordForEmail(email.trim(), { redirectTo });
        if (resetError) throw resetError;
        setNotice("If an account exists for this address, a password reset link is on its way.");
        return;
      }

      if (view === "signup") {
        const emailRedirectTo = authRedirectUrl("/auth/callback?next=/advisor");
        const { data, error: signupError } = await auth.signUp({
          email: email.trim(),
          password,
          options: { emailRedirectTo },
        });
        if (signupError) throw signupError;
        if (!data.session) {
          setNotice("Check your inbox and confirm your email address. The confirmation link will bring you back to MoatCheck.");
          return;
        }
        await connectApiSession(data.session.access_token);
      } else {
        const { data, error: loginError } = await auth.signInWithPassword({ email: email.trim(), password });
        if (loginError) throw loginError;
        await connectApiSession(data.session.access_token);
      }

      await refreshPortfolios();
      router.replace(destinationAfterLogin());
      router.refresh();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Authentication could not be completed.");
    } finally {
      setBusy(false);
    }
  }

  const title = view === "login" ? "Welcome back" : view === "signup" ? "Create your account" : "Reset your password";
  const submitLabel = view === "login" ? "Sign in" : view === "signup" ? "Create account" : "Send reset link";
  const inputClass = "mt-2 h-11 w-full rounded-xl border border-white/[0.08] bg-[#0B0F14]/75 px-3.5 text-sm text-slate-100 outline-none placeholder:text-slate-600 focus:border-emerald-400/40 focus:ring-2 focus:ring-emerald-400/10";

  return (
    <div className="mx-auto flex min-h-[68vh] max-w-lg items-center justify-center py-8">
      <Card className="w-full border-white/[0.08] p-6 sm:p-8">
        <div className="mb-7 flex items-center gap-3"><span className="grid h-12 w-12 place-items-center rounded-[18px] bg-[#0B0F14]"><Logo size={36} /></span><div><p className="text-sm font-semibold text-white">MoatCheck</p><p className="text-[10px] uppercase tracking-[0.16em] text-slate-500">Private portfolio workspace</p></div></div>
        <h1 className="text-2xl font-semibold tracking-tight text-white">{title}</h1>
        <p className="mt-2 text-sm leading-relaxed text-slate-400">Use one MoatCheck account to securely access your saved portfolio data.</p>

        <form onSubmit={(event) => void onSubmit(event)} className="mt-6 space-y-4">
          <label className="block text-xs font-medium text-slate-300">Email address<input className={inputClass} autoComplete="email" type="email" required value={email} onChange={(event) => setEmail(event.target.value)} placeholder="you@example.com" /></label>
          {view !== "forgot" && <label className="block text-xs font-medium text-slate-300">Password<input className={inputClass} autoComplete={view === "signup" ? "new-password" : "current-password"} type="password" minLength={8} required value={password} onChange={(event) => setPassword(event.target.value)} placeholder="At least 8 characters" /></label>}
          {error && <p role="alert" className="rounded-xl border border-rose-400/15 bg-rose-400/[0.06] px-3 py-2.5 text-xs text-rose-200">{error}</p>}
          {notice && <p role="status" className="rounded-xl border border-emerald-400/15 bg-emerald-400/[0.06] px-3 py-2.5 text-xs leading-relaxed text-emerald-200">{notice}</p>}
          <button type="submit" disabled={busy} className="flex h-11 w-full items-center justify-center gap-2 rounded-xl bg-[#22C55E] px-4 text-sm font-semibold text-[#07120B] shadow-neon-emerald transition hover:bg-[#4ade80] disabled:cursor-wait disabled:opacity-60">
            {busy ? <LoaderCircle size={16} className="animate-spin" /> : view === "forgot" ? <Mail size={15} /> : <ArrowRight size={15} />}{busy ? "Please wait…" : submitLabel}
          </button>
        </form>

        <div className="mt-5 flex flex-wrap items-center justify-between gap-3 text-xs">
          {view === "login" ? <><button type="button" onClick={() => { setView("forgot"); setError(null); setNotice(null); }} className="text-slate-400 hover:text-cyan-200">Forgot password?</button><button type="button" onClick={() => { setView("signup"); setError(null); setNotice(null); }} className="font-medium text-emerald-300 hover:text-emerald-200">Create an account</button></> : <button type="button" onClick={() => { setView("login"); setError(null); setNotice(null); }} className="font-medium text-emerald-300 hover:text-emerald-200">Back to sign in</button>}
        </div>
        <p className="mt-6 flex items-start gap-2 border-t border-white/[0.07] pt-4 text-[10px] leading-relaxed text-slate-500"><ShieldCheck size={13} className="mt-0.5 shrink-0 text-cyan-300" />Your user session is verified by Supabase. MoatCheck exchanges it for an API-only HttpOnly cookie; never enter a Supabase service-role key here.</p>
        <Link href="/privacy" className="mt-4 inline-flex items-center gap-1 text-[10px] text-slate-600 hover:text-slate-300"><KeyRound size={11} />Privacy and session details</Link>
      </Card>
    </div>
  );
}
