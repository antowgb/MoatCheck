"use client";

import { useEffect, useState } from "react";
import { LoaderCircle, ShieldAlert } from "lucide-react";
import { useRouter } from "next/navigation";
import { connectApiSession } from "@/lib/api";
import { getSupabaseBrowserClient } from "@/lib/supabase-browser";

function safeDestination(value: string | null): string {
  return value?.startsWith("/") && !value.startsWith("//") ? value : "/advisor";
}

export default function AuthCallbackPage() {
  const router = useRouter();
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    let completed = false;
    let auth: ReturnType<typeof getSupabaseBrowserClient>["auth"];

    async function finish(session: { access_token: string } | null, passwordRecovery = false) {
      if (!active || completed || !session) return;
      completed = true;
      const params = new URLSearchParams(window.location.search);
      const recovery = passwordRecovery || params.get("next") === "/auth/reset-password" || params.get("type") === "recovery";
      if (recovery) {
        router.replace("/auth/reset-password");
        return;
      }
      try {
        await connectApiSession(session.access_token);
        router.replace(safeDestination(params.get("next")));
        router.refresh();
      } catch (cause) {
        completed = false;
        if (active) setError(cause instanceof Error ? cause.message : "Could not establish the MoatCheck session.");
      }
    }

    try {
      auth = getSupabaseBrowserClient().auth;
      const { data: listener } = auth.onAuthStateChange((event, session) => {
        if (session && (event === "SIGNED_IN" || event === "PASSWORD_RECOVERY" || event === "INITIAL_SESSION")) {
          window.setTimeout(() => { void finish(session, event === "PASSWORD_RECOVERY"); }, 0);
        }
      });
      void auth.getSession().then(({ data, error: sessionError }) => {
        if (sessionError) throw sessionError;
        if (data.session) void finish(data.session);
        else if (active) setError("The confirmation link is invalid or expired. Return to sign in and request a new link.");
      }).catch((cause: unknown) => {
        if (active) setError(cause instanceof Error ? cause.message : "Could not verify the email link.");
      });
      return () => { active = false; listener.subscription.unsubscribe(); };
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Supabase Auth is not configured.");
      return () => { active = false; };
    }
  }, [router]);

  return <div className="mx-auto flex min-h-[55vh] max-w-xl flex-col items-center justify-center text-center">
    {error ? <><ShieldAlert size={28} className="text-amber-300" /><h1 className="mt-4 text-lg font-semibold text-white">Could not verify your link</h1><p role="alert" className="mt-2 text-sm text-slate-400">{error}</p><a href="/login" className="mt-5 rounded-xl bg-[#22C55E] px-4 py-2.5 text-xs font-semibold text-[#07120B]">Return to sign in</a></> : <><LoaderCircle size={24} className="animate-spin text-cyan-300" /><h1 className="mt-4 text-lg font-semibold text-white">Verifying your email</h1><p className="mt-2 text-sm text-slate-400">We’re securely connecting your MoatCheck session.</p></>}
  </div>;
}
