import { createClient, type SupportedStorage, type SupabaseClient } from "@supabase/supabase-js";

// Keep the short-lived Supabase login material in memory only. The API session
// is held in its HttpOnly cookie, so page reloads never need JS-readable tokens.
const ephemeralAuthStorage = new Map<string, string>();
const ephemeralAuthAdapter: SupportedStorage = {
  getItem(key) {
    return ephemeralAuthStorage.get(key) ?? null;
  },
  setItem(key, value) {
    ephemeralAuthStorage.set(key, value);
  },
  removeItem(key) {
    ephemeralAuthStorage.delete(key);
  },
};

let supabase: SupabaseClient | null = null;

/** Create the public-key Supabase Auth client. User sessions stay tab-scoped. */
export function getSupabaseBrowserClient(): SupabaseClient {
  if (supabase) return supabase;

  const url = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const key = process.env.NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY ?? process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;
  if (!url || !key) {
    throw new Error("Supabase Auth is not configured. Set NEXT_PUBLIC_SUPABASE_URL and NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY in the frontend environment.");
  }

  supabase = createClient(url, key, {
    auth: {
      autoRefreshToken: true,
      detectSessionInUrl: true,
      flowType: "pkce",
      persistSession: true,
      storage: ephemeralAuthAdapter,
    },
  });
  return supabase;
}
