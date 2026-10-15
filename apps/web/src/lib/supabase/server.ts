// Server-side Supabase access. The service-role key bypasses RLS, so it must only be used
// in route handlers, and every query here filters by the signed-in user's id explicitly.
import "server-only";

import { createServerClient } from "@supabase/ssr";
import { createClient, type SupabaseClient } from "@supabase/supabase-js";
import { cookies } from "next/headers";

import type { VerdictResult } from "@/lib/contracts";
import { randomPseudonym, verdictUpdate, type ScanRow } from "@/lib/scan-record";

const url = () => process.env.NEXT_PUBLIC_SUPABASE_URL;
const anonKey = () => process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;
const serviceKey = () => process.env.SUPABASE_SERVICE_ROLE_KEY;

/** False in local development without Supabase; scans are then simply not saved. */
export function supabaseConfigured(): boolean {
  return Boolean(url() && anonKey() && serviceKey());
}

/** The anonymous user behind this request (validated by Supabase Auth), or null. */
export async function currentUserId(): Promise<string | null> {
  if (!supabaseConfigured()) return null;
  const store = await cookies();
  const client = createServerClient(url()!, anonKey()!, {
    cookies: {
      getAll: () => store.getAll(),
      setAll: (toSet) => {
        for (const { name, value, options } of toSet) store.set(name, value, options);
      },
    },
  });
  const { data, error } = await client.auth.getUser();
  return error ? null : (data.user?.id ?? null);
}

export function adminClient(): SupabaseClient {
  return createClient(url()!, serviceKey()!, {
    auth: { persistSession: false, autoRefreshToken: false },
  });
}

/** Creates the user's profile with a random pseudonym on their first scan. */
async function ensureProfile(db: SupabaseClient, userId: string): Promise<void> {
  const { data, error } = await db.from("profiles").select("id").eq("id", userId).maybeSingle();
  if (error) throw new Error(`profile lookup failed: ${error.message}`);
  if (data) return;
  for (let attempt = 0; attempt < 5; attempt++) {
    const { error: insertError } = await db
      .from("profiles")
      .insert({ id: userId, pseudonym: randomPseudonym() });
    if (!insertError) return;
    if (insertError.code !== "23505")
      throw new Error(`profile insert failed: ${insertError.message}`);
    // 23505 = unique violation: pseudonym taken (or a parallel request created the profile).
    const { data: again } = await db.from("profiles").select("id").eq("id", userId).maybeSingle();
    if (again) return;
  }
  throw new Error("could not find a free pseudonym");
}

export async function saveScan(row: ScanRow): Promise<void> {
  const db = adminClient();
  await ensureProfile(db, row.user_id);
  const { error } = await db.from("scans").insert(row);
  if (error) throw new Error(`scan insert failed: ${error.message}`);
}

/** Fills in the verdict once the questions are answered. */
export async function saveVerdict(
  userId: string,
  scanId: string,
  verdict: VerdictResult,
): Promise<void> {
  const { error } = await adminClient()
    .from("scans")
    .update(verdictUpdate(verdict))
    .eq("id", scanId)
    .eq("user_id", userId);
  if (error) throw new Error(`scan update failed: ${error.message}`);
}
