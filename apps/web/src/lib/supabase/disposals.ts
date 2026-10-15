// Server-side reads and writes for verification. Uses the service role, so every query
// filters by the signed-in user's id explicitly.
import "server-only";

import type { BinQrPayload } from "@/lib/contracts";
import { adminClient } from "@/lib/supabase/server";
import type { SavedScan } from "@/lib/verify-record";

/** The user's own scan, or null if it does not exist or belongs to someone else. */
export async function loadScan(userId: string, scanId: string): Promise<SavedScan | null> {
  const { data, error } = await adminClient()
    .from("scans")
    .select("id, site_policy, bin_key, embedding, phash, created_at")
    .eq("id", scanId)
    .eq("user_id", userId)
    .maybeSingle<SavedScan>();
  if (error) throw new Error(`scan lookup failed: ${error.message}`);
  return data;
}

export async function loadBin(binId: string): Promise<BinQrPayload | null> {
  const { data, error } = await adminClient()
    .from("bins")
    .select("bin_id, site_policy, bin_key, location_label")
    .eq("bin_id", binId)
    .maybeSingle<Omit<BinQrPayload, "v">>();
  if (error) throw new Error(`bin lookup failed: ${error.message}`);
  return data ? { v: 1, ...data } : null;
}

export async function hasDisposal(scanId: string): Promise<boolean> {
  const { count, error } = await adminClient()
    .from("disposals")
    .select("id", { count: "exact", head: true })
    .eq("scan_id", scanId);
  if (error) throw new Error(`disposal lookup failed: ${error.message}`);
  return (count ?? 0) > 0;
}

/** Fingerprints of the user's earlier verification photos, newest first. */
export async function recentPhashes(userId: string, limit = 500): Promise<string[]> {
  const { data, error } = await adminClient()
    .from("disposals")
    .select("phash")
    .eq("user_id", userId)
    .order("created_at", { ascending: false })
    .limit(limit)
    .returns<{ phash: string }[]>();
  if (error) throw new Error(`phash lookup failed: ${error.message}`);
  return data.map((r) => r.phash);
}

export async function pointsSince(userId: string, since: Date): Promise<number> {
  const { data, error } = await adminClient()
    .from("disposals")
    .select("points")
    .eq("user_id", userId)
    .eq("passed", true)
    .gte("created_at", since.toISOString())
    .returns<{ points: number }[]>();
  if (error) throw new Error(`points lookup failed: ${error.message}`);
  return data.reduce((sum, r) => sum + r.points, 0);
}

export type DisposalRow = {
  scan_id: string;
  user_id: string;
  bin_id: string;
  similarity: number;
  passed: true;
  points: number;
  phash: string;
};

/** Inserts a passed disposal. Returns false if the item was already rewarded (one per item). */
export async function insertDisposal(row: DisposalRow): Promise<boolean> {
  const { error } = await adminClient().from("disposals").insert(row);
  if (!error) return true;
  if (error.code === "23505") return false; // unique scan_id: a parallel request won
  throw new Error(`disposal insert failed: ${error.message}`);
}
