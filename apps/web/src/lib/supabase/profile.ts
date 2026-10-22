// Server-side reads for the profile page. Uses the service role, so every query filters by the
// signed-in user's id explicitly (the leaderboard view by the user's own pseudonym, as loadMe()).
import "server-only";

import { connection } from "next/server";

import { historyItems, type HistoryItem, type ScanSummary } from "@/lib/profile";
import { pointsSince } from "@/lib/supabase/disposals";
import { adminClient, currentUserId, supabaseConfigured } from "@/lib/supabase/server";
import { startOfSingaporeDay } from "@/lib/verify-record";

export type Profile = {
  pseudonym: string;
  /** Capped total from the leaderboard view, the same number as the header. */
  totalPoints: number;
  scans: number;
  verifiedDisposals: number;
  /** Points earned since midnight in Singapore, before the daily cap is applied. */
  pointsToday: number;
  /** The latest finished scans, newest first: results only, never photos. */
  history: HistoryItem[];
};

const HISTORY_LENGTH = 20;

/** The signed-in user's profile, or null before their first scan. Throws if Supabase is down. */
export async function loadProfile(): Promise<Profile | null> {
  await connection(); // per-user data: never prerender
  if (!supabaseConfigured()) throw new Error("Supabase is not configured");
  const userId = await currentUserId();
  if (!userId) return null;
  const db = adminClient();
  const { data: profile, error } = await db
    .from("profiles")
    .select("pseudonym")
    .eq("id", userId)
    .maybeSingle<{ pseudonym: string }>();
  if (error) throw new Error(`profile lookup failed: ${error.message}`);
  if (!profile) return null;

  const [total, scans, disposals, pointsToday, recent] = await Promise.all([
    db
      .from("leaderboard")
      .select("total_points")
      .eq("pseudonym", profile.pseudonym)
      .maybeSingle<{ total_points: number }>(),
    db.from("scans").select("id", { count: "exact", head: true }).eq("user_id", userId),
    db
      .from("disposals")
      .select("id", { count: "exact", head: true })
      .eq("user_id", userId)
      .eq("passed", true),
    pointsSince(userId, startOfSingaporeDay(new Date())),
    db
      .from("scans")
      .select("id, material, channel, bin_key, created_at")
      .eq("user_id", userId)
      .not("channel", "is", null) // still waiting for an answer: no result yet
      .order("created_at", { ascending: false })
      .limit(HISTORY_LENGTH)
      .returns<ScanSummary[]>(),
  ]);
  for (const r of [total, scans, disposals, recent]) {
    if (r.error) throw new Error(`profile counts failed: ${r.error.message}`);
  }
  const recentScans = recent.data ?? [];
  const verified = recentScans.length
    ? await db
        .from("disposals")
        .select("scan_id")
        .eq("user_id", userId)
        .eq("passed", true)
        .in(
          "scan_id",
          recentScans.map((scan) => scan.id),
        )
        .returns<{ scan_id: string }[]>()
    : null;
  if (verified?.error) throw new Error(`profile history failed: ${verified.error.message}`);
  return {
    pseudonym: profile.pseudonym,
    totalPoints: total.data?.total_points ?? 0,
    scans: scans.count ?? 0,
    verifiedDisposals: disposals.count ?? 0,
    pointsToday,
    history: historyItems(recentScans, new Set(verified?.data?.map((d) => d.scan_id))),
  };
}
