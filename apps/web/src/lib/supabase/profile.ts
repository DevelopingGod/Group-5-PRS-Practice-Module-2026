// Server-side reads for the profile page. Uses the service role, so every query filters by the
// signed-in user's id explicitly (the leaderboard view by the user's own pseudonym, as loadMe()).
import "server-only";

import { connection } from "next/server";

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
};

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

  const [total, scans, disposals, pointsToday] = await Promise.all([
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
  ]);
  for (const r of [total, scans, disposals]) {
    if (r.error) throw new Error(`profile counts failed: ${r.error.message}`);
  }
  return {
    pseudonym: profile.pseudonym,
    totalPoints: total.data?.total_points ?? 0,
    scans: scans.count ?? 0,
    verifiedDisposals: disposals.count ?? 0,
    pointsToday,
  };
}
