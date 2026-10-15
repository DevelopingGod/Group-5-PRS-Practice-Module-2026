// Server-side reads for the leaderboard and the header points. The `leaderboard` view only
// exposes pseudonyms and points, so nothing else about other users can leak from here.
import "server-only";

import { connection } from "next/server";

import type { LeaderboardRow } from "@/lib/leaderboard";
import { adminClient, currentUserId, supabaseConfigured } from "@/lib/supabase/server";

export type Me = { pseudonym: string; points: number } | null;

/** The signed-in user's pseudonym and capped total, or null before their first scan. */
export async function loadMe(): Promise<Me> {
  await connection(); // per-user data: never prerender
  const userId = await currentUserId();
  if (!userId) return null;
  const db = adminClient();
  const { data: profile } = await db
    .from("profiles")
    .select("pseudonym")
    .eq("id", userId)
    .maybeSingle<{ pseudonym: string }>();
  if (!profile) return null;
  const { data: row } = await db
    .from("leaderboard")
    .select("total_points")
    .eq("pseudonym", profile.pseudonym)
    .maybeSingle<{ total_points: number }>();
  return { pseudonym: profile.pseudonym, points: row?.total_points ?? 0 };
}

/** The top of the leaderboard, or null when Supabase is not set up. */
export async function loadLeaderboard(limit = 50): Promise<LeaderboardRow[] | null> {
  await connection(); // live data: never prerender
  if (!supabaseConfigured()) return null;
  const { data, error } = await adminClient()
    .from("leaderboard")
    .select("pseudonym, total_points")
    .gt("total_points", 0)
    .order("total_points", { ascending: false })
    .order("pseudonym")
    .limit(limit)
    .returns<LeaderboardRow[]>();
  if (error) throw new Error(`leaderboard lookup failed: ${error.message}`);
  return data;
}
