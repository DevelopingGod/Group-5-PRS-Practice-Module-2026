// Ranking for the leaderboard page (context.md Section 19, screen 9). Pure, so it can be tested.

/** A row of the Supabase `leaderboard` view: pseudonyms and points only. */
export type LeaderboardRow = { pseudonym: string; total_points: number };

export type RankedRow = LeaderboardRow & { rank: number; isMe: boolean };

/**
 * Sorts by points (highest first), then pseudonym, with shared ranks for ties (1, 2, 2, 4).
 * Users with no points yet are left out.
 */
export function rankRows(rows: LeaderboardRow[], myPseudonym: string | null): RankedRow[] {
  const sorted = rows
    .filter((r) => r.total_points > 0)
    .sort((a, b) => b.total_points - a.total_points || a.pseudonym.localeCompare(b.pseudonym));
  return sorted.map((row) => ({
    ...row,
    rank: sorted.findIndex((r) => r.total_points === row.total_points) + 1,
    isMe: row.pseudonym === myPseudonym,
  }));
}
