import type { Metadata } from "next";

import { BottomNav } from "@/components/bottom-nav";
import { SiteHeader } from "@/components/site-header";
import { DAILY_POINTS_CAP } from "@/contracts/constants";
import { rankRows } from "@/lib/leaderboard";
import { loadLeaderboard, loadMe } from "@/lib/supabase/leaderboard";

export const metadata: Metadata = { title: "Leaderboard · Bin Buddy" };

export default async function LeaderboardPage() {
  const [rows, me] = await Promise.all([
    loadLeaderboard().catch(() => null),
    loadMe().catch(() => null),
  ]);
  const ranked = rows ? rankRows(rows, me?.pseudonym ?? null) : null;
  const meListed = ranked?.some((r) => r.isMe) ?? false;

  return (
    <>
      <SiteHeader />
      <main className="mx-auto w-full max-w-xl flex-1 space-y-5 px-4 py-6">
        <h1 className="font-heading text-3xl font-bold">Leaderboard</h1>
        <p className="text-muted">
          Points count correct disposals, up to {DAILY_POINTS_CAP} a day. Only pseudonyms are shown.
        </p>

        {ranked === null ? (
          <p className="rounded-2xl bg-white p-4">The leaderboard is not available right now.</p>
        ) : ranked.length === 0 ? (
          <p className="rounded-2xl bg-white p-4">
            No points yet. Scan an item, then tap &ldquo;I&apos;m at the bin&rdquo; to earn the
            first one.
          </p>
        ) : (
          <ol className="divide-y divide-paper overflow-hidden rounded-2xl bg-white">
            {ranked.map((row) => (
              <li
                key={row.pseudonym}
                aria-current={row.isMe ? "true" : undefined}
                className={`flex items-center gap-4 px-4 py-3 ${row.isMe ? "bg-sticker text-ink" : ""}`}
              >
                <span className="w-8 font-heading text-lg font-bold">{row.rank}</span>
                <span className="flex-1">
                  {row.pseudonym}
                  {row.isMe && <strong> (you)</strong>}
                </span>
                <span className="font-bold">
                  {row.total_points} {row.total_points === 1 ? "point" : "points"}
                </span>
              </li>
            ))}
          </ol>
        )}

        {me && !meListed && ranked !== null && (
          <p className="rounded-2xl bg-white p-4">
            You are <strong>{me.pseudonym}</strong> with {me.points}{" "}
            {me.points === 1 ? "point" : "points"}.
          </p>
        )}
      </main>
      <BottomNav />
    </>
  );
}
