import type { Metadata } from "next";

import { BottomNav } from "@/components/bottom-nav";
import { TrophyIcon } from "@/components/icons";
import { SiteHeader } from "@/components/site-header";
import { DAILY_POINTS_CAP } from "@/contracts/constants";
import { rankRows } from "@/lib/leaderboard";
import { loadLeaderboard, loadMe } from "@/lib/supabase/leaderboard";

export const metadata: Metadata = { title: "Leaderboard · Bin Buddy" };

const at = (i: number) => ({ "--i": i }) as React.CSSProperties;

export default async function LeaderboardPage() {
  const [rows, me] = await Promise.all([
    loadLeaderboard().catch(() => null),
    loadMe().catch(() => null),
  ]);
  const ranked = rows ? rankRows(rows, me?.pseudonym ?? null) : null;
  const meListed = ranked?.some((r) => r.isMe) ?? false;
  const top = ranked?.[0]?.total_points ?? 1;

  return (
    <>
      <SiteHeader />
      <main className="stagger mx-auto w-full max-w-xl flex-1 space-y-5 px-4 py-6">
        <header className="space-y-1">
          <p className="text-sm font-bold text-muted">Correct disposals only</p>
          <h1 className="font-heading text-4xl font-bold">Leaderboard</h1>
        </header>
        <p className="text-muted" style={at(1)}>
          Points count correct disposals, up to {DAILY_POINTS_CAP} a day. Only pseudonyms are shown.
        </p>

        {ranked === null ? (
          <p className="card p-4" style={at(2)}>
            The leaderboard is not available right now.
          </p>
        ) : ranked.length === 0 ? (
          <div className="card flex flex-col items-center gap-3 p-8 text-center" style={at(2)}>
            <span className="grid size-14 place-items-center rounded-full bg-sticker text-ink">
              <TrophyIcon className="size-7" />
            </span>
            <p className="font-heading text-xl font-bold">No points yet</p>
            <p className="text-muted">
              Scan an item, then tap &ldquo;I&apos;m at the bin&rdquo; to earn the first one.
            </p>
          </div>
        ) : (
          <ol className="card stagger divide-y divide-paper overflow-hidden" style={at(2)}>
            {ranked.map((row, i) => (
              <li
                key={row.pseudonym}
                aria-current={row.isMe ? "true" : undefined}
                style={at(i)}
                className={`relative flex items-center gap-4 px-4 py-3.5 ${
                  row.isMe ? "bg-sticker text-ink" : ""
                }`}
              >
                <span
                  className={`grid size-9 shrink-0 place-items-center rounded-full font-heading text-lg font-bold ${
                    row.rank <= 3 ? "bg-ink text-sticker" : "bg-paper text-ink"
                  }`}
                >
                  {row.rank}
                </span>
                <span className="min-w-0 flex-1 space-y-1.5">
                  <span className="block truncate font-bold">
                    {row.pseudonym}
                    {row.isMe && <span className="font-normal"> (you)</span>}
                  </span>
                  <span aria-hidden className="block h-1.5 rounded-full bg-ink/10">
                    <span
                      className="block h-full origin-left rounded-full bg-bin-blue animate-grow"
                      style={{
                        width: `${Math.max(6, (row.total_points / top) * 100)}%`,
                        animationDelay: `${200 + i * 70}ms`,
                      }}
                    />
                  </span>
                </span>
                <span className="text-right font-bold tabular-nums">
                  <span className="font-heading text-xl">{row.total_points}</span>{" "}
                  <span className="text-sm">{row.total_points === 1 ? "point" : "points"}</span>
                </span>
              </li>
            ))}
          </ol>
        )}

        {me && !meListed && ranked !== null && (
          <p className="card p-4" style={at(3)}>
            You are <strong>{me.pseudonym}</strong> with {me.points}{" "}
            {me.points === 1 ? "point" : "points"}.
          </p>
        )}
      </main>
      <BottomNav />
    </>
  );
}
