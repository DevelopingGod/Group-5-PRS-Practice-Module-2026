import { loadMe } from "@/lib/supabase/leaderboard";

// Top bar with the wordmark and the user's points (context.md Section 19, screen 1).
export async function SiteHeader() {
  const me = await loadMe().catch(() => null);
  return (
    <header className="bg-ink text-light">
      <div className="mx-auto flex max-w-xl items-center justify-between px-4 py-3">
        <span className="font-heading text-xl font-bold">Bin Buddy</span>
        {me && (
          <span className="rounded-full bg-sticker px-3 py-1 text-sm font-bold text-ink">
            {me.points} {me.points === 1 ? "point" : "points"}
          </span>
        )}
      </div>
    </header>
  );
}
