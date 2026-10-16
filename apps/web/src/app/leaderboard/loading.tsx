import { BottomNav } from "@/components/bottom-nav";
import { HeaderBar } from "@/components/site-header";

// Skeleton while the leaderboard loads from Supabase: same shape as the real list, no layout shift.
export default function LeaderboardLoading() {
  return (
    <>
      <HeaderBar points={null} />
      <LeaderboardSkeleton />
      <BottomNav />
    </>
  );
}

function LeaderboardSkeleton() {
  return (
    <main
      aria-busy="true"
      aria-label="Loading the leaderboard"
      className="mx-auto w-full max-w-xl flex-1 space-y-5 px-4 py-6"
    >
      <div className="space-y-2">
        <div className="h-4 w-40 animate-pulse rounded bg-ink/10" />
        <div className="h-10 w-56 animate-pulse rounded-lg bg-ink/10" />
      </div>
      <div className="h-5 w-full animate-pulse rounded bg-ink/10" />
      <div className="card divide-y divide-paper">
        {[0, 1, 2, 3].map((i) => (
          <div key={i} className="flex items-center gap-4 px-4 py-3.5">
            <div className="size-9 animate-pulse rounded-full bg-ink/10" />
            <div className="flex-1 space-y-2">
              <div className="h-4 w-1/2 animate-pulse rounded bg-ink/10" />
              <div className="h-1.5 w-full animate-pulse rounded bg-ink/10" />
            </div>
            <div className="h-6 w-16 animate-pulse rounded bg-ink/10" />
          </div>
        ))}
      </div>
    </main>
  );
}
