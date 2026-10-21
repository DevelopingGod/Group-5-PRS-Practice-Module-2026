import { BottomNav } from "@/components/bottom-nav";
import { HeaderBar } from "@/components/site-header";

// Skeleton while the profile loads from Supabase: same shape as the real page, no layout shift.
export default function ProfileLoading() {
  return (
    <>
      <HeaderBar points={null} />
      <main
        aria-busy="true"
        aria-label="Loading your profile"
        className="mx-auto w-full max-w-xl flex-1 space-y-5 px-4 py-6"
      >
        <div className="space-y-2">
          <div className="h-4 w-32 animate-pulse rounded bg-ink/10" />
          <div className="h-10 w-40 animate-pulse rounded-lg bg-ink/10" />
        </div>
        <div className="card flex items-center gap-4 p-5">
          <div className="size-14 animate-pulse rounded-full bg-ink/10" />
          <div className="flex-1 space-y-2">
            <div className="h-4 w-24 animate-pulse rounded bg-ink/10" />
            <div className="h-7 w-48 animate-pulse rounded bg-ink/10" />
          </div>
        </div>
        <div className="grid grid-cols-3 gap-3">
          {[0, 1, 2].map((i) => (
            <div key={i} className="card space-y-2 p-4">
              <div className="h-3 w-3/4 animate-pulse rounded bg-ink/10" />
              <div className="h-8 w-1/2 animate-pulse rounded bg-ink/10" />
            </div>
          ))}
        </div>
        <div className="card space-y-2 p-4">
          <div className="h-4 w-full animate-pulse rounded bg-ink/10" />
          <div className="h-3 w-full animate-pulse rounded bg-ink/10" />
        </div>
      </main>
      <BottomNav />
    </>
  );
}
