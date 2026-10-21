import Link from "next/link";

import { BinMark, ProfileIcon } from "@/components/icons";
import { PointsPill } from "@/components/points-pill";
import { loadMe } from "@/lib/supabase/leaderboard";

// Top bar with the wordmark and the user's points (context.md Section 19, screen 1).
// With `showProfile`, the user's pseudonym also shows on the right, once they have one.
export async function SiteHeader({ showProfile = false }: { showProfile?: boolean }) {
  const me = await loadMe().catch(() => null);
  return (
    <HeaderBar
      points={me?.points ?? null}
      pseudonym={showProfile ? (me?.pseudonym ?? null) : null}
    />
  );
}

/** The bar itself, without data loading; loading skeletons render it with no points. */
export function HeaderBar({
  points,
  pseudonym = null,
}: {
  points: number | null;
  pseudonym?: string | null;
}) {
  return (
    <header className="sticky top-0 z-20 bg-ink/95 text-light shadow-[0_8px_24px_-16px_rgb(16_35_63/0.8)] backdrop-blur">
      <div className="mx-auto flex max-w-xl items-center justify-between gap-3 px-4 py-3">
        <Link href="/" className="flex min-h-11 shrink-0 items-center gap-2 rounded-lg">
          <span className="grid size-8 place-items-center rounded-lg bg-bin-blue">
            <BinMark className="size-5" />
          </span>
          {/* On narrow phones only the logo shows, so the pseudonym fits; the name stays for screen readers. */}
          <span
            className={`font-heading text-xl font-bold tracking-tight ${pseudonym ? "max-[400px]:sr-only" : ""}`}
          >
            Bin Buddy
          </span>
        </Link>
        <div className="flex min-w-0 items-center gap-2">
          {pseudonym && (
            <Link
              href="/profile"
              aria-label={`Your profile: ${pseudonym}`}
              className="flex min-h-11 min-w-0 items-center gap-1.5 rounded-lg"
            >
              <span className="grid size-7 shrink-0 place-items-center rounded-full bg-light text-ink">
                <ProfileIcon className="size-4" />
              </span>
              <span className="truncate text-sm font-bold">{pseudonym}</span>
            </Link>
          )}
          {points !== null && (
            <span className="shrink-0">
              <PointsPill points={points} />
            </span>
          )}
        </div>
      </div>
    </header>
  );
}
