import Link from "next/link";

import { BinMark } from "@/components/icons";
import { PointsPill } from "@/components/points-pill";
import { loadMe } from "@/lib/supabase/leaderboard";

// Top bar with the wordmark and the user's points (context.md Section 19, screen 1).
export async function SiteHeader() {
  const me = await loadMe().catch(() => null);
  return <HeaderBar points={me?.points ?? null} />;
}

/** The bar itself, without data loading; loading skeletons render it with no points. */
export function HeaderBar({ points }: { points: number | null }) {
  return (
    <header className="sticky top-0 z-20 bg-ink/95 text-light shadow-[0_8px_24px_-16px_rgb(16_35_63/0.8)] backdrop-blur">
      <div className="mx-auto flex max-w-xl items-center justify-between px-4 py-3">
        <Link href="/" className="flex min-h-11 items-center gap-2 rounded-lg">
          <span className="grid size-8 place-items-center rounded-lg bg-bin-blue">
            <BinMark className="size-5" />
          </span>
          <span className="font-heading text-xl font-bold tracking-tight">Bin Buddy</span>
        </Link>
        {points !== null && <PointsPill points={points} />}
      </div>
    </header>
  );
}
