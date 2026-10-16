"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

import { ScanIcon, TrophyIcon } from "@/components/icons";

const ITEMS = [
  { href: "/", label: "Scan", Icon: ScanIcon },
  { href: "/leaderboard", label: "Leaderboard", Icon: TrophyIcon },
] as const;

// Bottom navigation (context.md Section 19, screen 1): Scan and Leaderboard.
// The current page is marked by a filled pill, bold text and aria-current, never colour alone.
export function BottomNav() {
  const pathname = usePathname();
  return (
    <nav
      aria-label="Main"
      className="sticky bottom-0 z-20 border-t border-ink/10 bg-white/90 pb-[env(safe-area-inset-bottom)] backdrop-blur"
    >
      <ul className="mx-auto flex max-w-xl gap-2 px-4 py-2">
        {ITEMS.map(({ href, label, Icon }) => {
          const current = pathname === href;
          return (
            <li key={href} className="flex-1">
              <Link
                href={href}
                aria-current={current ? "page" : undefined}
                className={`flex min-h-12 items-center justify-center gap-2 rounded-xl font-bold transition-colors duration-200 ${
                  current ? "bg-ink text-light" : "text-muted hover:bg-paper hover:text-ink"
                }`}
              >
                <Icon className="size-5" />
                {label}
              </Link>
            </li>
          );
        })}
      </ul>
    </nav>
  );
}
