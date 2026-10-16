"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const ITEMS = [
  { href: "/", label: "Scan" },
  { href: "/leaderboard", label: "Leaderboard" },
] as const;

// Bottom navigation (context.md Section 19, screen 1): Scan and Leaderboard.
export function BottomNav() {
  const pathname = usePathname();
  return (
    <nav aria-label="Main" className="sticky bottom-0 border-t border-ink/10 bg-white">
      <ul className="mx-auto flex max-w-xl">
        {ITEMS.map(({ href, label }) => {
          const current = pathname === href;
          return (
            <li key={href} className="flex-1">
              <Link
                href={href}
                aria-current={current ? "page" : undefined}
                className={`flex min-h-12 items-center justify-center font-bold ${
                  current ? "text-bin-blue underline underline-offset-4" : "text-muted"
                }`}
              >
                {label}
              </Link>
            </li>
          );
        })}
      </ul>
    </nav>
  );
}
