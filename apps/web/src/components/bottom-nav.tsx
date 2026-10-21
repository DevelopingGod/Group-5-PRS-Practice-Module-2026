"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

import { ProfileIcon, ScanIcon, TrophyIcon } from "@/components/icons";
import { useScan } from "@/components/scan-provider";

const ITEMS = [
  { href: "/", label: "Scan", Icon: ScanIcon },
  { href: "/leaderboard", label: "Leaderboard", Icon: TrophyIcon },
  { href: "/profile", label: "Profile", Icon: ProfileIcon },
] as const;

// Bottom navigation (context.md Section 19, screen 1): Scan, Leaderboard and Profile.
// The current page is marked by a filled pill, bold text and aria-current, never colour alone.
// A dot on Scan, named in its accessible label, shows a scan is still running or needs an answer.
export function BottomNav() {
  const pathname = usePathname();
  const { state } = useScan();
  const pending =
    state.step === "reading"
      ? "scan in progress"
      : state.step === "question"
        ? "question waiting"
        : null;
  return (
    <nav
      aria-label="Main"
      className="sticky bottom-0 z-20 border-t border-ink/10 bg-white/90 pb-[env(safe-area-inset-bottom)] backdrop-blur"
    >
      <ul className="mx-auto flex max-w-xl gap-2 px-4 py-2">
        {ITEMS.map(({ href, label, Icon }) => {
          const current = pathname === href;
          const badge = href === "/" ? pending : null;
          return (
            <li key={href} className="flex-1">
              <Link
                href={href}
                aria-current={current ? "page" : undefined}
                aria-label={badge ? `${label}, ${badge}` : undefined}
                className={`flex min-h-14 flex-col items-center justify-center gap-0.5 rounded-xl text-sm font-bold transition-colors duration-200 ${
                  current ? "bg-ink text-light" : "text-muted hover:bg-paper hover:text-ink"
                }`}
              >
                <span className="relative">
                  <Icon className="size-5" />
                  {badge && (
                    <span
                      aria-hidden
                      className="absolute -top-1 -right-1.5 size-2.5 rounded-full bg-sticker ring-2 ring-ink"
                    />
                  )}
                </span>
                {label}
              </Link>
            </li>
          );
        })}
      </ul>
    </nav>
  );
}
