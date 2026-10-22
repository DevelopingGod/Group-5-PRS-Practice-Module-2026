import type { Metadata } from "next";
import Link from "next/link";

import { BottomNav } from "@/components/bottom-nav";
import { CameraIcon, CheckIcon, ProfileIcon } from "@/components/icons";
import { ResetIdentity } from "@/components/reset-identity";
import { SiteHeader } from "@/components/site-header";
import { DAILY_POINTS_CAP } from "@/contracts/constants";
import { countText, todayPoints, type HistoryItem } from "@/lib/profile";
import { loadProfile, type Profile } from "@/lib/supabase/profile";

export const metadata: Metadata = { title: "Profile · Bin Buddy" };

const at = (i: number) => ({ "--i": i }) as React.CSSProperties;

export default async function ProfilePage() {
  const profile = await loadProfile().catch(() => "unavailable" as const);

  return (
    <>
      <SiteHeader />
      <main className="stagger mx-auto w-full max-w-xl flex-1 space-y-5 px-4 py-6">
        <header className="space-y-1">
          <h1 className="font-heading text-4xl font-bold">Profile</h1>
        </header>

        {profile === "unavailable" ? (
          <p className="card p-4" style={at(1)}>
            Your profile is not available right now.
          </p>
        ) : profile === null ? (
          <div className="card flex flex-col items-center gap-3 p-8 text-center" style={at(1)}>
            <span className="grid size-14 place-items-center rounded-full bg-sticker text-ink">
              <ProfileIcon className="size-7" />
            </span>
            <p className="font-heading text-xl font-bold">No profile yet</p>
            <p className="text-muted">
              Scan your first item and Bin Buddy gives you a pseudonym. No name or email needed.
            </p>
            <Link href="/" className="btn btn-primary mt-2">
              <CameraIcon className="size-5" />
              Scan an item
            </Link>
          </div>
        ) : (
          <ProfileDetails profile={profile} />
        )}

        <p className="text-sm text-muted" style={at(5)}>
          Your pseudonym stays with this browser on this phone, through reloads and switches between
          Wi-Fi and mobile data, but not if you clear site data, browse privately or use another
          browser.
        </p>
        {profile !== "unavailable" && profile !== null && (
          <div style={at(6)}>
            <ResetIdentity />
          </div>
        )}
      </main>
      <BottomNav />
    </>
  );
}

function ProfileDetails({ profile }: { profile: Profile }) {
  const today = todayPoints(profile.pointsToday, DAILY_POINTS_CAP);
  const stats = [
    { term: "Total points", value: profile.totalPoints, one: "point", many: "points" },
    { term: "Photos scanned", value: profile.scans, one: "photo", many: "photos" },
    {
      term: "Verified disposals",
      value: profile.verifiedDisposals,
      one: "disposal",
      many: "disposals",
    },
  ];
  return (
    <>
      <div className="card flex items-center gap-4 p-5" style={at(1)}>
        <span className="grid size-14 shrink-0 place-items-center rounded-full bg-ink text-sticker">
          <ProfileIcon className="size-7" />
        </span>
        <span className="min-w-0">
          <span className="block text-sm font-bold text-muted">You appear as</span>
          <span className="block truncate font-heading text-2xl font-bold">
            {profile.pseudonym}
          </span>
        </span>
      </div>

      <dl className="grid grid-cols-3 gap-3" style={at(2)}>
        {stats.map(({ term, value, one, many }) => (
          <div key={term} className="card p-4">
            <dt className="text-xs font-bold text-muted">{term}</dt>
            <dd className="font-heading text-3xl font-bold tabular-nums">
              <span aria-hidden>{value.toLocaleString("en-SG")}</span>
              <span className="sr-only">{countText(value, one, many)}</span>
            </dd>
          </div>
        ))}
      </dl>

      <div className="card space-y-2 p-4" style={at(3)}>
        <div className="flex items-baseline justify-between gap-3 text-sm">
          <span className="font-bold">Points today</span>
          <span className="font-heading text-lg font-bold tabular-nums">
            {today.points} of {today.cap}
          </span>
        </div>
        <div
          role="progressbar"
          aria-label="Points today"
          aria-valuemin={0}
          aria-valuemax={today.cap}
          aria-valuenow={today.points}
          aria-valuetext={`${countText(today.points, "point", "points")} of ${today.cap} today`}
          className="h-3 overflow-hidden rounded-full bg-ink/10"
        >
          {today.points > 0 && (
            <div
              className="h-full origin-left rounded-full bg-sticker ring-1 ring-ink/50 animate-grow"
              style={{ width: `${today.percent}%` }}
            />
          )}
        </div>
        <p className="text-xs text-muted">
          {today.left > 0
            ? `${countText(today.left, "more point", "more points")} can count today. Points count correct disposals, up to ${today.cap} a day.`
            : `You have reached today's limit of ${today.cap} points. Thank you!`}
        </p>
      </div>

      <History items={profile.history} />
    </>
  );
}

/** Results only, never photos: verification photos are deleted after the check. */
function History({ items }: { items: HistoryItem[] }) {
  return (
    <section aria-labelledby="history-title" className="card p-4" style={at(4)}>
      <h2 id="history-title" className="font-heading text-lg font-bold">
        Your scans
      </h2>
      {items.length === 0 ? (
        <p className="mt-1 text-muted">Your finished scans appear here.</p>
      ) : (
        <ul className="mt-2 divide-y divide-paper">
          {items.map((item) => (
            <li key={item.id} className="flex items-center gap-3 py-3">
              <span className="min-w-0 flex-1">
                <span className="block font-bold">{item.material}</span>
                <span className="block text-sm text-muted">
                  {item.bin} · {item.date}
                </span>
              </span>
              {item.verified && (
                <span className="inline-flex shrink-0 items-center gap-1 rounded-full bg-bin-blue px-2.5 py-1 text-xs font-bold text-light">
                  <CheckIcon className="size-3.5" />
                  Verified
                </span>
              )}
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
