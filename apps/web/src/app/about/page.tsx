import type { Metadata } from "next";

import { BottomNav } from "@/components/bottom-nav";
import { BinMark, CameraIcon, ExternalIcon, QuestionIcon } from "@/components/icons";
import { SiteHeader } from "@/components/site-header";
import {
  CONFIDENCE_THRESHOLD,
  DAILY_POINTS_CAP,
  POINTS_PER_VERIFIED_DISPOSAL,
  VERIFY_TIME_WINDOW_MINUTES,
} from "@/contracts/constants";
import { NEA_CHECK_MY_ITEM, percent } from "@/lib/scan-flow";

export const metadata: Metadata = { title: "About · Bin Buddy" };

const at = (i: number) => ({ "--i": i }) as React.CSSProperties;

// Sources from context.md Section 22 (the same links as the HDB policy's rules).
const SOURCE_EWASTE_CLOTHING =
  "https://asianews.network/unsure-if-you-can-recycle-a-frying-pan-ask-neas-new-ai-chatbot/";
const SOURCE_STYROFOAM = "https://www.c3a.org.sg/node/26053";

const STEPS = [
  { Icon: CameraIcon, title: "Photograph one item", body: "On its own, as it is now." },
  {
    Icon: QuestionIcon,
    title: "Answer a question, only if needed",
    body: `Bin Buddy asks only when it is less than ${percent(CONFIDENCE_THRESHOLD)}% sure from the photo.`,
  },
  {
    Icon: BinMark,
    title: "Get the bin and the reason",
    body: "With the rule it applied and the NEA source behind it.",
  },
];

const BINS = [
  {
    name: "Blue recycling bin",
    body: "Clean, empty paper, plastic, glass, metal and drink cartons.",
    href: NEA_CHECK_MY_ITEM,
    source: "NEA Check My Item",
  },
  {
    name: "General waste",
    body: "Food, stained paper, and containers with food or drink left inside. Styrofoam too, even with a recycling symbol.",
    href: SOURCE_STYROFOAM,
    source: "Straits Times: what goes in the blue bin",
  },
  {
    name: "E-waste collection point",
    body: "Small electronic appliances and light bulbs, not the blue bin.",
    href: SOURCE_EWASTE_CLOTHING,
    source: "Straits Times: NEA's recycling chatbot",
  },
  {
    name: "Donate or use another channel",
    body: "Old clothing, not the blue bin.",
    href: SOURCE_EWASTE_CLOTHING,
    source: "Straits Times: NEA's recycling chatbot",
  },
];

export default function AboutPage() {
  return (
    <>
      <SiteHeader />
      <main className="stagger mx-auto w-full max-w-xl flex-1 space-y-5 px-4 py-6">
        <header className="space-y-1">
          <p className="text-sm font-bold text-muted">An explainable recycling assistant</p>
          <h1 className="font-heading text-4xl font-bold">About Bin Buddy</h1>
        </header>

        <Section title="How it works" i={1}>
          <ol className="space-y-3">
            {STEPS.map(({ Icon, title, body }, i) => (
              <li key={title} className="flex items-start gap-3">
                <span className="grid size-10 shrink-0 place-items-center rounded-xl bg-paper text-bin-blue">
                  <Icon className="size-5" />
                </span>
                <span>
                  <span className="block text-xs font-bold text-muted">Step {i + 1}</span>
                  <span className="block font-bold">{title}</span>
                  <span className="block text-sm text-muted">{body}</span>
                </span>
              </li>
            ))}
          </ol>
        </Section>

        <Section title="Bins and rules" i={2}>
          <p className="text-muted">Bin Buddy follows the HDB blue-bin rules.</p>
          <ul className="space-y-3">
            {BINS.map((bin) => (
              <li key={bin.name} className="rounded-xl bg-paper p-3">
                <p className="font-bold">{bin.name}</p>
                <p>{bin.body}</p>
                <a
                  href={bin.href}
                  target="_blank"
                  rel="noreferrer"
                  className="mt-1 inline-flex min-h-11 items-center gap-1.5 font-bold text-bin-blue underline-offset-2 hover:underline"
                >
                  {bin.source}
                  <ExternalIcon className="size-4 shrink-0" />
                </a>
              </li>
            ))}
          </ul>
        </Section>

        <Section title="Privacy" i={3}>
          <ul className="list-disc space-y-1 pl-5">
            <li>Photograph the item only. Keep people and faces out of the picture.</li>
            <li>
              Photos taken at the bin are deleted after the check. Only a fingerprint is kept, to
              stop reuse.
            </li>
            <li>You appear only as a pseudonym. No name or email is needed.</li>
            <li>No location is kept beyond the ID of the bin you scan.</li>
          </ul>
        </Section>

        <Section title="How accurate it is" i={4}>
          <p>
            On 395 Singapore photos it had never seen in training, the model named the material
            correctly about 82% of the time.
          </p>
          <p>
            Bin Buddy is advisory only. When it is unsure, it asks a question or recommends general
            waste, so it does not contaminate the recycling. For unfamiliar items it gives no
            verdict and points you to{" "}
            <a
              href={NEA_CHECK_MY_ITEM}
              target="_blank"
              rel="noreferrer"
              className="font-bold text-bin-blue underline underline-offset-2"
            >
              NEA Check My Item
            </a>
            .
          </p>
        </Section>

        <Section title="Points" i={5}>
          <p>
            You earn {POINTS_PER_VERIFIED_DISPOSAL} point for each correct disposal you verify at a
            Bin Buddy demonstration bin: scan the bin&apos;s QR code and photograph the item there
            within {VERIFY_TIME_WINDOW_MINUTES} minutes. Points reward correct disposal, not volume,
            and count up to {DAILY_POINTS_CAP} a day.
          </p>
        </Section>
      </main>
      <BottomNav />
    </>
  );
}

function Section({ title, i, children }: { title: string; i: number; children: React.ReactNode }) {
  const id = `about-${title.toLowerCase().replaceAll(" ", "-")}`;
  return (
    <section aria-labelledby={id} className="card space-y-3 p-4" style={at(i)}>
      <h2 id={id} className="font-heading text-xl font-bold">
        {title}
      </h2>
      {children}
    </section>
  );
}
