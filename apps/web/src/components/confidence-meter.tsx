// Signature element: one confidence meter carried from the questions to the verdict.
// It fills from the previous value to the new one, so each answer visibly moves it, and it
// marks the threshold at which Bin Buddy stops asking (CONFIDENCE_THRESHOLD).
import { CONFIDENCE_THRESHOLD } from "@/contracts/constants";
import { percent } from "@/lib/scan-flow";

export function ConfidenceMeter({
  value,
  from = 0,
  tone = "light",
  caption,
}: {
  value: number;
  from?: number;
  /** "light" on paper or white; "onColour" inside a coloured bin card. */
  tone?: "light" | "onColour";
  caption?: string;
}) {
  const now = percent(value);
  const start = percent(from);
  const threshold = percent(CONFIDENCE_THRESHOLD);
  const sure = now >= threshold;
  const track = tone === "light" ? "bg-ink/10" : "bg-black/20";
  // The yellow fill gets an ink edge so it still reads against the pale track.
  const fill =
    tone === "light" ? (sure ? "bg-bin-blue" : "bg-sticker ring-1 ring-ink/50") : "bg-current";
  const label = caption ?? (sure ? "Sure enough to decide" : "Not sure yet");

  return (
    <div className="space-y-1.5">
      <div className="flex items-baseline justify-between gap-3 text-sm">
        <span className="font-bold">{label}</span>
        <span className="font-heading text-lg font-bold tabular-nums">{now}% sure</span>
      </div>
      <div
        role="progressbar"
        aria-label="How sure Bin Buddy is"
        aria-valuenow={now}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuetext={`${now}% sure. Bin Buddy decides at ${threshold}%.`}
        className={`relative h-3 rounded-full ${track}`}
      >
        <div
          key={`${start}-${now}`}
          className={`meter-fill h-full rounded-full ${fill}`}
          style={{ width: `${now}%`, "--from": `${start}%` } as React.CSSProperties}
        />
        {/* Threshold mark: a dark-and-light outline keeps it visible over any fill colour. */}
        <span
          aria-hidden
          className="absolute -top-1 -bottom-1 w-0.5 rounded-full bg-current shadow-[0_0_0_1px_rgb(0_0_0/0.45),0_0_0_2px_rgb(255_255_255/0.7)]"
          style={{ left: `${threshold}%` }}
        />
      </div>
      <p className="text-xs">Decides at {threshold}%</p>
    </div>
  );
}
