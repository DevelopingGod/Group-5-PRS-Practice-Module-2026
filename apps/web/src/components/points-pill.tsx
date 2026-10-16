"use client";

// The user's points in the header. It pops when the number changes, and is the landing spot
// for the "+1" sticker after a verified disposal (flyReward).
import { useState } from "react";

export const POINTS_PILL_ID = "points-pill";

export function PointsPill({ points }: { points: number }) {
  const [initial] = useState(points);
  return (
    <span
      id={POINTS_PILL_ID}
      key={points}
      className={`inline-flex items-center gap-1.5 rounded-full bg-sticker px-3 py-1 text-sm font-bold text-ink shadow-[0_2px_0_rgb(16_35_63/0.35)] ${
        points !== initial ? "animate-pop" : ""
      }`}
    >
      <span className="font-heading tabular-nums">{points}</span>
      {points === 1 ? "point" : "points"}
    </span>
  );
}

/**
 * Flies a "+N" sticker from `from` to the header points pill. Purely decorative: the real
 * result is announced in the page text. Skipped when the user prefers reduced motion.
 */
export function flyReward(from: HTMLElement, points: number): void {
  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
  const start = from.getBoundingClientRect();
  const target = document.getElementById(POINTS_PILL_ID)?.getBoundingClientRect();
  const sticker = document.createElement("span");
  sticker.textContent = `+${points}`;
  sticker.setAttribute("aria-hidden", "true");
  sticker.className =
    "pointer-events-none fixed z-50 rounded-full bg-sticker px-4 py-2 font-heading text-2xl " +
    "font-bold text-ink shadow-lg";
  sticker.style.left = `${start.left + start.width / 2 - 32}px`;
  sticker.style.top = `${start.top + start.height / 2 - 24}px`;
  document.body.appendChild(sticker);
  const dx = target ? target.left + target.width / 2 - (start.left + start.width / 2) : 0;
  const dy = target ? target.top + target.height / 2 - (start.top + start.height / 2) : -160;
  sticker
    .animate(
      [
        { transform: "translate(0, 0) scale(0.6) rotate(-8deg)", opacity: 0 },
        { transform: "translate(0, -24px) scale(1.15) rotate(4deg)", opacity: 1, offset: 0.25 },
        { transform: `translate(${dx}px, ${dy}px) scale(0.45) rotate(0)`, opacity: 0.9 },
      ],
      { duration: 1100, easing: "cubic-bezier(0.22, 1, 0.36, 1)", fill: "forwards" },
    )
    .finished.finally(() => sticker.remove());
}
