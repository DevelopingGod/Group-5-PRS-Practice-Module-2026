"use client";

// Shared building blocks for the app's screens (design system in globals.css).
import { useEffect, useRef, type ReactNode } from "react";

type ButtonProps = React.ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "primary" | "secondary";
};

export function Button({ variant = "primary", className = "", ...props }: ButtonProps) {
  return (
    <button
      type="button"
      {...props}
      className={`btn ${variant === "primary" ? "btn-primary" : "btn-secondary"} ${className}`}
    />
  );
}

/**
 * A hidden file input that hands back the chosen photo. With `camera`, phones open the back
 * camera straight away; without it they offer the photo library (and the camera).
 */
export function PhotoInput({
  ref,
  onPhoto,
  camera = false,
}: {
  ref: React.Ref<HTMLInputElement>;
  onPhoto: (file: File) => void;
  camera?: boolean;
}) {
  return (
    <input
      ref={ref}
      type="file"
      accept="image/*"
      capture={camera ? "environment" : undefined}
      className="sr-only"
      tabIndex={-1}
      aria-hidden
      onChange={(e) => {
        const file = e.target.files?.[0];
        e.target.value = "";
        if (file) onPhoto(file);
      }}
    />
  );
}

/**
 * One step of the flow. It animates in and, when it replaces another screen, moves keyboard
 * and screen-reader focus to its heading so the change is announced (the old button is gone).
 */
export function Screen({
  title,
  titleId,
  eyebrow,
  focusOnMount = true,
  live = false,
  className = "",
  titleClassName = "text-2xl",
  children,
}: {
  title: ReactNode;
  titleId: string;
  eyebrow?: ReactNode;
  focusOnMount?: boolean;
  live?: boolean;
  className?: string;
  titleClassName?: string;
  children?: ReactNode;
}) {
  const heading = useRef<HTMLHeadingElement>(null);
  useEffect(() => {
    if (focusOnMount) heading.current?.focus({ preventScroll: false });
  }, [focusOnMount]);
  return (
    <section
      aria-labelledby={titleId}
      aria-live={live ? "polite" : undefined}
      className={`stagger space-y-5 ${className}`}
    >
      <header className="space-y-1" style={{ "--i": 0 } as React.CSSProperties}>
        {eyebrow && <p className="text-sm font-bold text-muted">{eyebrow}</p>}
        <h1
          id={titleId}
          ref={heading}
          tabIndex={-1}
          data-screen-heading
          className={`font-heading font-bold leading-tight text-balance ${titleClassName}`}
        >
          {title}
        </h1>
      </header>
      {children}
    </section>
  );
}

/** A glowing line sweeping up and down a viewfinder (reading a photo, finding a QR code). */
export function ScanLine({ inset = "inset-x-6" }: { inset?: string }) {
  return (
    <span
      aria-hidden
      className={`absolute top-0 h-full will-change-transform animate-scanline ${inset}`}
    >
      <span className="block h-0.5 rounded-full bg-sticker shadow-[0_0_16px_4px_rgb(242_194_48/0.6)]" />
    </span>
  );
}

/** Inline style helper for the stagger index. */
export const at = (i: number) => ({ "--i": i }) as React.CSSProperties;

export function Notice({
  tone = "sticker",
  role,
  children,
}: {
  tone?: "sticker" | "plain";
  role?: "alert" | "status";
  children: ReactNode;
}) {
  const styles = tone === "sticker" ? "bg-sticker text-ink" : "card text-ink";
  return (
    <p role={role} className={`rounded-2xl p-4 ${styles}`}>
      {children}
    </p>
  );
}
