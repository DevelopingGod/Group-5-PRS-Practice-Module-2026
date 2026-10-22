import { ImageResponse } from "next/og";

import { OgMark } from "@/components/brand/og-mark";

// Link preview for WhatsApp, Telegram, LinkedIn and X (X falls back to this image).
// Generated at build time, so no image file is committed.
export const alt = "Bin Buddy: which bin does this go in?";
export const size = { width: 1200, height: 630 };
export const contentType = "image/png";

const TITLE = "Bin Buddy";
const TAGLINE = "Which bin does this go in?";

/** Bricolage Grotesque (the brand heading face) for just these words, or null when offline. */
async function headingFont(): Promise<ArrayBuffer | null> {
  try {
    const css = await fetch(
      `https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@700&text=${encodeURIComponent(TITLE + TAGLINE)}`,
    ).then((res) => res.text());
    const url = /src: url\((.+?)\) format\('(?:opentype|truetype)'\)/.exec(css)?.[1];
    return url ? await fetch(url).then((res) => res.arrayBuffer()) : null;
  } catch {
    return null; // the default font still gives a readable preview
  }
}

export default async function OpengraphImage() {
  const font = await headingFont();
  return new ImageResponse(
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        justifyContent: "center",
        width: "100%",
        height: "100%",
        padding: 88,
        gap: 40,
        background: "#10233F",
        color: "#F4F7FB",
        fontFamily: font ? "Bricolage Grotesque" : undefined,
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 32 }}>
        <OgMark size={132} />
        <div style={{ fontSize: 104, fontWeight: 700, letterSpacing: -2 }}>{TITLE}</div>
      </div>
      <div style={{ fontSize: 64, fontWeight: 700, color: "#F2C230" }}>{TAGLINE}</div>
      <div
        style={{ display: "flex", height: 12, width: 240, borderRadius: 6, background: "#1F5FD1" }}
      />
    </div>,
    {
      ...size,
      fonts: font
        ? [{ name: "Bricolage Grotesque", data: font, weight: 700, style: "normal" }]
        : [],
    },
  );
}
