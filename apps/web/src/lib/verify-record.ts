// Builds the model API /verify request from the saved scan, and works out the points.
// Pure functions, so they can be unit-tested without a database (context.md Section 5.5).
import { DAILY_POINTS_CAP, POINTS_PER_VERIFIED_DISPOSAL } from "@/contracts/constants";
import type { BinQrPayload, VerifyRequest, VerifyResponse } from "@/lib/contracts";

/** What the browser sends to our /api/verify route; the rest comes from Supabase. */
export type VerifyRouteRequest = Pick<VerifyRequest, "scan_id" | "image_base64" | "qr">;

/** Our /api/verify reply: the model's checks plus the points actually awarded. */
export type VerifyRouteResponse = VerifyResponse & { points_awarded: number };

/** The saved scan, as read back from the `scans` table. */
export type SavedScan = {
  id: string;
  site_policy: string;
  bin_key: string | null;
  embedding: string | number[] | null;
  phash: string | null;
  created_at: string;
};

export type BuildResult = { ok: true; request: VerifyRequest } | { ok: false; message: string };

export function buildVerifyRequest(
  scan: SavedScan,
  body: VerifyRouteRequest,
  knownPhashes: string[],
  now: Date,
): BuildResult {
  if (!scan.bin_key) {
    return { ok: false, message: "This item has no recommended bin to verify against." };
  }
  const embedding = parseVector(scan.embedding);
  if (!embedding || !scan.phash) {
    return { ok: false, message: "This scan cannot be verified. Please scan the item again." };
  }
  return {
    ok: true,
    request: {
      scan_id: scan.id,
      image_base64: body.image_base64,
      qr: body.qr,
      recommended_site_policy: scan.site_policy,
      recommended_bin_key: scan.bin_key,
      original_embedding: embedding,
      original_phash: scan.phash,
      original_created_at: new Date(scan.created_at).toISOString(),
      submitted_at: now.toISOString(), // server time, never the browser's clock
      // The original scan photo counts as "known", so re-sending it is caught as a duplicate.
      known_phashes: [...new Set([scan.phash, ...knownPhashes])],
    },
  };
}

/** True when the scanned QR code is the registered bin with that id. */
export function binIsRegistered(qr: BinQrPayload, registered: BinQrPayload | null): boolean {
  return (
    registered !== null &&
    registered.site_policy === qr.site_policy &&
    registered.bin_key === qr.bin_key
  );
}

/** Points for a verified disposal, given the points already earned today. */
export function pointsFor(pointsToday: number): number {
  return Math.max(0, Math.min(POINTS_PER_VERIFIED_DISPOSAL, DAILY_POINTS_CAP - pointsToday));
}

const SGT_OFFSET_MS = 8 * 60 * 60 * 1000; // Singapore is UTC+8 all year
const DAY_MS = 24 * 60 * 60 * 1000;

/** Midnight at the start of the Singapore day containing `now`, as a UTC instant. */
export function startOfSingaporeDay(now: Date): Date {
  const local = now.getTime() + SGT_OFFSET_MS;
  return new Date(Math.floor(local / DAY_MS) * DAY_MS - SGT_OFFSET_MS);
}

/** pgvector comes back from PostgREST as the text "[0.1,0.2,...]". */
export function parseVector(value: string | number[] | null): number[] | null {
  if (value === null) return null;
  if (Array.isArray(value)) return value;
  try {
    const parsed: unknown = JSON.parse(value);
    return Array.isArray(parsed) && parsed.every((x) => typeof x === "number") ? parsed : null;
  } catch {
    return null;
  }
}

// ---- Display ---------------------------------------------------------------

export type CheckRow = { key: keyof VerifyResponse["checks"]; label: string; passed: boolean };

/** The four checks, worded for the result screen (context.md Section 19, screen 8). */
export function checkRows(res: VerifyResponse, windowMinutes: number): CheckRow[] {
  const match = Math.round(Math.max(0, res.similarity) * 100);
  return [
    { key: "image_match", label: `Same item (${match}% match)`, passed: res.checks.image_match },
    { key: "bin_matches_verdict", label: "Right bin", passed: res.checks.bin_matches_verdict },
    {
      key: "within_time_window",
      label: `Within ${windowMinutes} minutes of your scan`,
      passed: res.checks.within_time_window,
    },
    { key: "not_duplicate", label: "New photo", passed: res.checks.not_duplicate },
  ];
}
