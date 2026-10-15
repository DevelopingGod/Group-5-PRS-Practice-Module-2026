import { describe, expect, it } from "vitest";

import { DAILY_POINTS_CAP } from "@/contracts/constants";
import { DEMO_BINS } from "@/lib/demo-bins";
import {
  binIsRegistered,
  buildVerifyRequest,
  checkRows,
  parseVector,
  pointsFor,
  startOfSingaporeDay,
  type SavedScan,
} from "@/lib/verify-record";

const blue = DEMO_BINS[0]!;
const scan: SavedScan = {
  id: "00000000-0000-0000-0000-000000000001",
  site_policy: "hdb_default",
  bin_key: "blue_bin",
  embedding: "[0.6,0.8]",
  phash: "00ff00ff00ff00ff",
  created_at: "2026-10-15T10:00:00+00:00",
};
const body = { scan_id: scan.id, image_base64: "abc", qr: blue };
const now = new Date("2026-10-15T10:05:00Z");

describe("buildVerifyRequest", () => {
  it("fills the model request from the saved scan and uses server time", () => {
    const res = buildVerifyRequest(scan, body, ["1111111111111111"], now);
    expect(res.ok).toBe(true);
    if (!res.ok) return;
    expect(res.request).toMatchObject({
      scan_id: scan.id,
      recommended_site_policy: "hdb_default",
      recommended_bin_key: "blue_bin",
      original_embedding: [0.6, 0.8],
      original_phash: scan.phash,
      original_created_at: "2026-10-15T10:00:00.000Z",
      submitted_at: "2026-10-15T10:05:00.000Z",
    });
  });

  it("treats the original scan photo as a known hash, so it cannot be re-sent", () => {
    const res = buildVerifyRequest(scan, body, [scan.phash!, "1111111111111111"], now);
    expect(res.ok && res.request.known_phashes).toEqual([scan.phash, "1111111111111111"]);
  });

  it("refuses scans without a bin or without an embedding", () => {
    expect(buildVerifyRequest({ ...scan, bin_key: null }, body, [], now).ok).toBe(false);
    expect(buildVerifyRequest({ ...scan, embedding: null }, body, [], now).ok).toBe(false);
    expect(buildVerifyRequest({ ...scan, phash: null }, body, [], now).ok).toBe(false);
  });
});

describe("binIsRegistered", () => {
  it("requires the QR code to match the registered bin", () => {
    expect(binIsRegistered(blue, blue)).toBe(true);
    expect(binIsRegistered({ ...blue, bin_key: "general_waste" }, blue)).toBe(false);
    expect(binIsRegistered(blue, null)).toBe(false);
  });
});

describe("points", () => {
  it("awards a point until the daily cap is reached", () => {
    expect(pointsFor(0)).toBe(1);
    expect(pointsFor(DAILY_POINTS_CAP - 1)).toBe(1);
    expect(pointsFor(DAILY_POINTS_CAP)).toBe(0);
    expect(pointsFor(DAILY_POINTS_CAP + 3)).toBe(0);
  });

  it("starts the day at midnight Singapore time", () => {
    // 15 Oct 01:30 SGT = 14 Oct 17:30 UTC; the day began at 14 Oct 16:00 UTC.
    expect(startOfSingaporeDay(new Date("2026-10-14T17:30:00Z")).toISOString()).toBe(
      "2026-10-14T16:00:00.000Z",
    );
    // 15 Oct 23:59 SGT is still the same day.
    expect(startOfSingaporeDay(new Date("2026-10-15T15:59:00Z")).toISOString()).toBe(
      "2026-10-14T16:00:00.000Z",
    );
  });
});

describe("parseVector", () => {
  it("reads pgvector text and rejects junk", () => {
    expect(parseVector("[0.1,-0.2]")).toEqual([0.1, -0.2]);
    expect(parseVector([0.5])).toEqual([0.5]);
    expect(parseVector("not a vector")).toBeNull();
    expect(parseVector('["a"]')).toBeNull();
    expect(parseVector(null)).toBeNull();
  });
});

describe("checkRows", () => {
  it("words the four checks", () => {
    const rows = checkRows(
      {
        passed: false,
        similarity: 0.923,
        reason: "failed: not_duplicate",
        phash: "00ff00ff00ff00ff",
        checks: {
          image_match: true,
          bin_matches_verdict: true,
          within_time_window: true,
          not_duplicate: false,
        },
      },
      30,
    );
    expect(rows.map((r) => [r.label, r.passed])).toEqual([
      ["Same item (92% match)", true],
      ["Right bin", true],
      ["Within 30 minutes of your scan", true],
      ["New photo", false],
    ]);
  });
});
