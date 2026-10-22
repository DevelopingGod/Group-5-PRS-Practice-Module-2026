import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { describe, expect, it } from "vitest";

import { countText, historyItems, todayPoints, type ScanSummary } from "@/lib/profile";

describe("countText", () => {
  it("uses the singular only for one", () => {
    expect(countText(0, "photo", "photos")).toBe("0 photos");
    expect(countText(1, "photo", "photos")).toBe("1 photo");
    expect(countText(2, "photo", "photos")).toBe("2 photos");
  });

  it("groups thousands", () => {
    expect(countText(1204, "point", "points")).toBe("1,204 points");
  });
});

describe("todayPoints", () => {
  it("shows today's points against the cap", () => {
    expect(todayPoints(3, 10)).toEqual({ points: 3, cap: 10, left: 7, percent: 30 });
  });

  it("never shows more than the cap or less than zero", () => {
    expect(todayPoints(14, 10)).toEqual({ points: 10, cap: 10, left: 0, percent: 100 });
    expect(todayPoints(-1, 10).points).toBe(0);
  });
});

describe("historyItems", () => {
  const scan: ScanSummary = {
    id: "a",
    material: "plastic",
    channel: "general_waste",
    bin_key: "general_waste",
    created_at: "2026-10-22T10:52:00Z",
  };

  it("shows the material, bin, Singapore time and verified badge", () => {
    expect(historyItems([scan], new Set(["a"]))).toEqual([
      {
        id: "a",
        material: "Plastic",
        bin: "General waste",
        date: expect.stringMatching(/22 Oct.*6:52/) as string,
        verified: true,
      },
    ]);
    expect(historyItems([scan], new Set())[0]?.verified).toBe(false);
  });

  it("says when there was no verdict", () => {
    const ood = { ...scan, material: null, channel: "abstain" as const, bin_key: null };
    expect(historyItems([ood], new Set())[0]).toMatchObject({
      material: "Unknown item",
      bin: "No verdict",
    });
  });

  it("names every HDB bin exactly as the policy does", () => {
    const yaml = readFileSync(resolve(__dirname, "../../../../policies/hdb_default.yaml"), "utf-8");
    const bins = [...yaml.matchAll(/^\s+- key: (\S+)\r?\n\s+label: (.+?)\r?$/gm)];
    expect(bins.length).toBeGreaterThan(0);
    for (const [, key, label] of bins) {
      const [item] = historyItems([{ ...scan, channel: "recycle", bin_key: key! }], new Set());
      expect(item?.bin).toBe(label);
    }
  });
});
