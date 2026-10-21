import { describe, expect, it } from "vitest";

import { countText, todayPoints } from "@/lib/profile";

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
