import { describe, expect, it } from "vitest";

import {
  ESTIMATE_CAP,
  UPLOAD_END,
  progressText,
  progressView,
  startProgress,
  withUpload,
} from "@/lib/scan-progress";

const WAKING = 8_000;

describe("upload", () => {
  it("measures the upload as 0 to 30%", () => {
    const p = startProgress(0);
    expect(progressView(p, 0, WAKING)).toMatchObject({ percent: 0, stage: "uploading" });
    const half = withUpload(p, 0.5, 100);
    expect(progressView(half, 100, WAKING)).toMatchObject({
      percent: UPLOAD_END / 2,
      stage: "uploading",
      label: "Uploading photo",
      estimated: false,
    });
  });

  it("never moves backwards and stamps when the upload finished", () => {
    const p = withUpload(withUpload(startProgress(0), 0.8, 50), 0.3, 60);
    expect(p.uploadFraction).toBe(0.8);
    const done = withUpload(withUpload(p, 1, 200), 1, 900);
    expect(done.uploadDoneAt).toBe(200);
  });
});

describe("estimate after the upload", () => {
  const uploaded = withUpload(startProgress(0), 1, 500);

  it("starts at 30%, rises, and is marked as an estimate", () => {
    const early = progressView(uploaded, 500, WAKING);
    const later = progressView(uploaded, 3_000, WAKING);
    expect(early).toMatchObject({ percent: UPLOAD_END, stage: "server", estimated: true });
    expect(later.percent).toBeGreaterThan(early.percent);
  });

  it("slows down and says the model is waking after the cold-start notice", () => {
    const before = progressView(uploaded, WAKING - 1, WAKING);
    const at = progressView(uploaded, WAKING, WAKING);
    const minute = progressView(uploaded, 60_000, WAKING);
    expect(at.stage).toBe("waking");
    expect(at.label).toMatch(/up to a minute/);
    expect(at.percent - before.percent).toBeLessThanOrEqual(1); // no jump at the switch
    expect(minute.percent).toBeGreaterThan(at.percent);
  });

  it("never passes 90% until the reply arrives, then shows 100%", () => {
    for (const t of [1_000, 10_000, 60_000, 600_000]) {
      expect(progressView(uploaded, t, WAKING).percent).toBeLessThanOrEqual(ESTIMATE_CAP);
    }
    const done = progressView({ ...uploaded, done: true }, 1_000, WAKING);
    expect(done).toMatchObject({ percent: 100, stage: "done", estimated: false });
  });

  it("only moves forward over time", () => {
    let last = 0;
    for (let t = 500; t <= 120_000; t += 250) {
      const { percent } = progressView(uploaded, t, WAKING);
      expect(percent).toBeGreaterThanOrEqual(last);
      last = percent;
    }
  });
});

describe("progressText", () => {
  it("names the percentage, whether it is estimated, and the stage", () => {
    const uploaded = withUpload(startProgress(0), 1, 0);
    expect(progressText(progressView(uploaded, 0, WAKING))).toBe(
      "30%, estimated. Checking it is sharp and well lit, then identifying the material.",
    );
  });
});
