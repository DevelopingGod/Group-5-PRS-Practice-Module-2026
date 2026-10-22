import { describe, expect, it } from "vitest";

import {
  ESTIMATE_CAP,
  UPLOAD_END,
  progressText,
  progressView,
  startProgress,
  wakingAfter,
  withUpload,
} from "@/lib/scan-progress";

const WAKING = 8_000;

describe("upload", () => {
  it("measures the upload as 0 to 30%", () => {
    const p = startProgress(0, WAKING);
    expect(progressView(p, 0)).toMatchObject({ percent: 0, stage: "uploading" });
    const half = withUpload(p, 0.5, 100);
    expect(progressView(half, 100)).toMatchObject({
      percent: UPLOAD_END / 2,
      stage: "uploading",
      label: "Uploading photo",
      estimated: false,
    });
  });

  it("never moves backwards and stamps when the upload finished", () => {
    const p = withUpload(withUpload(startProgress(0, WAKING), 0.8, 50), 0.3, 60);
    expect(p.uploadFraction).toBe(0.8);
    const done = withUpload(withUpload(p, 1, 200), 1, 900);
    expect(done.uploadDoneAt).toBe(200);
  });
});

describe("estimate after the upload", () => {
  const uploaded = withUpload(startProgress(0, WAKING), 1, 500);

  it("starts at 30%, rises, and is marked as an estimate", () => {
    const early = progressView(uploaded, 500);
    const later = progressView(uploaded, 3_000);
    expect(early).toMatchObject({ percent: UPLOAD_END, stage: "server", estimated: true });
    expect(later.percent).toBeGreaterThan(early.percent);
  });

  it("slows down and says the model is waking after the cold-start notice", () => {
    const before = progressView(uploaded, WAKING - 1);
    const at = progressView(uploaded, WAKING);
    const minute = progressView(uploaded, 60_000);
    expect(at.stage).toBe("waking");
    expect(at.label).toMatch(/up to a minute/);
    expect(at.percent - before.percent).toBeLessThanOrEqual(1); // no jump at the switch
    expect(minute.percent).toBeGreaterThan(at.percent);
  });

  it("never passes 90% until the reply arrives, then shows 100%", () => {
    for (const t of [1_000, 10_000, 60_000, 600_000]) {
      expect(progressView(uploaded, t).percent).toBeLessThanOrEqual(ESTIMATE_CAP);
    }
    const done = progressView({ ...uploaded, done: true }, 1_000);
    expect(done).toMatchObject({ percent: 100, stage: "done", estimated: false });
  });

  it("only moves forward over time", () => {
    let last = 0;
    for (let t = 500; t <= 120_000; t += 250) {
      const { percent } = progressView(uploaded, t);
      expect(percent).toBeGreaterThanOrEqual(last);
      last = percent;
    }
  });
});

describe("progressText", () => {
  it("names the percentage, whether it is estimated, and the stage", () => {
    const uploaded = withUpload(startProgress(0, WAKING), 1, 0);
    expect(progressText(progressView(uploaded, 0))).toBe(
      "Checking the photo and identifying the material. 0 seconds so far.",
    );
    expect(progressText(progressView(uploaded, 12_400))).toMatch(/12 seconds so far/);
  });
});

describe("seconds elapsed", () => {
  it("counts whole seconds from when the photo was sent", () => {
    const p = withUpload(startProgress(1_000, WAKING), 1, 1_500);
    expect(progressView(p, 1_999).seconds).toBe(0);
    expect(progressView(p, 4_200).seconds).toBe(3);
  });
});

describe("waking up the model", () => {
  it("is only said on a cold start", () => {
    const warm = withUpload(startProgress(0, null), 1, 500);
    expect(progressView(warm, 30_000).stage).toBe("server");
  });

  it("treats the model as awake for 15 minutes after a reply", () => {
    const now = 100 * 60_000;
    expect(wakingAfter(null, now)).toBe(WAKING);
    expect(wakingAfter(now - 60_000, now)).toBeNull();
    expect(wakingAfter(now - 15 * 60_000, now)).toBe(WAKING);
  });
});
