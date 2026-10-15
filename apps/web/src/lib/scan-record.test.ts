import { describe, expect, it } from "vitest";

import { EMBEDDING_DIM } from "@/contracts/constants";
import type { PredictResponse, VerdictResult } from "@/lib/contracts";
import { randomPseudonym, scanRowFromPredict, verdictUpdate } from "@/lib/scan-record";

const scanId = "00000000-0000-0000-0000-000000000001";
const analysis = {
  material_scores: [
    { label: "plastic" as const, probability: 0.6 },
    { label: "metal" as const, probability: 0.4 },
  ],
  condition_scores: [
    { label: "clean" as const, probability: 0.3 },
    { label: "item_inside" as const, probability: 0.7 },
  ],
  confidence: 0.58,
  is_ood: false,
  embedding: Array.from({ length: EMBEDDING_DIM }, () => 0.0625),
  phash: "00ff00ff00ff00ff",
};
const base: PredictResponse = {
  scan_id: scanId,
  model_revision: "v0.1-baseline",
  quality: { ok: true, blur_score: 300, brightness: 0.5 },
  status: "question",
  analysis,
};
const verdict: VerdictResult = {
  channel: "general_waste",
  bin_key: "general_waste",
  bin_label: "General waste",
  reason: "Food contaminates the blue bin.",
  rule_id: "container_item_inside",
  rule_trace: [],
  confidence: 0.91,
  questions_asked: [{ question_id: "q1_recycling_symbol", answer: "yes" }],
};

describe("scanRowFromPredict", () => {
  it("saves nothing for a retake", () => {
    expect(scanRowFromPredict("u1", "hdb_default", { ...base, status: "retake" })).toBeNull();
  });

  it("saves a pending scan with the top labels while questions are asked", () => {
    const row = scanRowFromPredict("u1", "hdb_default", base);
    expect(row).toMatchObject({
      id: scanId,
      user_id: "u1",
      site_policy: "hdb_default",
      material: "plastic",
      condition: "item_inside",
      channel: null,
      bin_key: null,
      confidence: 0.58,
      questions: [],
      model_revision: "v0.1-baseline",
      phash: "00ff00ff00ff00ff",
    });
    expect(row?.embedding).toMatch(/^\[0\.0625(,0\.0625){255}\]$/);
  });

  it("saves the verdict directly when no questions were needed", () => {
    const row = scanRowFromPredict("u1", "hdb_default", { ...base, status: "verdict", verdict });
    expect(row).toMatchObject(verdictUpdate(verdict));
  });

  it("drops a malformed embedding or hash instead of failing the insert", () => {
    const bad = { ...analysis, embedding: [1], phash: "xyz" };
    const row = scanRowFromPredict("u1", "hdb_default", { ...base, analysis: bad });
    expect(row?.embedding).toBeNull();
    expect(row?.phash).toBeNull();
  });
});

describe("randomPseudonym", () => {
  it("fits the profiles check constraint", () => {
    for (const r of [0, 0.5, 0.999]) {
      const name = randomPseudonym(() => r);
      expect(name.length).toBeGreaterThanOrEqual(3);
      expect(name.length).toBeLessThanOrEqual(24);
    }
    expect(randomPseudonym(() => 0)).toBe("Tidy Otter 10");
  });
});
