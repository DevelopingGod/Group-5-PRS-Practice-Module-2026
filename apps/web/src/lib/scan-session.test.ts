import { describe, expect, it } from "vitest";

import type { DecisionState, Question, VerdictResult } from "@/lib/contracts";
import type { ScanState, VisionAnalysis } from "@/lib/scan-flow";
import { restoreScan, serialiseScan } from "@/lib/scan-session";

const decision: DecisionState = {
  scan_id: "00000000-0000-0000-0000-000000000001",
  site_policy: "hdb_default",
  material_scores: [{ label: "plastic", probability: 1 }],
  condition_scores: [{ label: "clean", probability: 1 }],
  confidence: 0.58,
  signature: "abc",
};
const question: Question = {
  id: "q1_recycling_symbol",
  text: "Q1?",
  options: ["yes", "no", "not_sure"],
};
const verdict: VerdictResult = {
  channel: "recycle",
  bin_key: "blue_bin",
  bin_label: "Blue recycling bin",
  reason: "Clean metal can.",
  rule_id: "clean_metal",
  rule_trace: ["metal", "clean"],
  confidence: 0.94,
};
const analysis = {
  material_scores: decision.material_scores,
  condition_scores: decision.condition_scores,
  embedding: [0.6, 0.8],
  heatmap_png_base64: "iVBOR",
} as unknown as VisionAnalysis;
const thumb = "data:image/jpeg;base64,/9j/4AAQ";

const asking: ScanState = {
  step: "question",
  photoUrl: "blob:http://localhost/1",
  scanId: decision.scan_id,
  analysis,
  question,
  decision,
  submitting: true,
};
const decided: ScanState = {
  step: "verdict",
  photoUrl: "blob:http://localhost/1",
  scanId: decision.scan_id,
  analysis,
  verdict,
};

describe("serialiseScan", () => {
  it("saves nothing for steps a reload cannot resume", () => {
    expect(serialiseScan({ step: "capture" }, 0, null)).toBeNull();
    expect(serialiseScan({ step: "reading", photoUrl: "blob:x" }, 0, null)).toBeNull();
    expect(serialiseScan({ step: "retake", photoUrl: "blob:x", issues: [] }, 0, null)).toBeNull();
    const error: ScanState = {
      step: "error",
      photoUrl: null,
      error: { code: "internal", message: "x" },
    };
    expect(serialiseScan(error, 0, null)).toBeNull();
  });

  it("drops the embedding and the blob URL, keeps the heatmap and signed state", () => {
    const raw = serialiseScan(asking, 0.4, thumb)!;
    expect(raw).not.toContain("blob:");
    const saved = JSON.parse(raw) as { state: Record<string, unknown> };
    expect(saved.state.photoUrl).toBe(thumb);
    expect((saved.state.analysis as VisionAnalysis).embedding).toEqual([]);
    expect((saved.state.analysis as VisionAnalysis).heatmap_png_base64).toBe("iVBOR");
    expect(saved.state.decision).toEqual(decision);
  });
});

describe("restoreScan", () => {
  it("round-trips a question, ready to be answered again", () => {
    const restored = restoreScan(serialiseScan(asking, 0.4, thumb));
    expect(restored?.previousConfidence).toBe(0.4);
    expect(restored?.state).toMatchObject({
      step: "question",
      photoUrl: thumb,
      question,
      decision,
      submitting: false,
    });
  });

  it("round-trips a verdict without a photo", () => {
    const restored = restoreScan(serialiseScan(decided, 0.7, null));
    expect(restored?.state).toMatchObject({ step: "verdict", photoUrl: null, verdict });
  });

  it("rejects missing, broken or unexpected data", () => {
    expect(restoreScan(null)).toBeNull();
    expect(restoreScan("{not json")).toBeNull();
    expect(restoreScan(JSON.stringify({ v: 2, state: {} }))).toBeNull();
    const raw = JSON.parse(serialiseScan(decided, 0, thumb)!) as { state: { photoUrl: string } };
    raw.state.photoUrl = "https://example.com/x.jpg";
    expect(restoreScan(JSON.stringify(raw))).toBeNull();
    const noSignature = JSON.parse(serialiseScan(asking, 0, null)!) as {
      state: { decision: Record<string, unknown> };
    };
    delete noSignature.state.decision.signature;
    expect(restoreScan(JSON.stringify(noSignature))).toBeNull();
  });
});
