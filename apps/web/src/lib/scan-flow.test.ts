import { describe, expect, it } from "vitest";

import type { DecisionState, PredictResponse, VerdictResult } from "@/lib/contracts";
import {
  binColour,
  contrast,
  fromAnswer,
  fromPredict,
  questionsLine,
  textOn,
  topCondition,
  topMaterial,
  type ScanState,
} from "@/lib/scan-flow";

const quality = { ok: true, blur_score: 300, brightness: 0.5 };
const decision: DecisionState = {
  scan_id: "00000000-0000-0000-0000-000000000001",
  site_policy: "hdb_default",
  material_scores: [],
  condition_scores: [],
  confidence: 0.58,
  signature: "abc",
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
const base = { scan_id: decision.scan_id, model_revision: "stub", quality };

describe("fromPredict", () => {
  it("asks for a retake with the quality issues", () => {
    const res: PredictResponse = {
      ...base,
      status: "retake",
      quality: { ...quality, ok: false, issues: ["too_dark"] },
    };
    expect(fromPredict("blob:x", res)).toEqual({
      step: "retake",
      photoUrl: "blob:x",
      issues: ["too_dark"],
    });
  });

  it("moves to a question with the signed state", () => {
    const question = { id: "q1_recycling_symbol" as const, text: "Q1?", options: ["yes", "no"] };
    const s = fromPredict("blob:x", {
      ...base,
      status: "question",
      next_question: question,
      state: decision,
    });
    expect(s.step).toBe("question");
    if (s.step === "question") {
      expect(s.question).toEqual(question);
      expect(s.decision).toBe(decision);
      expect(s.submitting).toBe(false);
    }
  });

  it("goes straight to a verdict", () => {
    const s = fromPredict("blob:x", { ...base, status: "verdict", verdict });
    expect(s).toMatchObject({ step: "verdict", verdict, scanId: decision.scan_id });
  });

  it("treats a question without state as an error", () => {
    const s = fromPredict("blob:x", { ...base, status: "question" });
    expect(s).toMatchObject({ step: "error", error: { code: "internal" } });
  });
});

describe("fromAnswer", () => {
  const current: Extract<ScanState, { step: "question" }> = {
    step: "question",
    photoUrl: "blob:x",
    scanId: decision.scan_id,
    analysis: null,
    question: { id: "q1_recycling_symbol", text: "Q1?", options: ["yes"] },
    decision,
    submitting: true,
  };

  it("keeps the photo and scan id when the verdict arrives", () => {
    expect(fromAnswer(current, { status: "verdict", verdict })).toEqual({
      step: "verdict",
      photoUrl: "blob:x",
      scanId: decision.scan_id,
      analysis: null,
      verdict,
    });
  });
});

describe("display helpers", () => {
  it("states the question count", () => {
    expect(questionsLine(verdict)).toBe("No questions needed");
    const one = [{ question_id: "q1_recycling_symbol" as const, answer: "yes" }];
    expect(questionsLine({ ...verdict, questions_asked: one })).toBe("After 1 question");
    const two = [...one, { question_id: "q2_cleaned" as const, answer: "no" }];
    expect(questionsLine({ ...verdict, questions_asked: two })).toBe("After 2 questions");
  });

  it("picks the most likely material", () => {
    const analysis = {
      material_scores: [
        { label: "plastic" as const, probability: 0.3 },
        { label: "metal" as const, probability: 0.7 },
      ],
      condition_scores: [],
      confidence: 0.7,
      is_ood: false,
      embedding: [],
      phash: "0",
    };
    expect(topMaterial(analysis)).toBe("Metal");
    expect(topMaterial(null)).toBeNull();
  });

  it("shows a condition only when the model judged one", () => {
    const base = {
      material_scores: [],
      confidence: 0.7,
      is_ood: false,
      embedding: [],
      phash: "0",
    };
    const scores = (clean: number, residue: number, liquid: number, inside: number) => [
      { label: "clean" as const, probability: clean },
      { label: "food_residue" as const, probability: residue },
      { label: "leftover_liquid" as const, probability: liquid },
      { label: "item_inside" as const, probability: inside },
    ];
    // The vision model sends equal scores: cleanliness is left to the questions, not "Clean".
    expect(topCondition({ ...base, condition_scores: scores(0.25, 0.25, 0.25, 0.25) })).toBeNull();
    expect(topCondition({ ...base, condition_scores: scores(0.1, 0.7, 0.1, 0.1) })).toBe(
      "Food residue",
    );
    expect(topCondition({ ...base, condition_scores: [] })).toBeNull();
    expect(topCondition(null)).toBeNull();
  });

  it("falls back to the channel colour and keeps text readable", () => {
    expect(binColour({ ...verdict, bin_colour: "#1F5FAD" })).toBe("#1F5FAD");
    expect(binColour({ ...verdict, channel: "general_waste", bin_colour: null })).toBe("#4F5B66");
    expect(textOn("#1F5FD1")).toBe("#F4F7FB");
    expect(textOn("#F2C230")).toBe("#10233F");
    // Mid-tone orange (campus plastic bin): ink is far more readable than light text.
    expect(textOn("#F5821F")).toBe("#10233F");
  });

  it("measures contrast like WCAG", () => {
    expect(contrast("#000000", "#FFFFFF")).toBeCloseTo(21, 5);
    expect(contrast("#10233F", "#EEF3F1")).toBeGreaterThan(4.5);
  });
});
