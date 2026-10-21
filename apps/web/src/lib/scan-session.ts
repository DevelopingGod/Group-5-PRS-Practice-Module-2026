// Saves the question and verdict steps to sessionStorage so a reload keeps them. Pure (the
// provider does the storage calls). Only the signed decision state, the question or verdict and
// the analysis without its embedding are kept, with at most a small thumbnail of the item photo.
// The verification photo is never saved.
import type { ScanState, VisionAnalysis } from "@/lib/scan-flow";

export const SCAN_SESSION_KEY = "binbuddy.scan";

type Resumable = Extract<ScanState, { step: "question" | "verdict" }>;

type SavedScan = { v: 1; previousConfidence: number; state: Resumable };

export type RestoredScan = { state: Resumable; previousConfidence: number };

/** The analysis for display only: the 256-number embedding is not needed after /predict. */
function withoutEmbedding(analysis: VisionAnalysis | null): VisionAnalysis | null {
  return analysis && { ...analysis, embedding: [] };
}

/**
 * JSON for the question or verdict step, or null for any other step. `thumbnail` (a small
 * `data:image/` URL, or null) replaces the photo's blob URL, which does not survive a reload.
 */
export function serialiseScan(
  state: ScanState,
  previousConfidence: number,
  thumbnail: string | null,
): string | null {
  const common = { photoUrl: thumbnail, analysis: null as VisionAnalysis | null };
  let saved: Resumable;
  if (state.step === "question") {
    // An answer still in flight is lost on reload; the signed state lets the user answer again.
    saved = { ...state, ...common, analysis: withoutEmbedding(state.analysis), submitting: false };
  } else if (state.step === "verdict") {
    saved = { ...state, ...common, analysis: withoutEmbedding(state.analysis) };
  } else {
    return null;
  }
  return JSON.stringify({ v: 1, previousConfidence, state: saved } satisfies SavedScan);
}

/** The saved step, or null if there is none or it is not in the expected shape. */
export function restoreScan(raw: string | null): RestoredScan | null {
  if (!raw) return null;
  let saved: unknown;
  try {
    saved = JSON.parse(raw);
  } catch {
    return null;
  }
  if (!isRecord(saved) || saved.v !== 1 || !isRecord(saved.state)) return null;
  const state = saved.state;
  const photoOk =
    state.photoUrl === null ||
    (typeof state.photoUrl === "string" && state.photoUrl.startsWith("data:image/"));
  const common =
    photoOk &&
    typeof state.scanId === "string" &&
    (state.analysis === null || isRecord(state.analysis));
  const shapeOk =
    state.step === "question"
      ? isRecord(state.question) &&
        typeof state.question.id === "string" &&
        Array.isArray(state.question.options) &&
        isRecord(state.decision) &&
        typeof state.decision.signature === "string" &&
        typeof state.decision.confidence === "number"
      : state.step === "verdict" &&
        isRecord(state.verdict) &&
        typeof state.verdict.bin_label === "string" &&
        typeof state.verdict.confidence === "number" &&
        Array.isArray(state.verdict.rule_trace);
  if (!common || !shapeOk) return null;
  const previous = typeof saved.previousConfidence === "number" ? saved.previousConfidence : 0;
  return { state: state as Resumable, previousConfidence: previous };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
