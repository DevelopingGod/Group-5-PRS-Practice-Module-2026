// Pure state machine for the scan -> question -> verdict flow (context.md Section 5.1 and 19).
// Kept free of React and fetch so it can be unit-tested.
import type {
  AnswerResponse,
  DecisionState,
  ErrorResponse,
  PredictResponse,
  Question,
  VerdictResult,
} from "@/lib/contracts";
import type { components } from "@/contracts/api";

type Schemas = components["schemas"];
export type QualityIssue = Schemas["QualityIssue"];
export type VisionAnalysis = Schemas["VisionAnalysis"];

export type ScanState =
  | { step: "capture" }
  | { step: "reading"; photoUrl: string }
  | { step: "retake"; photoUrl: string; issues: QualityIssue[] }
  | {
      step: "question";
      photoUrl: string;
      scanId: string;
      analysis: VisionAnalysis | null;
      question: Question;
      decision: DecisionState;
      submitting: boolean;
    }
  | {
      step: "verdict";
      photoUrl: string;
      scanId: string;
      analysis: VisionAnalysis | null;
      verdict: VerdictResult;
    }
  | { step: "error"; photoUrl: string | null; error: ErrorResponse };

/** Applies a /predict response. A malformed response becomes an `internal` error. */
export function fromPredict(photoUrl: string, res: PredictResponse): ScanState {
  if (res.status === "retake") {
    return { step: "retake", photoUrl, issues: res.quality.issues ?? [] };
  }
  const base = { photoUrl, scanId: res.scan_id, analysis: res.analysis ?? null };
  if (res.status === "question" && res.next_question && res.state) {
    return {
      ...base,
      step: "question",
      question: res.next_question,
      decision: res.state,
      submitting: false,
    };
  }
  if (res.status === "verdict" && res.verdict) {
    return { ...base, step: "verdict", verdict: res.verdict };
  }
  return malformed(photoUrl);
}

/** Applies an /answer response to the current question state. */
export function fromAnswer(
  current: Extract<ScanState, { step: "question" }>,
  res: AnswerResponse,
): ScanState {
  const base = { photoUrl: current.photoUrl, scanId: current.scanId, analysis: current.analysis };
  if (res.status === "question" && res.next_question && res.state) {
    return {
      ...base,
      step: "question",
      question: res.next_question,
      decision: res.state,
      submitting: false,
    };
  }
  if (res.status === "verdict" && res.verdict) {
    return { ...base, step: "verdict", verdict: res.verdict };
  }
  return malformed(current.photoUrl);
}

function malformed(photoUrl: string): ScanState {
  return {
    step: "error",
    photoUrl,
    error: { code: "internal", message: "The model sent an unexpected reply." },
  };
}

// ---- Display helpers -------------------------------------------------------

const ANSWER_LABELS: Record<string, string> = {
  yes: "Yes",
  no: "No",
  not_sure: "Not sure",
  under_1_hour: "Under 1 hour",
  same_day: "Same day",
  more_than_1_day: "More than 1 day",
};

export function answerLabel(answer: string): string {
  return ANSWER_LABELS[answer] ?? answer.replaceAll("_", " ");
}

const MATERIAL_LABELS: Record<Schemas["Material"], string> = {
  paper: "Paper or cardboard",
  plastic: "Plastic",
  glass: "Glass",
  metal: "Metal",
  beverage_carton: "Drink carton",
  styrofoam: "Styrofoam",
  e_waste: "Electronic item",
  textile: "Clothing or textile",
  food_waste: "Food waste",
  other_general: "Other general waste",
};

const CONDITION_LABELS: Record<Schemas["Condition"], string> = {
  clean: "Clean",
  food_residue: "Food residue",
  leftover_liquid: "Drink left inside",
  item_inside: "Something left inside",
};

export function topMaterial(analysis: VisionAnalysis | null): string | null {
  const best = top(analysis?.material_scores);
  return best ? MATERIAL_LABELS[best] : null;
}

export function topCondition(analysis: VisionAnalysis | null): string | null {
  const best = top(analysis?.condition_scores);
  return best ? CONDITION_LABELS[best] : null;
}

function top<L extends string>(scores?: { label: L; probability: number }[]): L | null {
  if (!scores || scores.length === 0) return null;
  return scores.reduce((a, b) => (b.probability > a.probability ? b : a)).label;
}

export function percent(confidence: number): number {
  return Math.round(Math.min(1, Math.max(0, confidence)) * 100);
}

/** "94% sure, no questions needed" / "91% sure after 2 questions". */
export function confidenceLine(verdict: VerdictResult): string {
  const n = verdict.questions_asked?.length ?? 0;
  const sure = `${percent(verdict.confidence)}% sure`;
  if (n === 0) return `${sure}, no questions needed`;
  return `${sure} after ${n} question${n === 1 ? "" : "s"}`;
}

const ISSUE_TEXT: Record<QualityIssue, string> = {
  too_blurry: "The photo is blurred. Hold the phone steady and tap to focus.",
  too_dark: "The photo is too dark. Move somewhere brighter.",
  too_bright: "The photo is too bright. Avoid direct light or glare on the item.",
};

export function issueText(issue: QualityIssue): string {
  return ISSUE_TEXT[issue];
}

const CHANNEL_COLOURS: Record<VerdictResult["channel"], string> = {
  recycle: "#1F5FD1",
  general_waste: "#4F5B66",
  e_waste: "#10233F",
  donation: "#10233F",
  abstain: "#10233F",
};

/** Bin card colour: the policy's colour if valid, else the brand colour for the channel. */
export function binColour(verdict: VerdictResult): string {
  const c = verdict.bin_colour;
  return c && /^#[0-9a-f]{6}$/i.test(c) ? c : CHANNEL_COLOURS[verdict.channel];
}

/** Ink or light text, whichever contrasts better with a #rrggbb background. */
export function textOn(hex: string): "#10233F" | "#F4F7FB" {
  const [r, g, b] = [1, 3, 5].map((i) => {
    const v = parseInt(hex.slice(i, i + 2), 16) / 255;
    return v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4;
  });
  const lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;
  return lum > 0.4 ? "#10233F" : "#F4F7FB";
}

export const NEA_CHECK_MY_ITEM = "https://www.cgs.gov.sg/recycleright/check-my-item/";
