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
      /** Null after a reload restored the step without a thumbnail. */
      photoUrl: string | null;
      scanId: string;
      analysis: VisionAnalysis | null;
      question: Question;
      decision: DecisionState;
      submitting: boolean;
    }
  | {
      step: "verdict";
      photoUrl: string | null;
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

function malformed(photoUrl: string | null): ScanState {
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

type QuestionAnswer = Schemas["QuestionAnswer"];

// What the user said, as a sentence: cleanliness comes from these answers, not from the photo.
const ANSWER_SENTENCES: Record<QuestionAnswer["question_id"], Record<string, string>> = {
  q1_recycling_symbol: {
    yes: "It has a recycling symbol",
    no: "It has no recycling symbol",
    not_sure: "Not sure about a recycling symbol",
  },
  q2_cleaned: {
    yes: "It is empty and rinsed",
    no: "It is not empty and rinsed",
    not_sure: "Not sure it is empty and rinsed",
  },
  q3_sitting_time: {
    under_1_hour: "Used less than an hour ago",
    same_day: "Used earlier today",
    more_than_1_day: "Left for more than a day",
  },
};

export function answerSentence(qa: QuestionAnswer): string {
  return ANSWER_SENTENCES[qa.question_id][qa.answer] ?? answerLabel(qa.answer);
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

// Below this the material is a guess, worded as "Possibly ..." so it does not read as a fact
// next to a firm verdict (the answers explain the decision).
const POSSIBLY_BELOW = 0.5;

export function materialLabel(material: Schemas["Material"]): string {
  return MATERIAL_LABELS[material];
}

/** "Metal", or "Possibly metal" when the model is less than 50% sure. */
export function topMaterial(analysis: VisionAnalysis | null): string | null {
  const scores = analysis?.material_scores;
  const best = top(scores);
  if (!best || !scores) return null;
  const p = scores.find((s) => s.label === best)?.probability ?? 0;
  const label = MATERIAL_LABELS[best];
  return p < POSSIBLY_BELOW ? `Possibly ${label.toLowerCase()}` : label;
}

/**
 * The condition the photo showed, or null when the model did not judge it. The vision model
 * sends equal condition scores by design (cleanliness is decided by Q2 and Q3), and picking
 * the first of equal scores would wrongly show "Clean".
 */
export function topCondition(analysis: VisionAnalysis | null): string | null {
  const scores = analysis?.condition_scores;
  if (!scores || scores.length === 0) return null;
  const probabilities = scores.map((s) => s.probability);
  if (Math.max(...probabilities) - Math.min(...probabilities) < 0.01) return null;
  const best = top(scores);
  return best ? CONDITION_LABELS[best] : null;
}

function top<L extends string>(scores?: { label: L; probability: number }[]): L | null {
  if (!scores || scores.length === 0) return null;
  return scores.reduce((a, b) => (b.probability > a.probability ? b : a)).label;
}

export function percent(confidence: number): number {
  return Math.round(Math.min(1, Math.max(0, confidence)) * 100);
}

// The deciding rule in plain words, for "Rule applied". Rule ids are shared by the HDB and
// campus policies; an unknown id falls back to the last step of the model's rule trace.
const RULE_NAMES: Record<string, string> = {
  e_waste_items: "Small electronics and light bulbs",
  textiles: "Old clothing",
  styrofoam: "Styrofoam, even with a recycling symbol",
  food_waste: "Food",
  other_general: "Not accepted for recycling",
  stained_paper: "Oil- or food-stained paper",
  wet_or_filled_paper: "Wet or filled paper",
  container_item_inside: "Container with something left inside",
  container_leftover_liquid: "Container with drink left inside",
  container_food_residue: "Container with food residue",
  clean_recyclables: "Clean paper, plastic, glass, metal or carton",
  clean_paper: "Clean paper",
  clean_plastic: "Clean plastic",
  clean_cans_glass: "Clean cans and glass",
  uncertain_fallback: "Still not sure after the questions",
  abstain_ood: "Unfamiliar item",
};

/** "Container with food residue → General waste". */
export function ruleLine(verdict: VerdictResult): string {
  const name = RULE_NAMES[verdict.rule_id];
  if (name) return `${name} → ${verdict.bin_label}`;
  return verdict.rule_trace.at(-1) ?? verdict.rule_id;
}

/** "No questions needed" / "After 2 questions", shown next to the "94% sure" figure. */
export function questionsLine(verdict: VerdictResult): string {
  const n = verdict.questions_asked?.length ?? 0;
  if (n === 0) return "No questions needed";
  return `After ${n} question${n === 1 ? "" : "s"}`;
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

/** WCAG relative luminance of a #rrggbb colour. */
function luminance(hex: string): number {
  const [r, g, b] = [1, 3, 5].map((i) => {
    const v = parseInt(hex.slice(i, i + 2), 16) / 255;
    return v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

/** WCAG contrast ratio between two #rrggbb colours (1 to 21). */
export function contrast(a: string, b: string): number {
  const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x);
  return (hi + 0.05) / (lo + 0.05);
}

/** Ink or light text, whichever contrasts better with a #rrggbb background. */
export function textOn(hex: string): "#10233F" | "#F4F7FB" {
  return contrast(hex, "#10233F") >= contrast(hex, "#F4F7FB") ? "#10233F" : "#F4F7FB";
}

export const NEA_CHECK_MY_ITEM = "https://www.cgs.gov.sg/recycleright/check-my-item/";
