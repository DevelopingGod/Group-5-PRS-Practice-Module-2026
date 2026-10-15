// Maps model API responses to rows in the Supabase `scans` table (supabase/migrations).
// Pure functions, so they can be unit-tested without a database.
import { EMBEDDING_DIM } from "@/contracts/constants";
import type { Channel, Condition, Material, PredictResponse, VerdictResult } from "@/lib/contracts";
import type { components } from "@/contracts/api";

type QuestionAnswer = components["schemas"]["QuestionAnswer"];

export type ScanRow = {
  id: string;
  user_id: string;
  site_policy: string;
  material: Material | null;
  condition: Condition | null;
  channel: Channel | null;
  bin_key: string | null;
  confidence: number;
  questions: QuestionAnswer[];
  model_revision: string;
  embedding: string | null;
  phash: string | null;
};

export type VerdictUpdate = Pick<ScanRow, "channel" | "bin_key" | "confidence" | "questions">;

/** The row to insert after /predict, or null when there is nothing to save (retake). */
export function scanRowFromPredict(
  userId: string,
  sitePolicy: string,
  res: PredictResponse,
): ScanRow | null {
  if (res.status === "retake" || !res.analysis) return null;
  const { analysis, verdict } = res;
  return {
    id: res.scan_id,
    user_id: userId,
    site_policy: sitePolicy,
    material: top(analysis.material_scores),
    condition: top(analysis.condition_scores),
    model_revision: res.model_revision,
    embedding: analysis.embedding.length === EMBEDDING_DIM ? toVector(analysis.embedding) : null,
    phash: /^[0-9a-f]{16}$/.test(analysis.phash) ? analysis.phash : null,
    ...(verdict
      ? verdictUpdate(verdict)
      : { channel: null, bin_key: null, confidence: analysis.confidence, questions: [] }),
  };
}

export function verdictUpdate(verdict: VerdictResult): VerdictUpdate {
  return {
    channel: verdict.channel,
    bin_key: verdict.bin_key,
    confidence: verdict.confidence,
    questions: verdict.questions_asked ?? [],
  };
}

function top<L extends string>(scores: { label: L; probability: number }[]): L | null {
  if (scores.length === 0) return null;
  return scores.reduce((a, b) => (b.probability > a.probability ? b : a)).label;
}

/** pgvector text format: "[0.1,0.2,...]". */
function toVector(values: number[]): string {
  return `[${values.join(",")}]`;
}

// ---- Pseudonyms ------------------------------------------------------------

const ADJECTIVES = [
  "Tidy",
  "Clean",
  "Bright",
  "Quick",
  "Sunny",
  "Keen",
  "Calm",
  "Bold",
  "Green",
  "Swift",
];
const ANIMALS = [
  "Otter",
  "Hornbill",
  "Kingfisher",
  "Pangolin",
  "Heron",
  "Gecko",
  "Sunbird",
  "Egret",
];

/** e.g. "Tidy Otter 42". Never derived from the user, so it cannot identify them. */
export function randomPseudonym(random: () => number = Math.random): string {
  const pick = <T>(xs: T[]): T => xs[Math.floor(random() * xs.length)] as T;
  const n = 10 + Math.floor(random() * 90);
  return `${pick(ADJECTIVES)} ${pick(ANIMALS)} ${n}`;
}
