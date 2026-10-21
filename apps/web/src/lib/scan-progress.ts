// Progress for one /predict call (the "Reading your photo" screen). Pure, so it can be tested.
// Only the upload can be measured. The server runs the quality check and the vision model inside
// one request and cannot report its own progress, so after the upload the bar shows an eased
// estimate that never passes ESTIMATE_CAP until the reply arrives; then it jumps to 100%.

export const UPLOAD_END = 30;
export const ESTIMATE_CAP = 90;
// Before the cold-start notice the estimate eases towards ESTIMATE_SOFT_CAP; once the model is
// waking up it continues slowly towards ESTIMATE_CAP over about a minute.
const ESTIMATE_SOFT_CAP = 70;
const NORMAL_TIME_CONSTANT_MS = 2_500;
const WAKING_TIME_CONSTANT_MS = 20_000;

export type ReadingProgress = {
  startedAt: number;
  /** 0 to 1, from the browser's upload progress events. */
  uploadFraction: number;
  uploadDoneAt: number | null;
  /** The reply has arrived. */
  done: boolean;
};

export type ProgressStage = "uploading" | "server" | "waking" | "done";

export type ProgressView = {
  percent: number;
  stage: ProgressStage;
  label: string;
  /** True once the bar is an estimate rather than a measurement. */
  estimated: boolean;
};

export function startProgress(now: number): ReadingProgress {
  return { startedAt: now, uploadFraction: 0, uploadDoneAt: null, done: false };
}

/** Records an upload progress event; a full upload stamps the time the server's work began. */
export function withUpload(p: ReadingProgress, fraction: number, now: number): ReadingProgress {
  const f = Math.min(1, Math.max(p.uploadFraction, fraction));
  return { ...p, uploadFraction: f, uploadDoneAt: f >= 1 ? (p.uploadDoneAt ?? now) : null };
}

const ease = (ms: number, timeConstant: number) => 1 - Math.exp(-Math.max(0, ms) / timeConstant);

/** The estimate for the server's part, given how long ago the upload finished. */
function estimate(sinceUpload: number, wakingAt: number | null): number {
  const normal = (ms: number) =>
    UPLOAD_END + (ESTIMATE_SOFT_CAP - UPLOAD_END) * ease(ms, NORMAL_TIME_CONSTANT_MS);
  if (wakingAt === null || sinceUpload < wakingAt) return normal(sinceUpload);
  // Continue from wherever the estimate was when the model started waking up, more slowly.
  const from = normal(wakingAt);
  return from + (ESTIMATE_CAP - from) * ease(sinceUpload - wakingAt, WAKING_TIME_CONSTANT_MS);
}

const LABELS: Record<ProgressStage, string> = {
  uploading: "Uploading photo",
  server: "Checking it is sharp and well lit, then identifying the material",
  waking: "Waking up the model. This can take up to a minute",
  done: "Done",
};

/** What the progress bar shows at time `now`. `wakingAfterMs` counts from the start. */
export function progressView(p: ReadingProgress, now: number, wakingAfterMs: number): ProgressView {
  if (p.done) return { percent: 100, stage: "done", label: LABELS.done, estimated: false };
  if (p.uploadDoneAt === null) {
    const percent = Math.floor(UPLOAD_END * p.uploadFraction);
    return { percent, stage: "uploading", label: LABELS.uploading, estimated: false };
  }
  const waking = now - p.startedAt >= wakingAfterMs;
  const wakingAt = waking ? Math.max(0, p.startedAt + wakingAfterMs - p.uploadDoneAt) : null;
  const percent = Math.min(ESTIMATE_CAP, Math.floor(estimate(now - p.uploadDoneAt, wakingAt)));
  const stage = waking ? "waking" : "server";
  return { percent, stage, label: LABELS[stage], estimated: true };
}

/** The progress bar's aria-valuetext, e.g. "45%, estimated. Checking it is sharp ...". */
export function progressText(view: ProgressView): string {
  return `${view.percent}%${view.estimated ? ", estimated" : ""}. ${view.label}.`;
}
