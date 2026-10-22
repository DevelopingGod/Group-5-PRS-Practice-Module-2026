// Progress for one /predict call (the "Reading your photo" screen). Pure, so it can be tested.
// Only the upload can be measured. The server runs the quality check and the vision model inside
// one request and cannot report its own progress, so after the upload the bar shows an eased
// estimate that never passes ESTIMATE_CAP until the reply arrives; then it jumps to 100%.
// The screen shows the bar and the seconds elapsed, never the percentage: it is only an estimate.

export const UPLOAD_END = 30;
export const ESTIMATE_CAP = 90;
// Before the cold-start notice the estimate eases towards ESTIMATE_SOFT_CAP; once the model is
// waking up it continues slowly towards ESTIMATE_CAP over about a minute.
const ESTIMATE_SOFT_CAP = 70;
const NORMAL_TIME_CONSTANT_MS = 2_500;
const WAKING_TIME_CONSTANT_MS = 20_000;

export type ReadingProgress = {
  startedAt: number;
  /** Milliseconds after the start at which a slow reply means the model is waking up; null when
   * the model answered recently and is known to be awake (later scans take about 10 s). */
  wakingAfterMs: number | null;
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
  /** Whole seconds since the photo was sent. */
  seconds: number;
};

// The free model host sleeps after 15 minutes without traffic and takes about 50 s to wake
// (context.md Section 14). A reply within the last few minutes means it is awake.
const MODEL_SLEEPS_AFTER_MS = 15 * 60_000;
export const WAKING_AFTER_MS = 8_000;

/** When to say the model is waking up: never if it replied less than 15 minutes ago. */
export function wakingAfter(lastReplyAt: number | null, now: number): number | null {
  const awake = lastReplyAt !== null && now - lastReplyAt < MODEL_SLEEPS_AFTER_MS;
  return awake ? null : WAKING_AFTER_MS;
}

export function startProgress(now: number, wakingAfterMs: number | null): ReadingProgress {
  return { startedAt: now, wakingAfterMs, uploadFraction: 0, uploadDoneAt: null, done: false };
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
  server: "Checking the photo and identifying the material",
  waking: "Waking up the model. This can take up to a minute",
  done: "Done",
};

/** What the progress bar shows at time `now`. */
export function progressView(p: ReadingProgress, now: number): ProgressView {
  const seconds = Math.max(0, Math.floor((now - p.startedAt) / 1000));
  if (p.done) return { percent: 100, stage: "done", label: LABELS.done, estimated: false, seconds };
  if (p.uploadDoneAt === null) {
    const percent = Math.floor(UPLOAD_END * p.uploadFraction);
    return { percent, stage: "uploading", label: LABELS.uploading, estimated: false, seconds };
  }
  const waking = p.wakingAfterMs !== null && now - p.startedAt >= p.wakingAfterMs;
  const wakingAt =
    waking && p.wakingAfterMs !== null
      ? Math.max(0, p.startedAt + p.wakingAfterMs - p.uploadDoneAt)
      : null;
  const percent = Math.min(ESTIMATE_CAP, Math.floor(estimate(now - p.uploadDoneAt, wakingAt)));
  const stage = waking ? "waking" : "server";
  return { percent, stage, label: LABELS[stage], estimated: true, seconds };
}

/** The progress bar's aria-valuetext, e.g. "Checking the photo ... 4 seconds so far.". */
export function progressText(view: ProgressView): string {
  if (view.stage === "done") return `${view.label}.`;
  return `${view.label}. ${view.seconds} second${view.seconds === 1 ? "" : "s"} so far.`;
}
