"use client";

// Holds the scan in progress for the whole app. It is mounted in the root layout, which stays
// mounted when BottomNav switches between Scan, Leaderboard and Profile, so a photo, a running
// /predict or /answer call and a question in progress all survive a tab switch. The question and
// verdict steps are also saved to sessionStorage, so a reload keeps them.
import { createContext, use, useEffect, useLayoutEffect, useRef, useState } from "react";

import { DEFAULT_SITE_POLICY } from "@/contracts/constants";
import * as api from "@/lib/api-client";
import { fileToBase64Jpeg, fileToThumbnail } from "@/lib/image";
import { fromAnswer, fromPredict, type ScanState } from "@/lib/scan-flow";
import { startProgress, withUpload, type ReadingProgress } from "@/lib/scan-progress";
import { SCAN_SESSION_KEY, restoreScan, serialiseScan } from "@/lib/scan-session";
import { ensureSignedIn } from "@/lib/supabase/browser";

// How long the bar rests at 100% before the result replaces it.
const DONE_HOLD_MS = 300;

type Scan = {
  state: ScanState;
  /** Confidence before the latest answer, so the meter can fill from it. */
  previousConfidence: number;
  /** A photo was taken in this visit (focus then moves to each new screen's heading). */
  started: boolean;
  /** False until sessionStorage has been read, so a restored step does not flash the capture screen. */
  ready: boolean;
  progress: ReadingProgress | null;
  /** The last /predict can be sent again (the model was not ready). */
  canRetry: boolean;
  takePhoto: (file: File) => Promise<void>;
  retry: () => void;
  answer: (value: string) => Promise<void>;
  restart: () => void;
};

const ScanContext = createContext<Scan | null>(null);

export function useScan(): Scan {
  const scan = use(ScanContext);
  if (!scan) throw new Error("useScan must be used inside <ScanProvider>");
  return scan;
}

// sessionStorage can be missing or throw (private mode, storage full, blocked site data).
function readSaved(): string | null {
  try {
    return sessionStorage.getItem(SCAN_SESSION_KEY);
  } catch {
    return null;
  }
}

function writeSaved(value: string | null) {
  try {
    if (value === null) sessionStorage.removeItem(SCAN_SESSION_KEY);
    else sessionStorage.setItem(SCAN_SESSION_KEY, value);
    return true;
  } catch {
    return false;
  }
}

export function ScanProvider({ children }: { children: React.ReactNode }) {
  const [state, setState] = useState<ScanState>({ step: "capture" });
  const [previousConfidence, setPreviousConfidence] = useState(0);
  const [started, setStarted] = useState(false);
  const [ready, setReady] = useState(false);
  const [progress, setProgress] = useState<ReadingProgress | null>(null);
  const [lastImage, setLastImage] = useState<string | null>(null);
  // Bumped by every new photo, retry and restart: replies to an older run are dropped.
  const run = useRef(0);
  // The current photo's blob URL. Revoked only when a new photo replaces it or on restart,
  // never because a screen unmounted.
  const photo = useRef<string | null>(null);
  const thumbnail = useRef<string | null>(null);

  // Start the anonymous (pseudonymous) session early so the first scan can be saved. The browser
  // client also keeps the session cookie refreshed while the app is open.
  useEffect(() => {
    void ensureSignedIn();
  }, []);

  // Restore before the first paint, so a reload on a question shows the question at once.
  // sessionStorage only exists in the browser, so this cannot be the initial state: the server
  // render and hydration must both start from the capture step.
  useLayoutEffect(() => {
    const saved = restoreScan(readSaved());
    if (saved) {
      thumbnail.current = saved.state.photoUrl;
      // eslint-disable-next-line react-hooks/set-state-in-effect -- one-off read after hydration
      setState(saved.state);
      setPreviousConfidence(saved.previousConfidence);
    }
    setReady(true);
  }, []);

  // Save the question and verdict steps. If the thumbnail does not fit, save without it.
  useEffect(() => {
    if (state.step !== "question" && state.step !== "verdict") return;
    if (!writeSaved(serialiseScan(state, previousConfidence, thumbnail.current))) {
      writeSaved(serialiseScan(state, previousConfidence, null));
    }
  }, [state, previousConfidence]);

  function replacePhoto(url: string | null) {
    if (photo.current) URL.revokeObjectURL(photo.current);
    photo.current = url;
  }

  async function predict(id: number, photoUrl: string, image: string) {
    setPreviousConfidence(0);
    setProgress(startProgress(Date.now()));
    setState({ step: "reading", photoUrl });
    const res = await api.predict(
      { image_base64: image, site_policy: DEFAULT_SITE_POLICY },
      (fraction) => {
        if (id === run.current) setProgress((p) => p && withUpload(p, fraction, Date.now()));
      },
    );
    if (id !== run.current) return;
    if (res.ok) {
      setProgress((p) => p && { ...p, done: true });
      await new Promise((r) => setTimeout(r, DONE_HOLD_MS));
      if (id !== run.current) return;
    }
    setProgress(null);
    setState(
      res.ok ? fromPredict(photoUrl, res.data) : { step: "error", photoUrl, error: res.error },
    );
  }

  async function takePhoto(file: File) {
    const id = ++run.current;
    setStarted(true);
    const url = URL.createObjectURL(file);
    replacePhoto(url);
    thumbnail.current = null;
    writeSaved(null);
    try {
      const [image, thumb] = await Promise.all([
        fileToBase64Jpeg(file),
        fileToThumbnail(file).catch(() => null),
        ensureSignedIn(),
      ]);
      if (id !== run.current) return;
      thumbnail.current = thumb;
      setLastImage(image);
      await predict(id, url, image);
    } catch {
      if (id !== run.current) return;
      setState({
        step: "error",
        photoUrl: url,
        error: { code: "invalid_image", message: "We could not read that photo. Try another one." },
      });
    }
  }

  const canRetry =
    state.step === "error" &&
    state.error.code === "model_unavailable" &&
    lastImage !== null &&
    state.photoUrl !== null;

  function retry() {
    if (state.step !== "error" || !canRetry || state.photoUrl === null || lastImage === null)
      return;
    void predict(++run.current, state.photoUrl, lastImage);
  }

  async function answer(value: string) {
    if (state.step !== "question" || state.submitting) return;
    const id = run.current;
    const current = { ...state, submitting: true };
    setState(current);
    const res = await api.answer({
      state: current.decision,
      answer: { question_id: current.question.id, answer: value },
    });
    if (id !== run.current) return;
    setPreviousConfidence(current.decision.confidence);
    setState(
      res.ok
        ? fromAnswer(current, res.data)
        : { step: "error", photoUrl: current.photoUrl, error: res.error },
    );
  }

  function restart() {
    run.current++;
    replacePhoto(null);
    thumbnail.current = null;
    writeSaved(null);
    setLastImage(null);
    setPreviousConfidence(0);
    setProgress(null);
    setState({ step: "capture" });
  }

  const value: Scan = {
    state,
    previousConfidence,
    started,
    ready,
    progress,
    canRetry,
    takePhoto,
    retry,
    answer,
    restart,
  };
  return <ScanContext value={value}>{children}</ScanContext>;
}
