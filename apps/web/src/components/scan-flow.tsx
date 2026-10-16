"use client";

// Capture -> reading -> (retake | questions) -> verdict, as in context.md Section 19.
// The state machine lives in lib/scan-flow.ts; this file is presentation only.
import { useRouter } from "next/navigation";
import { useEffect, useRef, useState } from "react";

import { ConfidenceMeter } from "@/components/confidence-meter";
import {
  ArrowLeftIcon,
  BinMark,
  CameraIcon,
  CheckIcon,
  CrossIcon,
  ExternalIcon,
  QuestionIcon,
  SparkIcon,
  Spinner,
} from "@/components/icons";
import { flyReward } from "@/components/points-pill";
import { QrScanner } from "@/components/qr-scanner";
import { Button, CaptureButton, Notice, PhotoInput, ScanLine, Screen, at } from "@/components/ui";
import {
  DAILY_POINTS_CAP,
  DEFAULT_SITE_POLICY,
  QUESTION_BANK,
  VERIFY_TIME_WINDOW_MINUTES,
} from "@/contracts/constants";
import * as api from "@/lib/api-client";
import type { BinQrPayload, VerdictResult } from "@/lib/contracts";
import { fileToBase64Jpeg } from "@/lib/image";
import { checkBin } from "@/lib/qr";
import { ensureSignedIn } from "@/lib/supabase/browser";
import { checkRows, type VerifyRouteResponse } from "@/lib/verify-record";
import {
  NEA_CHECK_MY_ITEM,
  answerLabel,
  binColour,
  fromAnswer,
  fromPredict,
  issueText,
  questionsLine,
  textOn,
  topCondition,
  topMaterial,
  type ScanState,
  type VisionAnalysis,
} from "@/lib/scan-flow";

const WAKING_AFTER_MS = 8_000;

const QUESTION_SHORT: Record<keyof typeof QUESTION_BANK, string> = {
  q1_recycling_symbol: "Recycling symbol",
  q2_cleaned: "Cleaned or rinsed",
  q3_sitting_time: "Time sitting",
};

export function ScanFlow() {
  const [state, setState] = useState<ScanState>({ step: "capture" });
  // Confidence before the latest answer, so the meter can fill from it.
  const [previousConfidence, setPreviousConfidence] = useState(0);
  // Focus moves to each new screen's heading, except on the very first page load.
  const [started, setStarted] = useState(false);

  // Start the anonymous (pseudonymous) session early so the first scan can be saved.
  useEffect(() => {
    void ensureSignedIn();
  }, []);
  const [lastImage, setLastImage] = useState<string | null>(null);
  const photoUrl = state.step === "capture" ? null : state.photoUrl;

  // Free the object URL when the photo changes or the component unmounts.
  useEffect(() => {
    return () => {
      if (photoUrl) URL.revokeObjectURL(photoUrl);
    };
  }, [photoUrl]);

  async function runPredict(url: string, image: string) {
    setPreviousConfidence(0);
    setState({ step: "reading", photoUrl: url });
    const res = await api.predict({ image_base64: image, site_policy: DEFAULT_SITE_POLICY });
    setState(
      res.ok ? fromPredict(url, res.data) : { step: "error", photoUrl: url, error: res.error },
    );
  }

  async function onPhoto(file: File) {
    setStarted(true);
    const url = URL.createObjectURL(file);
    try {
      const [image] = await Promise.all([fileToBase64Jpeg(file), ensureSignedIn()]);
      setLastImage(image);
      await runPredict(url, image);
    } catch {
      setState({
        step: "error",
        photoUrl: url,
        error: { code: "invalid_image", message: "We could not read that photo. Try another one." },
      });
    }
  }

  async function onAnswer(value: string) {
    if (state.step !== "question" || state.submitting) return;
    const current = { ...state, submitting: true };
    setState(current);
    const res = await api.answer({
      state: current.decision,
      answer: { question_id: current.question.id, answer: value },
    });
    setPreviousConfidence(current.decision.confidence);
    setState(
      res.ok
        ? fromAnswer(current, res.data)
        : { step: "error", photoUrl: current.photoUrl, error: res.error },
    );
  }

  const restart = () => {
    setLastImage(null);
    setPreviousConfidence(0);
    setState({ step: "capture" });
  };

  switch (state.step) {
    case "capture":
      return <CaptureScreen onPhoto={onPhoto} focusOnMount={started} />;
    case "reading":
      return <ReadingScreen photoUrl={state.photoUrl} />;
    case "retake":
      return (
        <Screen titleId="retake-title" title="Let’s try that photo again" eyebrow="Photo check">
          <ul className="card space-y-3 p-4" style={at(1)}>
            {state.issues.map((issue) => (
              <li key={issue} className="flex gap-3">
                <span className="mt-0.5 grid size-6 shrink-0 place-items-center rounded-full bg-sticker text-ink">
                  <CrossIcon className="size-3.5" />
                </span>
                <span>{issueText(issue)}</span>
              </li>
            ))}
          </ul>
          <ItemFrame url={state.photoUrl} style={at(2)} dim small />
          <div style={at(3)}>
            <CaptureButton label="Retake photo" onPhoto={onPhoto} />
          </div>
        </Screen>
      );
    case "question":
      return <QuestionScreen state={state} from={previousConfidence} onAnswer={onAnswer} />;
    case "verdict":
      return (
        <VerdictScreen
          key={state.scanId}
          state={state}
          from={previousConfidence}
          onRestart={restart}
        />
      );
    case "error": {
      const image = lastImage;
      const url = state.photoUrl;
      const waking = state.error.code === "model_unavailable";
      const canRetry = waking && image !== null && url !== null;
      return (
        <div role="alert">
          <Screen
            titleId="error-title"
            title={waking ? "The model is not ready yet" : "Something went wrong"}
            eyebrow={waking ? "Waking up" : "Error"}
          >
            <div className="card flex gap-4 p-4" style={at(1)}>
              <span className="grid size-11 shrink-0 place-items-center rounded-full bg-sticker text-ink">
                {waking ? <Spinner className="size-5" /> : <QuestionIcon className="size-6" />}
              </span>
              <p className="self-center">{state.error.message}</p>
            </div>
            <div className="flex flex-col gap-3" style={at(2)}>
              {canRetry && <Button onClick={() => void runPredict(url, image)}>Try again</Button>}
              <Button variant={canRetry ? "secondary" : "primary"} onClick={restart}>
                Start again
              </Button>
            </div>
          </Screen>
        </div>
      );
    }
  }
}

// ---- Screens ---------------------------------------------------------------

function CaptureScreen({
  onPhoto,
  focusOnMount,
}: {
  onPhoto: (file: File) => void;
  focusOnMount: boolean;
}) {
  const input = useRef<HTMLInputElement>(null);
  const steps = [
    { Icon: CameraIcon, title: "Photograph one item", body: "On its own, as it is now." },
    { Icon: QuestionIcon, title: "Answer a question", body: "Only if the photo is not enough." },
    { Icon: BinMark, title: "Get the bin and why", body: "With the rule and NEA source." },
  ];
  return (
    <Screen
      titleId="capture-title"
      title="What are you throwing away?"
      eyebrow="HDB blue bin rules"
      titleClassName="text-4xl"
      focusOnMount={focusOnMount}
    >
      <PhotoInput ref={input} onPhoto={onPhoto} />
      <button
        type="button"
        aria-label="Take a photo of one item"
        onClick={() => input.current?.click()}
        style={at(1)}
        className="viewfinder group relative grid aspect-[4/3] w-full place-items-center overflow-hidden rounded-[1.75rem] bg-ink text-sticker shadow-[var(--shadow-lift)] transition-transform duration-200 hover:-translate-y-0.5 active:scale-[0.99]"
      >
        <span aria-hidden className="corners" />
        <span
          aria-hidden
          className="absolute inset-0 bg-[radial-gradient(70%_60%_at_50%_45%,rgb(31_95_209/0.45),transparent_70%)]"
        />
        <span className="relative flex flex-col items-center gap-3 text-light">
          <span className="grid size-20 place-items-center rounded-full bg-bin-blue shadow-[0_0_0_10px_rgb(31_95_209/0.25)] transition-transform duration-300 group-hover:scale-105 animate-breathe">
            <CameraIcon className="size-9" />
          </span>
          <span className="font-heading text-2xl font-bold">Take a photo</span>
          <span className="text-sm text-light/80">One item, on its own</span>
        </span>
      </button>
      <p className="text-sm text-muted" style={at(2)}>
        Keep people and faces out of the picture. Bin Buddy keeps a fingerprint of each photo, not
        the photo itself.
      </p>
      <ol className="grid gap-3 sm:grid-cols-3" style={at(3)}>
        {steps.map(({ Icon, title, body }, i) => (
          <li key={title} className="card flex items-start gap-3 p-4 sm:flex-col">
            <span className="grid size-10 shrink-0 place-items-center rounded-xl bg-paper text-bin-blue">
              <Icon className="size-5" />
            </span>
            <span>
              <span className="block text-xs font-bold text-muted">Step {i + 1}</span>
              <span className="block font-bold">{title}</span>
              <span className="block text-sm text-muted">{body}</span>
            </span>
          </li>
        ))}
      </ol>
    </Screen>
  );
}

function ReadingScreen({ photoUrl }: { photoUrl: string }) {
  const [waking, setWaking] = useState(false);
  useEffect(() => {
    const t = setTimeout(() => setWaking(true), WAKING_AFTER_MS);
    return () => clearTimeout(t);
  }, []);
  return (
    <div aria-busy="true">
      <Screen titleId="reading-title" title="Reading your photo" eyebrow="One moment" live>
        <ItemFrame url={photoUrl} style={at(1)} scanning />
        <ul className="card divide-y divide-paper px-4" style={at(2)}>
          {["Checking it is sharp and well lit", "Identifying material and condition"].map(
            (line) => (
              <li key={line} className="flex items-center gap-3 py-3">
                <Spinner className="size-5 text-bin-blue" />
                <span>{line}</span>
              </li>
            ),
          )}
        </ul>
        {waking && (
          <Notice role="status">
            Waking up the model. The first scan can take up to a minute.
          </Notice>
        )}
      </Screen>
    </div>
  );
}

function QuestionScreen({
  state,
  from,
  onAnswer,
}: {
  state: Extract<ScanState, { step: "question" }>;
  from: number;
  onAnswer: (value: string) => void;
}) {
  const asked = state.decision.asked ?? [];
  return (
    <div className="space-y-5">
      <div className="card flex items-center gap-4 p-4 animate-rise">
        <ItemThumb url={state.photoUrl} />
        <div className="min-w-0 flex-1">
          <ConfidenceMeter value={state.decision.confidence} from={from} />
        </div>
      </div>
      <QuestionBody
        key={state.question.id}
        number={asked.length + 1}
        state={state}
        onAnswer={onAnswer}
      />
    </div>
  );
}

function QuestionBody({
  number,
  state,
  onAnswer,
}: {
  number: number;
  state: Extract<ScanState, { step: "question" }>;
  onAnswer: (value: string) => void;
}) {
  const [picked, setPicked] = useState<string | null>(null);
  const asked = state.decision.asked ?? [];
  return (
    <Screen
      titleId="question-title"
      title={state.question.text}
      eyebrow={`Question ${number} of up to 3`}
    >
      <p className="flex items-center gap-2 text-sm text-muted" style={at(1)}>
        <QuestionIcon className="size-4 shrink-0" />
        Asked only because the photo is not enough.
      </p>
      <div className="flex flex-col gap-3" style={at(2)}>
        {state.question.options.map((option) => (
          <button
            key={option}
            type="button"
            className="choice"
            aria-pressed={picked === option}
            disabled={state.submitting}
            onClick={() => {
              setPicked(option);
              onAnswer(option);
            }}
          >
            <span>{answerLabel(option)}</span>
            {picked === option && state.submitting && <Spinner className="size-5 text-bin-blue" />}
          </button>
        ))}
      </div>
      {asked.length > 0 && (
        <ul className="flex flex-wrap gap-2" style={at(3)} aria-label="Your answers so far">
          {asked.map((a) => (
            <li key={a.question_id} className="rounded-full bg-white px-3 py-1.5 text-sm shadow-sm">
              <span className="text-muted">{QUESTION_SHORT[a.question_id]}: </span>
              <strong>{answerLabel(a.answer)}</strong>
            </li>
          ))}
        </ul>
      )}
    </Screen>
  );
}

function VerdictScreen({
  state,
  from,
  onRestart,
}: {
  state: Extract<ScanState, { step: "verdict" }>;
  from: number;
  onRestart: () => void;
}) {
  const { verdict, analysis } = state;
  const abstain = verdict.channel === "abstain";
  // Demonstration bins exist only for blue-bin and general-waste verdicts.
  const hasDemoBin = verdict.channel === "recycle" || verdict.channel === "general_waste";
  const [atBin, setAtBin] = useState(false);
  if (atBin)
    return (
      <AtBinScreen
        scanId={state.scanId}
        verdict={verdict}
        onBack={() => setAtBin(false)}
        onDone={onRestart}
      />
    );
  return (
    <section aria-labelledby="verdict-title" className="stagger space-y-5">
      <BinCard verdict={verdict} photoUrl={state.photoUrl} from={from} />
      {verdict.tip && (
        <div
          className="relative -rotate-[0.6deg] rounded-2xl bg-sticker p-4 pt-5 text-ink shadow-[var(--shadow-card)]"
          style={at(1)}
        >
          <span className="absolute -top-3 left-4 inline-flex items-center gap-1 rounded-full bg-ink px-2.5 py-0.5 text-xs font-bold text-sticker">
            <SparkIcon className="size-3.5" /> Tip
          </span>
          <p>{verdict.tip}</p>
        </div>
      )}
      {!abstain && <HeatmapFigure url={state.photoUrl} analysis={analysis} style={at(2)} />}
      <DecisionTrace verdict={verdict} analysis={analysis} style={at(3)} />
      <a
        className="card flex min-h-12 items-center justify-between gap-3 px-4 py-3 font-bold text-bin-blue"
        href={verdict.source_url ?? NEA_CHECK_MY_ITEM}
        target="_blank"
        rel="noreferrer"
        style={at(4)}
      >
        <span>
          <span className="block text-xs font-bold text-muted">Source</span>
          {abstain ? "Check this item on NEA Check My Item" : "NEA guidance"}
        </span>
        <ExternalIcon className="size-5 shrink-0" />
      </a>
      <div className="flex flex-col gap-3" style={at(5)}>
        {hasDemoBin && <Button onClick={() => setAtBin(true)}>I’m at the bin</Button>}
        <Button variant={hasDemoBin ? "secondary" : "primary"} onClick={onRestart}>
          Scan another item
        </Button>
      </div>
    </section>
  );
}

function AtBinScreen({
  scanId,
  verdict,
  onBack,
  onDone,
}: {
  scanId: string;
  verdict: VerdictResult;
  onBack: () => void;
  onDone: () => void;
}) {
  const [bin, setBin] = useState<BinQrPayload | null>(null);
  if (!bin) {
    return (
      <Screen
        titleId="at-bin-title"
        title="Scan the QR code on the bin"
        eyebrow="At the bin · 1 of 2"
      >
        <p className="text-muted" style={at(1)}>
          Point your camera at the Bin Buddy sticker on the bin.
        </p>
        <div style={at(2)}>
          <QrScanner onBin={setBin} />
        </div>
        <div style={at(3)}>
          <BackButton onClick={onBack}>Back to the verdict</BackButton>
        </div>
      </Screen>
    );
  }
  const check = checkBin(verdict, DEFAULT_SITE_POLICY, bin);
  if (check.kind === "match")
    return <VerifyScreen scanId={scanId} bin={bin} onBack={onBack} onDone={onDone} />;
  const message = {
    wrong_bin: `Bin Buddy recommended the ${verdict.bin_label}. Please find that bin instead.`,
    other_site:
      "This bin follows a different site's sorting rules. Please find your nearest " +
      `${verdict.bin_label}.`,
    no_bin: "Bin Buddy did not recommend a bin for this item.",
  }[check.kind];
  return (
    <Screen
      titleId="at-bin-title"
      title="Not this bin"
      eyebrow={`Bin identified: ${bin.location_label}`}
      live
    >
      <div className="card flex gap-4 p-4" style={at(1)}>
        <span className="grid size-11 shrink-0 place-items-center rounded-full bg-general text-light">
          <CrossIcon className="size-5" />
        </span>
        <p className="self-center">{message}</p>
      </div>
      <div className="flex flex-col gap-3" style={at(2)}>
        <Button onClick={() => setBin(null)}>Scan another bin</Button>
        <BackButton onClick={onBack}>Back to the verdict</BackButton>
      </div>
    </Screen>
  );
}

type VerifyPhase =
  | { kind: "ready" }
  | { kind: "checking" }
  | { kind: "done"; result: VerifyRouteResponse }
  | { kind: "error"; message: string };

function VerifyScreen({
  scanId,
  bin,
  onBack,
  onDone,
}: {
  scanId: string;
  bin: BinQrPayload;
  onBack: () => void;
  onDone: () => void;
}) {
  const [phase, setPhase] = useState<VerifyPhase>({ kind: "ready" });

  async function onPhoto(file: File) {
    setPhase({ kind: "checking" });
    try {
      const image = await fileToBase64Jpeg(file);
      const res = await api.verify({ scan_id: scanId, image_base64: image, qr: bin });
      setPhase(
        res.ok ? { kind: "done", result: res.data } : { kind: "error", message: res.error.message },
      );
    } catch {
      setPhase({ kind: "error", message: "We could not read that photo. Try again." });
    }
  }

  if (phase.kind === "checking") {
    return (
      <div aria-busy="true">
        <Screen
          titleId="verify-title"
          title="Checking your disposal"
          eyebrow="At the bin · 2 of 2"
          live
        >
          <ul className="card divide-y divide-paper px-4" style={at(1)}>
            {[
              "Same item",
              "Right bin",
              `Within ${VERIFY_TIME_WINDOW_MINUTES} minutes`,
              "New photo",
            ].map((line) => (
              <li key={line} className="flex items-center gap-3 py-3">
                <Spinner className="size-5 text-bin-blue" />
                <span>{line}</span>
              </li>
            ))}
          </ul>
        </Screen>
      </div>
    );
  }

  if (phase.kind === "done")
    return <VerifyResult result={phase.result} onPhoto={onPhoto} onDone={onDone} />;

  return (
    <Screen
      titleId="verify-title"
      title="Right bin. Photograph the item here"
      eyebrow="At the bin · 2 of 2"
    >
      <p
        className="flex items-center gap-3 rounded-2xl bg-bin-blue p-4 font-bold text-light"
        style={at(1)}
      >
        <span className="grid size-8 shrink-0 place-items-center rounded-full bg-light text-bin-blue">
          <CheckIcon className="size-4" />
        </span>
        This is the bin Bin Buddy recommended: {bin.location_label}.
      </p>
      <p style={at(2)}>
        Take a new photo of the item at the bin, before you drop it in. Keep people out of the
        picture.
      </p>
      {phase.kind === "error" && <Notice role="alert">{phase.message}</Notice>}
      <div className="flex flex-col gap-3" style={at(3)}>
        <CaptureButton label="Take the photo" onPhoto={onPhoto} />
        <BackButton onClick={onBack}>Back to the verdict</BackButton>
      </div>
    </Screen>
  );
}

function VerifyResult({
  result,
  onPhoto,
  onDone,
}: {
  result: VerifyRouteResponse;
  onPhoto: (file: File) => void;
  onDone: () => void;
}) {
  const router = useRouter();
  const reward = useRef<HTMLParagraphElement>(null);
  const flown = useRef(false);
  const rows = checkRows(result, VERIFY_TIME_WINDOW_MINUTES);

  // Fly the sticker to the header, then re-render the server header so the new total shows.
  useEffect(() => {
    if (flown.current || result.points_awarded <= 0) return;
    flown.current = true;
    if (reward.current) flyReward(reward.current, result.points_awarded);
    const t = setTimeout(() => router.refresh(), 900);
    return () => clearTimeout(t);
  }, [result.points_awarded, router]);

  return (
    <Screen
      titleId="verify-title"
      title={result.passed ? "Verified" : "Not verified this time"}
      eyebrow="At the bin · done"
      titleClassName="text-4xl"
      live
    >
      {result.passed && (
        <p
          ref={reward}
          className="flex items-center gap-4 rounded-2xl bg-sticker p-4 font-bold text-ink shadow-[var(--shadow-card)] animate-drop"
        >
          <span className="grid size-12 shrink-0 place-items-center rounded-full bg-ink text-sticker">
            <CheckIcon className="size-6" />
          </span>
          <span className="font-heading text-xl">
            {result.points_awarded > 0
              ? `+${result.points_awarded} point${result.points_awarded === 1 ? "" : "s"}`
              : `Thanks! You have reached today's limit of ${DAILY_POINTS_CAP} points.`}
          </span>
        </p>
      )}
      <ul className="card stagger divide-y divide-paper px-4" style={at(1)} aria-label="Checks">
        {rows.map((row, i) => (
          <li key={row.key} className="flex items-center gap-3 py-3" style={at(i + 1)}>
            <span
              aria-hidden
              className={`grid size-7 shrink-0 place-items-center rounded-full ${
                row.passed ? "bg-bin-blue text-light" : "bg-general text-light"
              }`}
            >
              {row.passed ? <CheckIcon className="size-4" /> : <CrossIcon className="size-4" />}
            </span>
            <span className="flex-1">{row.label}</span>
            <strong className="text-sm">{row.passed ? "Passed" : "Not passed"}</strong>
          </li>
        ))}
      </ul>
      <p className="text-sm text-muted" style={at(2)}>
        The photo is deleted now. Only a fingerprint of it is kept, to stop reuse.
      </p>
      <div className="flex flex-col gap-3" style={at(3)}>
        {!result.passed && <CaptureButton label="Try another photo" onPhoto={onPhoto} />}
        <Button variant={result.passed ? "primary" : "secondary"} onClick={onDone}>
          Scan another item
        </Button>
      </div>
    </Screen>
  );
}

// ---- Pieces ----------------------------------------------------------------

/** Signature element: the verdict lands like an item dropped in a bin, and the lid lifts. */
function BinCard({
  verdict,
  photoUrl,
  from,
}: {
  verdict: VerdictResult;
  photoUrl: string;
  from: number;
}) {
  const bg = binColour(verdict);
  const fg = textOn(bg);
  // The meter panel shifts the card colour away from the text colour, so contrast only grows.
  const panel = fg === "#10233F" ? "bg-white/25" : "bg-black/15";
  const heading = useRef<HTMLHeadingElement>(null);
  useEffect(() => heading.current?.focus(), []);
  return (
    <div className="pt-3 animate-drop" style={{ color: fg }}>
      <div
        aria-hidden
        className="mx-auto h-3 w-2/5 origin-left rounded-t-lg animate-lid [animation-delay:350ms]"
        style={{ background: bg, filter: "brightness(0.8)" }}
      />
      <div
        className="overflow-hidden rounded-[1.75rem] shadow-[var(--shadow-lift)]"
        style={{ background: bg }}
      >
        <div className="flex items-start gap-4 p-5 pb-4">
          <div className="min-w-0 flex-1 space-y-2">
            <p className="text-sm font-bold">
              {verdict.channel === "abstain" ? "No verdict" : "Where it goes"}
            </p>
            <h1
              id="verdict-title"
              ref={heading}
              tabIndex={-1}
              data-screen-heading
              className="font-heading text-3xl font-bold leading-tight text-balance"
            >
              {verdict.bin_label}
            </h1>
          </div>
          <ItemThumb url={photoUrl} className="ring-4 ring-white/30" />
        </div>
        <p className="px-5">{verdict.reason}</p>
        <div className={`m-3 mt-4 rounded-2xl p-4 ${panel}`}>
          <ConfidenceMeter
            value={verdict.confidence}
            from={from}
            tone="onColour"
            caption={questionsLine(verdict)}
          />
        </div>
      </div>
    </div>
  );
}

/**
 * Signature element: the Grad-CAM highlight sweeps over the photo. The model reads the centre
 * square of the photo (resize to 256, centre crop 224), so the photo is shown with the same
 * crop and the overlay lines up with it.
 */
function HeatmapFigure({
  url,
  analysis,
  style,
}: {
  url: string;
  analysis: VisionAnalysis | null;
  style?: React.CSSProperties;
}) {
  const heatmap = analysis?.heatmap_png_base64;
  const [showHeatmap, setShowHeatmap] = useState(true);
  if (!heatmap) return <ItemFrame url={url} style={style} />;
  return (
    <figure className="card space-y-3 p-3" style={style}>
      <div className="relative aspect-square overflow-hidden rounded-2xl bg-ink">
        {/* eslint-disable-next-line @next/next/no-img-element -- local blob URL */}
        <img src={url} alt="" className="absolute inset-0 size-full scale-[1.143] object-cover" />
        {/* eslint-disable-next-line @next/next/no-img-element -- inline data URL */}
        <img
          src={`data:image/png;base64,${heatmap}`}
          alt={showHeatmap ? "Your photo with the area that decided the verdict highlighted" : ""}
          className={`absolute inset-0 size-full object-cover transition-opacity duration-300 animate-sweep ${
            showHeatmap ? "opacity-100" : "opacity-0"
          }`}
        />
        {!showHeatmap && <span className="sr-only">Your photo, without the highlight</span>}
      </div>
      <figcaption className="flex flex-wrap items-center justify-between gap-3 px-1 text-sm text-muted">
        <span className="min-w-0 flex-1">
          {showHeatmap
            ? "Highlighted: the area that decided the verdict."
            : "The part of your photo the model looked at."}
        </span>
        <span role="group" aria-label="Photo view" className="flex rounded-xl bg-paper p-1">
          {[
            { on: true, label: "Highlight" },
            { on: false, label: "Photo" },
          ].map(({ on, label }) => (
            <button
              key={label}
              type="button"
              aria-pressed={showHeatmap === on}
              onClick={() => setShowHeatmap(on)}
              className={`min-h-11 rounded-lg px-3 font-bold transition-colors ${
                showHeatmap === on ? "bg-white text-ink shadow-sm" : "text-muted"
              }`}
            >
              {label}
            </button>
          ))}
        </span>
      </figcaption>
    </figure>
  );
}

/** "How Bin Buddy decided": what it saw, what you said, and the rule, as one trail. */
function DecisionTrace({
  verdict,
  analysis,
  style,
}: {
  verdict: VerdictResult;
  analysis: VisionAnalysis | null;
  style?: React.CSSProperties;
}) {
  const material = topMaterial(analysis);
  const condition = topCondition(analysis);
  const answers = verdict.questions_asked ?? [];
  const trace = verdict.rule_trace.length > 0 ? verdict.rule_trace : [verdict.rule_id];
  return (
    <div className="card space-y-4 p-4" style={style}>
      <h2 className="font-heading text-lg font-bold">How Bin Buddy decided</h2>
      <dl className="grid grid-cols-2 gap-2">
        {material && <Fact term="Material" value={material} />}
        {condition && <Fact term="Condition" value={condition} />}
        {answers.map((qa) => (
          <Fact
            key={qa.question_id}
            term={QUESTION_SHORT[qa.question_id]}
            value={answerLabel(qa.answer)}
            you
          />
        ))}
      </dl>
      <div>
        <p className="mb-2 text-sm font-bold text-muted">Rule applied</p>
        <ol className="relative space-y-3 pl-6 before:absolute before:top-2 before:bottom-2 before:left-[7px] before:w-0.5 before:rounded-full before:bg-line">
          {trace.map((step, i) => (
            <li key={`${i}-${step}`} className="relative">
              <span
                aria-hidden
                className={`absolute top-1.5 -left-6 size-4 rounded-full border-[3px] border-white ${
                  i === trace.length - 1 ? "bg-bin-blue" : "bg-muted"
                } shadow-[0_0_0_1px_var(--color-line)]`}
              />
              {step}
            </li>
          ))}
        </ol>
      </div>
    </div>
  );
}

function Fact({ term, value, you = false }: { term: string; value: string; you?: boolean }) {
  return (
    <div className="rounded-xl bg-paper p-3">
      <dt className="text-xs font-bold text-muted">
        {term}
        {you && " · you said"}
      </dt>
      <dd className="font-bold">{value}</dd>
    </div>
  );
}

/** The item as the model sees it: the centre square of the photo. */
function ItemFrame({
  url,
  style,
  scanning = false,
  dim = false,
  small = false,
}: {
  url: string;
  style?: React.CSSProperties;
  scanning?: boolean;
  dim?: boolean;
  small?: boolean;
}) {
  return (
    <div
      style={style}
      className={`viewfinder relative mx-auto aspect-square w-full overflow-hidden rounded-[1.75rem] bg-ink text-sticker shadow-[var(--shadow-lift)] ${
        small ? "max-w-56" : "max-w-sm"
      }`}
    >
      {/* eslint-disable-next-line @next/next/no-img-element -- local blob URL */}
      <img
        src={url}
        alt="Your photo"
        className={`absolute inset-0 size-full scale-[1.143] object-cover ${dim ? "opacity-60" : ""}`}
      />
      <span aria-hidden className="corners" />
      {scanning && <ScanLine />}
    </div>
  );
}

function ItemThumb({ url, className = "" }: { url: string; className?: string }) {
  return (
    // eslint-disable-next-line @next/next/no-img-element -- local blob URL
    <img
      src={url}
      alt=""
      className={`size-16 shrink-0 rounded-2xl bg-white object-cover shadow-sm ${className}`}
    />
  );
}

function BackButton({ onClick, children }: { onClick: () => void; children: React.ReactNode }) {
  return (
    <Button variant="secondary" onClick={onClick}>
      <ArrowLeftIcon className="size-5" />
      {children}
    </Button>
  );
}
