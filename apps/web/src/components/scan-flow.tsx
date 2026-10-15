"use client";

// Capture -> reading -> (retake | questions) -> verdict, as in context.md Section 19.
import { useEffect, useRef, useState } from "react";

import { DEFAULT_SITE_POLICY } from "@/contracts/constants";
import * as api from "@/lib/api-client";
import type { BinQrPayload, VerdictResult } from "@/lib/contracts";
import { fileToBase64Jpeg } from "@/lib/image";
import { checkBin } from "@/lib/qr";
import { QrScanner } from "@/components/qr-scanner";
import { ensureSignedIn } from "@/lib/supabase/browser";
import {
  NEA_CHECK_MY_ITEM,
  answerLabel,
  binColour,
  confidenceLine,
  fromAnswer,
  fromPredict,
  issueText,
  percent,
  textOn,
  topCondition,
  topMaterial,
  type ScanState,
  type VisionAnalysis,
} from "@/lib/scan-flow";

const WAKING_AFTER_MS = 8_000;

export function ScanFlow() {
  const [state, setState] = useState<ScanState>({ step: "capture" });

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
    setState({ step: "reading", photoUrl: url });
    const res = await api.predict({ image_base64: image, site_policy: DEFAULT_SITE_POLICY });
    setState(
      res.ok ? fromPredict(url, res.data) : { step: "error", photoUrl: url, error: res.error },
    );
  }

  async function onPhoto(file: File) {
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
    setState(
      res.ok
        ? fromAnswer(current, res.data)
        : { step: "error", photoUrl: current.photoUrl, error: res.error },
    );
  }

  const restart = () => {
    setLastImage(null);
    setState({ step: "capture" });
  };

  switch (state.step) {
    case "capture":
      return <CaptureScreen onPhoto={onPhoto} />;
    case "reading":
      return <ReadingScreen photoUrl={state.photoUrl} />;
    case "retake":
      return (
        <section aria-labelledby="retake-title" className="space-y-4">
          <Photo url={state.photoUrl} />
          <h1 id="retake-title" className="font-heading text-2xl font-bold">
            Let&apos;s try that photo again
          </h1>
          <ul className="list-disc space-y-1 pl-6">
            {state.issues.map((issue) => (
              <li key={issue}>{issueText(issue)}</li>
            ))}
          </ul>
          <CaptureButton label="Retake photo" onPhoto={onPhoto} />
        </section>
      );
    case "question":
      return <QuestionScreen state={state} onAnswer={onAnswer} />;
    case "verdict":
      return <VerdictScreen state={state} onRestart={restart} />;
    case "error": {
      const image = lastImage;
      const url = state.photoUrl;
      const canRetry = state.error.code === "model_unavailable" && image !== null && url !== null;
      return (
        <section role="alert" className="space-y-4">
          <h1 className="font-heading text-2xl font-bold">
            {state.error.code === "model_unavailable"
              ? "The model is not ready yet"
              : "Something went wrong"}
          </h1>
          <p>{state.error.message}</p>
          <div className="flex flex-col gap-3">
            {canRetry && <Button onClick={() => void runPredict(url, image)}>Try again</Button>}
            <Button variant="secondary" onClick={restart}>
              Start again
            </Button>
          </div>
        </section>
      );
    }
  }
}

// ---- Screens ---------------------------------------------------------------

function CaptureScreen({ onPhoto }: { onPhoto: (file: File) => void }) {
  return (
    <section aria-labelledby="capture-title" className="space-y-5">
      <h1 id="capture-title" className="font-heading text-3xl font-bold">
        What are you throwing away?
      </h1>
      <p className="text-muted">
        Take a photo of one item, on its own. Keep people and faces out of the picture.
      </p>
      <CaptureButton label="Take a photo" onPhoto={onPhoto} />
    </section>
  );
}

function ReadingScreen({ photoUrl }: { photoUrl: string }) {
  const [waking, setWaking] = useState(false);
  useEffect(() => {
    const t = setTimeout(() => setWaking(true), WAKING_AFTER_MS);
    return () => clearTimeout(t);
  }, []);
  return (
    <section aria-live="polite" aria-busy="true" className="space-y-4">
      <Photo url={photoUrl} />
      <p className="font-heading text-xl font-bold">Identifying material and condition…</p>
      {waking && (
        <p className="rounded-lg bg-sticker p-3 text-ink">
          Waking up the model. The first scan can take up to a minute.
        </p>
      )}
    </section>
  );
}

function QuestionScreen({
  state,
  onAnswer,
}: {
  state: Extract<ScanState, { step: "question" }>;
  onAnswer: (value: string) => void;
}) {
  const asked = state.decision.asked ?? [];
  const confidence = percent(state.decision.confidence);
  return (
    <section aria-labelledby="question-title" className="space-y-5">
      <div>
        <div className="flex justify-between text-sm text-muted">
          <span>Not sure yet</span>
          <span>{confidence}% sure</span>
        </div>
        <div
          role="progressbar"
          aria-label="Confidence"
          aria-valuenow={confidence}
          aria-valuemin={0}
          aria-valuemax={100}
          className="mt-1 h-3 overflow-hidden rounded-full bg-white"
        >
          <div className="h-full bg-bin-blue" style={{ width: `${confidence}%` }} />
        </div>
      </div>

      {asked.length > 0 && (
        <ul className="space-y-1 text-sm text-muted">
          {asked.map((a) => (
            <li key={a.question_id}>
              You answered: <strong className="text-ink">{answerLabel(a.answer)}</strong>
            </li>
          ))}
        </ul>
      )}

      <p className="text-sm text-muted">
        Question {asked.length + 1} of up to 3. Asked only because the photo is not enough.
      </p>
      <h1 id="question-title" className="font-heading text-2xl font-bold">
        {state.question.text}
      </h1>
      <div className="flex flex-col gap-3">
        {state.question.options.map((option) => (
          <Button
            key={option}
            variant="secondary"
            disabled={state.submitting}
            onClick={() => onAnswer(option)}
          >
            {answerLabel(option)}
          </Button>
        ))}
      </div>
    </section>
  );
}

function VerdictScreen({
  state,
  onRestart,
}: {
  state: Extract<ScanState, { step: "verdict" }>;
  onRestart: () => void;
}) {
  const { verdict, analysis } = state;
  const abstain = verdict.channel === "abstain";
  // Demonstration bins exist only for blue-bin and general-waste verdicts.
  const hasDemoBin = verdict.channel === "recycle" || verdict.channel === "general_waste";
  const [atBin, setAtBin] = useState(false);
  if (atBin) return <AtBinScreen verdict={verdict} onBack={() => setAtBin(false)} />;
  return (
    <section aria-labelledby="verdict-title" className="space-y-5">
      <BinCard verdict={verdict} />
      {verdict.tip && (
        <p className="rounded-lg bg-sticker p-4 text-ink">
          <strong>Tip: </strong>
          {verdict.tip}
        </p>
      )}
      {!abstain && <PhotoWithHeatmap url={state.photoUrl} analysis={analysis} />}
      <WhyRows verdict={verdict} analysis={analysis} />
      <p>
        Source:{" "}
        <a
          className="font-bold text-bin-blue underline"
          href={verdict.source_url ?? NEA_CHECK_MY_ITEM}
          target="_blank"
          rel="noreferrer"
        >
          {abstain ? "Check this item on NEA Check My Item" : "NEA guidance"}
        </a>
      </p>
      <div className="flex flex-col gap-3">
        {hasDemoBin && <Button onClick={() => setAtBin(true)}>I&apos;m at the bin</Button>}
        <Button variant={hasDemoBin ? "secondary" : "primary"} onClick={onRestart}>
          Scan another item
        </Button>
      </div>
    </section>
  );
}

function AtBinScreen({ verdict, onBack }: { verdict: VerdictResult; onBack: () => void }) {
  const [bin, setBin] = useState<BinQrPayload | null>(null);
  if (!bin) {
    return (
      <section aria-labelledby="at-bin-title" className="space-y-4">
        <h1 id="at-bin-title" className="font-heading text-2xl font-bold">
          Scan the QR code on the bin
        </h1>
        <p className="text-muted">Point your camera at the Bin Buddy sticker on the bin.</p>
        <QrScanner onBin={setBin} />
        <Button variant="secondary" onClick={onBack}>
          Back to the verdict
        </Button>
      </section>
    );
  }
  const check = checkBin(verdict, DEFAULT_SITE_POLICY, bin);
  const match = check.kind === "match";
  const message = {
    match: "This is the bin Bin Buddy recommended.",
    wrong_bin: `Bin Buddy recommended the ${verdict.bin_label}. Please find that bin instead.`,
    other_site:
      "This bin follows a different site's sorting rules. Please find your nearest " +
      `${verdict.bin_label}.`,
    no_bin: "Bin Buddy did not recommend a bin for this item.",
  }[check.kind];
  return (
    <section aria-labelledby="at-bin-title" aria-live="polite" className="space-y-4">
      <p className="text-sm text-muted">Bin identified: {bin.location_label}</p>
      <h1 id="at-bin-title" className="font-heading text-2xl font-bold">
        {match ? "Right bin" : "Not this bin"}
      </h1>
      <p>{message}</p>
      <div className="flex flex-col gap-3">
        {!match && <Button onClick={() => setBin(null)}>Scan another bin</Button>}
        <Button variant={match ? "primary" : "secondary"} onClick={onBack}>
          Back to the verdict
        </Button>
      </div>
    </section>
  );
}

// ---- Pieces ----------------------------------------------------------------

function BinCard({ verdict }: { verdict: VerdictResult }) {
  const bg = binColour(verdict);
  const fg = textOn(bg);
  return (
    <div className="overflow-hidden rounded-2xl shadow-md" style={{ background: bg, color: fg }}>
      <div aria-hidden className="mx-auto h-2 w-1/3 rounded-b-md bg-black/25" />
      <div className="space-y-2 p-5">
        <h1 id="verdict-title" className="font-heading text-3xl font-bold">
          {verdict.bin_label}
        </h1>
        <p>{verdict.reason}</p>
        <p className="text-sm font-bold">{confidenceLine(verdict)}</p>
      </div>
    </div>
  );
}

function WhyRows({
  verdict,
  analysis,
}: {
  verdict: VerdictResult;
  analysis: VisionAnalysis | null;
}) {
  const material = topMaterial(analysis);
  const condition = topCondition(analysis);
  return (
    <div className="rounded-2xl bg-white p-4">
      <h2 className="font-heading text-lg font-bold">Why</h2>
      <dl className="mt-2 divide-y divide-paper">
        {material && <Row term="Material" value={material} />}
        {condition && <Row term="Condition" value={condition} />}
        {verdict.questions_asked?.map((qa) => (
          <Row key={qa.question_id} term="Your answer" value={answerLabel(qa.answer)} />
        ))}
        <Row term="Rule" value={verdict.rule_trace.join(" → ") || verdict.rule_id} />
      </dl>
    </div>
  );
}

function Row({ term, value }: { term: string; value: string }) {
  return (
    <div className="flex gap-4 py-2">
      <dt className="w-28 shrink-0 text-muted">{term}</dt>
      <dd>{value}</dd>
    </div>
  );
}

function PhotoWithHeatmap({ url, analysis }: { url: string; analysis: VisionAnalysis | null }) {
  const heatmap = analysis?.heatmap_png_base64;
  const [showHeatmap, setShowHeatmap] = useState(true);
  if (!heatmap) return <Photo url={url} />;
  return (
    <figure className="space-y-2">
      <Photo
        url={showHeatmap ? `data:image/png;base64,${heatmap}` : url}
        alt={showHeatmap ? "Your photo with the deciding area highlighted" : "Your photo"}
      />
      <figcaption className="flex items-center justify-between gap-2 text-sm text-muted">
        <span>
          {showHeatmap ? "Highlighted: the area that decided the verdict." : "Your photo."}
        </span>
        <button
          type="button"
          className="min-h-11 rounded-lg px-3 font-bold text-bin-blue underline"
          onClick={() => setShowHeatmap((v) => !v)}
        >
          {showHeatmap ? "Show photo" : "Show highlight"}
        </button>
      </figcaption>
    </figure>
  );
}

function Photo({ url, alt = "Your photo" }: { url: string; alt?: string }) {
  return (
    // eslint-disable-next-line @next/next/no-img-element -- local blob/data URL, nothing to optimise
    <img src={url} alt={alt} className="max-h-80 w-full rounded-2xl bg-white object-contain" />
  );
}

function CaptureButton({ label, onPhoto }: { label: string; onPhoto: (file: File) => void }) {
  const input = useRef<HTMLInputElement>(null);
  return (
    <>
      <input
        ref={input}
        type="file"
        accept="image/*"
        capture="environment"
        className="sr-only"
        tabIndex={-1}
        aria-hidden
        onChange={(e) => {
          const file = e.target.files?.[0];
          e.target.value = "";
          if (file) onPhoto(file);
        }}
      />
      <Button onClick={() => input.current?.click()}>{label}</Button>
    </>
  );
}

function Button({
  variant = "primary",
  ...props
}: React.ButtonHTMLAttributes<HTMLButtonElement> & { variant?: "primary" | "secondary" }) {
  const styles =
    variant === "primary" ? "bg-bin-blue text-light" : "border-2 border-ink bg-white text-ink";
  return (
    <button
      type="button"
      {...props}
      className={`min-h-11 w-full rounded-xl px-4 py-3 font-bold disabled:opacity-50 ${styles}`}
    />
  );
}
