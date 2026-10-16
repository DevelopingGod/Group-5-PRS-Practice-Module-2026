"use client";

// Live camera QR scanner for bin codes. Uses ZXing rather than BarcodeDetector,
// which Safari does not support.
import { BrowserQRCodeReader, type IScannerControls } from "@zxing/browser";
import { useEffect, useRef, useState } from "react";

import { ScanLine } from "@/components/ui";
import type { BinQrPayload } from "@/lib/contracts";
import { parseBinQr } from "@/lib/qr";

export function QrScanner({ onBin }: { onBin: (bin: BinQrPayload) => void }) {
  const video = useRef<HTMLVideoElement>(null);
  const onBinRef = useRef(onBin);
  const [problem, setProblem] = useState<string | null>(null);

  useEffect(() => {
    onBinRef.current = onBin;
  }, [onBin]);

  useEffect(() => {
    const el = video.current;
    if (!el) return;
    let controls: IScannerControls | null = null;
    let stopped = false;
    let lastBad = "";

    new BrowserQRCodeReader()
      .decodeFromConstraints({ video: { facingMode: { ideal: "environment" } } }, el, (result) => {
        if (!result || stopped) return;
        const text = result.getText();
        const parsed = parseBinQr(text);
        if (parsed.ok) {
          stopped = true;
          controls?.stop();
          onBinRef.current(parsed.payload);
        } else if (text !== lastBad) {
          lastBad = text;
          setProblem(
            "That is not a Bin Buddy bin code. Look for the Bin Buddy sticker on the bin.",
          );
        }
      })
      .then((c) => {
        controls = c;
        if (stopped) c.stop();
      })
      .catch((err: unknown) => {
        const denied = err instanceof Error && err.name === "NotAllowedError";
        setProblem(
          denied
            ? "Bin Buddy needs your camera to read the bin's code. Allow camera access and try again."
            : "Could not start the camera on this device.",
        );
      });

    return () => {
      stopped = true;
      controls?.stop();
    };
  }, []);

  return (
    <div className="space-y-3">
      <div className="viewfinder relative overflow-hidden rounded-[1.75rem] bg-ink text-sticker shadow-[var(--shadow-lift)]">
        <video
          ref={video}
          muted
          playsInline
          aria-label="Camera view for scanning the bin's QR code"
          className="aspect-square w-full object-cover"
        />
        <span aria-hidden className="corners" />
        <ScanLine inset="inset-x-8" />
      </div>
      {problem && (
        <p role="alert" className="rounded-lg bg-sticker p-3 text-ink">
          {problem}
        </p>
      )}
    </div>
  );
}
