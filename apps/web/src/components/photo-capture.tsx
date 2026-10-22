"use client";

// Taking or choosing the item photo. Phones use the native camera (capture="environment") or
// the photo library. Laptops get a live webcam preview through getUserMedia; if the browser
// refuses the camera or has none, the user picks a file instead, with a one-line reason.
import { useEffect, useRef, useState } from "react";

import { CameraIcon, GalleryIcon, SwitchCameraIcon } from "@/components/icons";
import { Button, Notice, PhotoInput } from "@/components/ui";

type Mode = "idle" | "webcam" | "denied" | "no_camera";

/** A laptop or desktop with a camera API: show a live preview rather than a file picker. */
function prefersLivePreview(): boolean {
  return (
    typeof navigator.mediaDevices?.getUserMedia === "function" &&
    !window.matchMedia("(pointer: coarse)").matches
  );
}

/**
 * `take()` opens the camera, `choose()` the photo library. Render `inputs` once, `webcam` where
 * the live preview should appear (null when closed) and `notice` near the buttons.
 */
export function usePhotoCapture(onPhoto: (file: File) => void) {
  const camera = useRef<HTMLInputElement>(null);
  const gallery = useRef<HTMLInputElement>(null);
  const [mode, setMode] = useState<Mode>("idle");

  function take() {
    // After a refusal the file picker stands in for the camera.
    if (mode === "idle" && prefersLivePreview()) setMode("webcam");
    else camera.current?.click();
  }

  const inputs = (
    <>
      <PhotoInput ref={camera} camera onPhoto={onPhoto} />
      <PhotoInput ref={gallery} onPhoto={onPhoto} />
    </>
  );
  const webcam =
    mode === "webcam" ? (
      <WebcamCapture
        onPhoto={(file) => {
          setMode("idle");
          onPhoto(file);
        }}
        onCancel={() => setMode("idle")}
        onUnavailable={(denied) => setMode(denied ? "denied" : "no_camera")}
      />
    ) : null;
  const notice =
    mode === "denied" || mode === "no_camera" ? (
      <Notice role="status">
        {mode === "denied"
          ? "Camera access was not allowed, so choose a photo from your files instead."
          : "No camera could be started, so choose a photo from your files instead."}
      </Notice>
    ) : null;

  return { take, choose: () => gallery.current?.click(), inputs, webcam, notice };
}

/**
 * "Take photo" and, with `gallery`, "Choose from gallery". The disposal check leaves the gallery
 * out: the photo has to be taken at the bin.
 */
export function CaptureButton({
  label,
  onPhoto,
  gallery = false,
}: {
  label: string;
  onPhoto: (file: File) => void;
  gallery?: boolean;
}) {
  const capture = usePhotoCapture(onPhoto);
  if (capture.webcam) return capture.webcam;
  return (
    <div className="flex flex-col gap-3">
      {capture.inputs}
      {capture.notice}
      <Button onClick={capture.take}>
        <CameraIcon className="size-5" />
        {label}
      </Button>
      {gallery && <GalleryButton onClick={capture.choose} />}
    </div>
  );
}

export function GalleryButton({ onClick }: { onClick: () => void }) {
  return (
    <Button variant="secondary" onClick={onClick}>
      <GalleryIcon className="size-5" />
      Choose from gallery
    </Button>
  );
}

function stopAll(stream: MediaStream) {
  for (const track of stream.getTracks()) track.stop();
}

/** Live preview from the back camera (or the webcam), with "Switch camera" when there are two. */
function WebcamCapture({
  onPhoto,
  onCancel,
  onUnavailable,
}: {
  onPhoto: (file: File) => void;
  onCancel: () => void;
  onUnavailable: (denied: boolean) => void;
}) {
  const video = useRef<HTMLVideoElement>(null);
  const unavailable = useRef(onUnavailable);
  // null: let the browser choose, preferring a back camera.
  const [deviceId, setDeviceId] = useState<string | null>(null);
  const [cameras, setCameras] = useState<string[]>([]);
  const [active, setActive] = useState<string | null>(null);
  const [playing, setPlaying] = useState(false);

  useEffect(() => {
    unavailable.current = onUnavailable;
  }, [onUnavailable]);

  useEffect(() => {
    const el = video.current;
    if (!el) return;
    let stream: MediaStream | null = null;
    let cancelled = false;
    navigator.mediaDevices
      .getUserMedia({
        audio: false,
        video: deviceId
          ? { deviceId: { exact: deviceId } }
          : { facingMode: { ideal: "environment" }, width: { ideal: 1280 } },
      })
      .then(async (s) => {
        if (cancelled) return stopAll(s);
        stream = s;
        el.srcObject = s;
        await el.play().catch(() => undefined);
        // Cameras are only listed in full once access is allowed.
        const devices = await navigator.mediaDevices.enumerateDevices();
        if (cancelled) return;
        setCameras(devices.filter((d) => d.kind === "videoinput").map((d) => d.deviceId));
        setActive(s.getVideoTracks()[0]?.getSettings().deviceId ?? null);
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        const name = err instanceof Error ? err.name : "";
        unavailable.current(name === "NotAllowedError" || name === "SecurityError");
      });
    return () => {
      cancelled = true;
      if (stream) stopAll(stream);
      el.srcObject = null;
    };
  }, [deviceId]);

  function switchCamera() {
    const next = cameras[(cameras.indexOf(active ?? "") + 1) % cameras.length];
    if (!next) return;
    setPlaying(false);
    setDeviceId(next);
  }

  function capture() {
    const el = video.current;
    if (!el || el.videoWidth === 0) return;
    const canvas = document.createElement("canvas");
    canvas.width = el.videoWidth;
    canvas.height = el.videoHeight;
    canvas.getContext("2d")?.drawImage(el, 0, 0);
    canvas.toBlob(
      (blob) => {
        if (blob) onPhoto(new File([blob], "camera.jpg", { type: "image/jpeg" }));
      },
      "image/jpeg",
      0.92,
    );
  }

  return (
    <div className="space-y-3">
      {/* Square, like the centre of the photo the model reads. */}
      <div className="viewfinder relative overflow-hidden rounded-[1.75rem] bg-ink text-sticker shadow-[var(--shadow-lift)]">
        <video
          ref={video}
          muted
          playsInline
          aria-label="Camera preview"
          onPlaying={() => setPlaying(true)}
          className="aspect-square w-full object-cover"
        />
        <span aria-hidden className="corners" />
      </div>
      <Button onClick={capture} disabled={!playing}>
        <CameraIcon className="size-5" />
        Take the photo
      </Button>
      <div className="flex gap-3">
        {cameras.length > 1 && (
          <Button variant="secondary" onClick={switchCamera}>
            <SwitchCameraIcon className="size-5" />
            Switch camera
          </Button>
        )}
        <Button variant="secondary" onClick={onCancel}>
          Cancel
        </Button>
      </div>
    </div>
  );
}
