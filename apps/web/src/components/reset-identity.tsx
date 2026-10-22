"use client";

// "Reset my identity": a simple privacy control on the profile page. Asks once more on the page
// (no browser dialog), then deletes the pseudonym, points and history and starts a new session.
import { useRouter } from "next/navigation";
import { useState } from "react";

import { useScan } from "@/components/scan-provider";
import { Button, Notice } from "@/components/ui";
import { startNewIdentity } from "@/lib/supabase/browser";

type Phase = "idle" | "confirm" | "working" | "error";

export function ResetIdentity() {
  const router = useRouter();
  const { restart } = useScan();
  const [phase, setPhase] = useState<Phase>("idle");

  async function reset() {
    setPhase("working");
    try {
      const res = await fetch("/api/reset-identity", { method: "POST" });
      if (!res.ok) throw new Error(`reset failed: ${res.status}`);
      restart(); // a scan in progress belonged to the old identity
      await startNewIdentity();
      setPhase("idle");
      router.refresh();
    } catch {
      setPhase("error");
    }
  }

  if (phase === "idle")
    return (
      <Button variant="secondary" onClick={() => setPhase("confirm")}>
        Reset my identity
      </Button>
    );

  return (
    <div className="card space-y-3 p-4" role="group" aria-labelledby="reset-title">
      <p id="reset-title" className="font-heading text-lg font-bold">
        Reset your identity?
      </p>
      <p>
        This deletes your pseudonym, your points and your scan history. Your next scan gets a new
        pseudonym. This cannot be undone.
      </p>
      {phase === "error" && (
        <Notice role="alert">We could not reset your identity. Please try again.</Notice>
      )}
      <div className="flex flex-col gap-3">
        <Button onClick={() => void reset()} disabled={phase === "working"}>
          {phase === "working" ? "Resetting…" : "Yes, delete and reset"}
        </Button>
        <Button variant="secondary" onClick={() => setPhase("idle")} disabled={phase === "working"}>
          Keep my identity
        </Button>
      </div>
    </div>
  );
}
