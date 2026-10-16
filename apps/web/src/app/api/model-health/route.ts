import { callModel, configFromEnv } from "@/lib/model-api";
import type { HealthResponse } from "@/lib/contracts";

// Longer than MODEL_TIMEOUT_MS so the platform does not cut off a cold start.
export const maxDuration = 120;

// Call this before a demo to wake the model API (it sleeps after 15 minutes idle).
export async function GET() {
  const result = await callModel<HealthResponse>(configFromEnv(), "/health");
  return result.ok
    ? Response.json(result.data)
    : Response.json(result.error, { status: result.status });
}
