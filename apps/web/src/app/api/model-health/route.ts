import { callModel, configFromEnv } from "@/lib/model-api";
import type { HealthResponse } from "@/lib/contracts";

// Call this before a demo to wake the Hugging Face Space.
export async function GET() {
  const result = await callModel<HealthResponse>(configFromEnv(), "/health");
  return result.ok
    ? Response.json(result.data)
    : Response.json(result.error, { status: result.status });
}
