import { callModel, configFromEnv } from "@/lib/model-api";
import type { PredictRequest, PredictResponse } from "@/lib/contracts";

// Proxies to the model API so the HF token stays on the server.
// TODO (WS4): after a successful call, insert the scan into Supabase (scans table).
export async function POST(request: Request) {
  const body = (await request.json()) as PredictRequest;
  const result = await callModel<PredictResponse>(configFromEnv(), "/predict", body);
  return result.ok
    ? Response.json(result.data)
    : Response.json(result.error, { status: result.status });
}
