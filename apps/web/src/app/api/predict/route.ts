import { callModel, configFromEnv } from "@/lib/model-api";
import type { PredictRequest, PredictResponse } from "@/lib/contracts";
import { scanRowFromPredict } from "@/lib/scan-record";
import { currentUserId, saveScan } from "@/lib/supabase/server";

// Proxies to the model API so the HF token stays on the server, then saves the scan.
// The insert is awaited so the row exists before /answer updates it; a failure is logged
// and never blocks the verdict.
export async function POST(request: Request) {
  const body = (await request.json()) as PredictRequest;
  const result = await callModel<PredictResponse>(configFromEnv(), "/predict", body);
  if (!result.ok) return Response.json(result.error, { status: result.status });

  const userId = await currentUserId();
  const row = userId ? scanRowFromPredict(userId, body.site_policy, result.data) : null;
  if (row) await saveScan(row).catch((err: unknown) => console.error("[scans]", err));
  return Response.json(result.data);
}
