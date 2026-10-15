import { after } from "next/server";

import { callModel, configFromEnv } from "@/lib/model-api";
import type { AnswerRequest, AnswerResponse } from "@/lib/contracts";
import { currentUserId, saveVerdict } from "@/lib/supabase/server";

// Proxies to the model API; once the questions end in a verdict, records it on the scan.
// The scan id comes from the HMAC-signed decision state, which the model API has checked.
export async function POST(request: Request) {
  const body = (await request.json()) as AnswerRequest;
  const result = await callModel<AnswerResponse>(configFromEnv(), "/answer", body);
  if (!result.ok) return Response.json(result.error, { status: result.status });

  const verdict = result.data.verdict;
  const userId = verdict ? await currentUserId() : null;
  if (verdict && userId) {
    const scanId = body.state.scan_id;
    after(() =>
      saveVerdict(userId, scanId, verdict).catch((err: unknown) => console.error("[scans]", err)),
    );
  }
  return Response.json(result.data);
}
