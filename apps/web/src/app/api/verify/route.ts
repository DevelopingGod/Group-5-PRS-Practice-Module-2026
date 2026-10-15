import { callModel, configFromEnv } from "@/lib/model-api";
import type { ErrorResponse, VerifyResponse } from "@/lib/contracts";
import { parseBinQr } from "@/lib/qr";
import {
  hasDisposal,
  insertDisposal,
  loadBin,
  loadScan,
  pointsSince,
  recentPhashes,
} from "@/lib/supabase/disposals";
import { currentUserId } from "@/lib/supabase/server";
import {
  binIsRegistered,
  buildVerifyRequest,
  pointsFor,
  startOfSingaporeDay,
  type VerifyRouteRequest,
  type VerifyRouteResponse,
} from "@/lib/verify-record";

// Verified disposal (context.md Section 5.5): loads the original scan from Supabase, asks the
// model API to compare the new photo, and records the reward. The photo is never stored: it is
// passed straight to the model and dropped, and only its perceptual hash is kept.
export async function POST(request: Request) {
  try {
    return await verify(request);
  } catch (err) {
    console.error("[verify]", err);
    return fail(500, "internal", "We could not check your disposal. Please try again.");
  }
}

async function verify(request: Request): Promise<Response> {
  const body = (await request.json()) as VerifyRouteRequest;
  const qr = parseBinQr(JSON.stringify(body.qr ?? null));
  if (!qr.ok) return fail(400, "invalid_state", "That is not a Bin Buddy bin code.");

  const userId = await currentUserId();
  if (!userId)
    return fail(401, "invalid_state", "Please reload Bin Buddy and scan the item again.");

  const scan = await loadScan(userId, body.scan_id);
  if (!scan) return fail(404, "invalid_state", "We could not find that scan. Please scan again.");
  if (await hasDisposal(scan.id)) {
    return fail(409, "invalid_state", "You have already been rewarded for this item.");
  }
  if (!binIsRegistered(qr.payload, await loadBin(qr.payload.bin_id))) {
    return fail(400, "invalid_state", "This bin is not a registered Bin Buddy demonstration bin.");
  }

  const now = new Date();
  const built = buildVerifyRequest(scan, body, await recentPhashes(userId), now);
  if (!built.ok) return fail(400, "invalid_state", built.message);

  const result = await callModel<VerifyResponse>(configFromEnv(), "/verify", built.request);
  if (!result.ok) return Response.json(result.error, { status: result.status });

  const verdict = result.data;
  let points = 0;
  if (verdict.passed) {
    points = pointsFor(await pointsSince(userId, startOfSingaporeDay(now)));
    const inserted = await insertDisposal({
      scan_id: scan.id,
      user_id: userId,
      bin_id: qr.payload.bin_id,
      similarity: verdict.similarity,
      passed: true,
      points,
      phash: verdict.phash,
    });
    if (!inserted)
      return fail(409, "invalid_state", "You have already been rewarded for this item.");
  }
  return Response.json({ ...verdict, points_awarded: points } satisfies VerifyRouteResponse);
}

function fail(status: number, code: ErrorResponse["code"], message: string) {
  return Response.json({ code, message } satisfies ErrorResponse, { status });
}
