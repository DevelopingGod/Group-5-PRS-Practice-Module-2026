import { callModel, configFromEnv } from "@/lib/model-api";
import type { AnswerRequest, AnswerResponse } from "@/lib/contracts";

export async function POST(request: Request) {
  const body = (await request.json()) as AnswerRequest;
  const result = await callModel<AnswerResponse>(configFromEnv(), "/answer", body);
  return result.ok
    ? Response.json(result.data)
    : Response.json(result.error, { status: result.status });
}
