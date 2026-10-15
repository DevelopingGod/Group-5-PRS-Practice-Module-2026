// Browser-side calls to our own route handlers (never to the Space directly).
import type {
  AnswerRequest,
  AnswerResponse,
  ErrorResponse,
  PredictRequest,
  PredictResponse,
} from "@/lib/contracts";

export type ApiResult<T> = { ok: true; data: T } | { ok: false; error: ErrorResponse };

async function post<T>(path: string, body: unknown): Promise<ApiResult<T>> {
  try {
    const res = await fetch(path, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    });
    const json: unknown = await res.json().catch(() => null);
    if (res.ok) return { ok: true, data: json as T };
    const error =
      json && typeof json === "object" && "code" in json && "message" in json
        ? (json as ErrorResponse)
        : { code: "internal" as const, message: "Something went wrong. Please try again." };
    return { ok: false, error };
  } catch {
    return {
      ok: false,
      error: { code: "model_unavailable", message: "No connection. Check your internet." },
    };
  }
}

export const predict = (body: PredictRequest) => post<PredictResponse>("/api/predict", body);
export const answer = (body: AnswerRequest) => post<AnswerResponse>("/api/answer", body);
