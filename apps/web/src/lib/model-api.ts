// Server-side client for the model API on the Hugging Face Space.
// The HF token must never reach the browser: only import this from route handlers.
import type { ErrorResponse } from "@/lib/contracts";

export type ModelApiConfig = {
  baseUrl: string;
  token?: string;
  timeoutMs?: number;
  fetchImpl?: typeof fetch;
};

export type ModelResult<T> =
  { ok: true; data: T } | { ok: false; status: number; error: ErrorResponse };

export function configFromEnv(): ModelApiConfig {
  const baseUrl = process.env.HF_SPACE_URL;
  if (!baseUrl) throw new Error("HF_SPACE_URL is not set");
  return { baseUrl, token: process.env.HF_TOKEN, timeoutMs: 60_000 };
}

/** POST or GET a model API route. Free Spaces can take tens of seconds to wake up. */
export async function callModel<T>(
  config: ModelApiConfig,
  path: "/health" | "/version" | "/predict" | "/answer" | "/verify",
  body?: unknown,
): Promise<ModelResult<T>> {
  const fetchImpl = config.fetchImpl ?? fetch;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), config.timeoutMs ?? 60_000);
  try {
    const res = await fetchImpl(new URL(path, config.baseUrl), {
      method: body === undefined ? "GET" : "POST",
      headers: {
        "content-type": "application/json",
        ...(config.token ? { authorization: `Bearer ${config.token}` } : {}),
      },
      body: body === undefined ? undefined : JSON.stringify(body),
      signal: controller.signal,
    });
    const json: unknown = await res.json().catch(() => null);
    if (res.ok) return { ok: true, data: json as T };
    const error = isErrorResponse(json)
      ? json
      : { code: "internal" as const, message: `Model API returned ${res.status}` };
    return { ok: false, status: res.status, error };
  } catch (err) {
    const waking = err instanceof Error && err.name === "AbortError";
    return {
      ok: false,
      status: 503,
      error: {
        code: "model_unavailable",
        message: waking
          ? "The model is waking up. Please try again in a moment."
          : "Could not reach the model API.",
      },
    };
  } finally {
    clearTimeout(timer);
  }
}

function isErrorResponse(value: unknown): value is ErrorResponse {
  return (
    typeof value === "object" &&
    value !== null &&
    "code" in value &&
    "message" in value &&
    typeof (value as { message: unknown }).message === "string"
  );
}
