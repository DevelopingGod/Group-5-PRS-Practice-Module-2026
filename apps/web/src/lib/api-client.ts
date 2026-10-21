// Browser-side calls to our own route handlers (never to the model API directly).
import type {
  AnswerRequest,
  AnswerResponse,
  ErrorResponse,
  PredictRequest,
  PredictResponse,
} from "@/lib/contracts";
import type { VerifyRouteRequest, VerifyRouteResponse } from "@/lib/verify-record";

export type ApiResult<T> = { ok: true; data: T } | { ok: false; error: ErrorResponse };

const OFFLINE: ApiResult<never> = {
  ok: false,
  error: { code: "model_unavailable", message: "No connection. Check your internet." },
};

function toResult<T>(ok: boolean, json: unknown): ApiResult<T> {
  if (ok) return { ok: true, data: json as T };
  const error =
    json && typeof json === "object" && "code" in json && "message" in json
      ? (json as ErrorResponse)
      : { code: "internal" as const, message: "Something went wrong. Please try again." };
  return { ok: false, error };
}

async function post<T>(path: string, body: unknown): Promise<ApiResult<T>> {
  try {
    const res = await fetch(path, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    });
    const json: unknown = await res.json().catch(() => null);
    return toResult<T>(res.ok, json);
  } catch {
    return OFFLINE;
  }
}

/**
 * /api/predict through XMLHttpRequest, because fetch cannot report upload progress.
 * `onUpload` receives the fraction of the photo sent so far (0 to 1).
 */
export function predict(
  body: PredictRequest,
  onUpload: (fraction: number) => void = () => {},
): Promise<ApiResult<PredictResponse>> {
  return new Promise((resolve) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/predict");
    xhr.setRequestHeader("content-type", "application/json");
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable && e.total > 0) onUpload(e.loaded / e.total);
    };
    xhr.upload.onload = () => onUpload(1);
    xhr.onload = () => {
      onUpload(1);
      let json: unknown = null;
      try {
        json = JSON.parse(xhr.responseText);
      } catch {
        // not JSON: handled as an internal error below
      }
      resolve(toResult<PredictResponse>(xhr.status >= 200 && xhr.status < 300, json));
    };
    xhr.onerror = xhr.onabort = xhr.ontimeout = () => resolve(OFFLINE);
    xhr.send(JSON.stringify(body));
  });
}

export const answer = (body: AnswerRequest) => post<AnswerResponse>("/api/answer", body);
export const verify = (body: VerifyRouteRequest) => post<VerifyRouteResponse>("/api/verify", body);
