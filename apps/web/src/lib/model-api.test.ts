import { describe, expect, it } from "vitest";

import { callModel } from "@/lib/model-api";

const json = (status: number, body: unknown): Response =>
  new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });

describe("callModel", () => {
  it("returns data and sends the token server-side", async () => {
    let auth: string | null = null;
    const fetchImpl: typeof fetch = async (_url, init) => {
      auth = new Headers(init?.headers).get("authorization");
      return json(200, { status: "ok", model_loaded: true });
    };
    const res = await callModel(
      { baseUrl: "https://x.hf.space", token: "t", fetchImpl },
      "/health",
    );
    expect(res).toEqual({ ok: true, data: { status: "ok", model_loaded: true } });
    expect(auth).toBe("Bearer t");
  });

  it("passes contract errors through", async () => {
    const fetchImpl: typeof fetch = async () =>
      json(400, { code: "unknown_policy", message: "unknown site policy" });
    const res = await callModel({ baseUrl: "https://x.hf.space", fetchImpl }, "/predict", {});
    expect(res).toEqual({
      ok: false,
      status: 400,
      error: { code: "unknown_policy", message: "unknown site policy" },
    });
  });

  it("reports a sleeping model as model_unavailable", async () => {
    const fetchImpl: typeof fetch = async () => {
      throw Object.assign(new Error("aborted"), { name: "AbortError" });
    };
    const res = await callModel({ baseUrl: "https://x.hf.space", fetchImpl }, "/health");
    expect(res.ok).toBe(false);
    if (!res.ok) expect(res.error.code).toBe("model_unavailable");
  });
});
