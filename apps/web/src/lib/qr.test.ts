import { describe, expect, it } from "vitest";

import { parseBinQr } from "@/lib/qr";

const valid = {
  v: 1,
  bin_id: "demo-hdb-01",
  site_policy: "hdb_default",
  bin_key: "blue_bin",
  location_label: "Demo block",
};

describe("parseBinQr", () => {
  it("accepts a contract-valid payload", () => {
    expect(parseBinQr(JSON.stringify(valid))).toEqual({ ok: true, payload: valid });
  });

  it("rejects non-JSON text", () => {
    expect(parseBinQr("https://example.com").ok).toBe(false);
  });

  it("rejects unknown fields and bad ids", () => {
    expect(parseBinQr(JSON.stringify({ ...valid, extra: 1 })).ok).toBe(false);
    expect(parseBinQr(JSON.stringify({ ...valid, bin_id: "Demo Bin" })).ok).toBe(false);
  });
});
