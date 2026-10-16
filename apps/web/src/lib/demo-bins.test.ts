import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { describe, expect, it } from "vitest";

import { DEMO_BINS, encodeBinQr } from "@/lib/demo-bins";
import { checkBin, parseBinQr } from "@/lib/qr";

const repo = resolve(__dirname, "../../../..");
const read = (path: string) => readFileSync(resolve(repo, path), "utf-8");

/** Bin keys declared in a policy file (`- key: x` under `bins:`). */
function policyBinKeys(site: string): string[] {
  return [...read(`policies/${site}.yaml`).matchAll(/^\s+- key: (\S+)/gm)].map((m) => m[1]!);
}

describe("DEMO_BINS", () => {
  it("encode to contract-valid QR payloads", () => {
    for (const bin of DEMO_BINS) {
      expect(parseBinQr(encodeBinQr(bin))).toEqual({ ok: true, payload: bin });
    }
  });

  it("have unique ids and name real bins in their site policy", () => {
    expect(new Set(DEMO_BINS.map((b) => b.bin_id)).size).toBe(DEMO_BINS.length);
    for (const bin of DEMO_BINS) {
      expect(policyBinKeys(bin.site_policy)).toContain(bin.bin_key);
    }
  });

  it("match the seed migration", () => {
    const sql = read("supabase/migrations/20261016000000_demo_bins.sql");
    const rows = [...sql.matchAll(/\('([^']+)', '([^']+)', '([^']+)', '([^']+)'\)/g)].map(
      ([, bin_id, site_policy, bin_key, location_label]) => ({
        v: 1,
        bin_id,
        site_policy,
        bin_key,
        location_label,
      }),
    );
    expect(rows).toEqual(DEMO_BINS);
  });
});

describe("checkBin", () => {
  const blue = DEMO_BINS[0]!;
  const general = DEMO_BINS[1]!;
  const campusGeneral = DEMO_BINS[5]!;

  it("matches the recommended bin", () => {
    expect(checkBin({ bin_key: "blue_bin" }, "hdb_default", blue).kind).toBe("match");
  });

  it("flags the wrong bin, another site's bin and no recommendation", () => {
    expect(checkBin({ bin_key: "blue_bin" }, "hdb_default", general).kind).toBe("wrong_bin");
    expect(checkBin({ bin_key: "general_waste" }, "hdb_default", campusGeneral).kind).toBe(
      "other_site",
    );
    expect(checkBin({ bin_key: null }, "hdb_default", blue).kind).toBe("no_bin");
  });
});
