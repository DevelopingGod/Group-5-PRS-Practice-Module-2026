// The demonstration bins that carry Bin Buddy QR codes (context.md Section 5.5, 7).
// QR codes go on demonstration bins only, never on public HDB bins.
// Keep in sync with supabase/migrations/20261016000000_demo_bins.sql (checked by a test).
import type { BinQrPayload } from "@/lib/contracts";

export const DEMO_BINS: readonly BinQrPayload[] = [
  {
    v: 1,
    bin_id: "demo-hdb-01",
    site_policy: "hdb_default",
    bin_key: "blue_bin",
    location_label: "Demo blue recycling bin",
  },
  {
    v: 1,
    bin_id: "demo-hdb-02",
    site_policy: "hdb_default",
    bin_key: "general_waste",
    location_label: "Demo general waste bin",
  },
  {
    v: 1,
    bin_id: "demo-campus-01",
    site_policy: "campus_example",
    bin_key: "paper_bin",
    location_label: "Demo campus paper bin",
  },
  {
    v: 1,
    bin_id: "demo-campus-02",
    site_policy: "campus_example",
    bin_key: "plastic_bin",
    location_label: "Demo campus plastic bin",
  },
  {
    v: 1,
    bin_id: "demo-campus-03",
    site_policy: "campus_example",
    bin_key: "cans_glass_bin",
    location_label: "Demo campus cans and glass bin",
  },
  {
    v: 1,
    bin_id: "demo-campus-04",
    site_policy: "campus_example",
    bin_key: "general_waste",
    location_label: "Demo campus general waste bin",
  },
];

/** The exact text to encode in a bin's QR code. Fixed key order keeps codes stable. */
export function encodeBinQr(bin: BinQrPayload): string {
  const { v, bin_id, site_policy, bin_key, location_label } = bin;
  return JSON.stringify({ v, bin_id, site_policy, bin_key, location_label });
}
