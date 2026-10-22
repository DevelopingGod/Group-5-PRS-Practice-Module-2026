// Display helpers for the profile page. Pure, so they can be tested.
import type { Channel, Material } from "@/lib/contracts";
import { materialLabel } from "@/lib/scan-flow";

const number = new Intl.NumberFormat("en-SG");

/** "1 point", "12 points", "1,204 photos". */
export function countText(n: number, one: string, many: string): string {
  return `${number.format(n)} ${n === 1 ? one : many}`;
}

export type TodayPoints = { points: number; cap: number; left: number; percent: number };

/** Today's points against the daily cap; the total shown never passes the cap. */
export function todayPoints(raw: number, cap: number): TodayPoints {
  const points = Math.min(Math.max(0, raw), cap);
  return {
    points,
    cap,
    left: cap - points,
    percent: cap > 0 ? Math.round((points / cap) * 100) : 0,
  };
}

/** One finished scan as stored in `scans` (results only: photos are never kept). */
export type ScanSummary = {
  id: string;
  material: Material | null;
  channel: Channel | null;
  bin_key: string | null;
  created_at: string;
};

export type HistoryItem = {
  id: string;
  material: string;
  bin: string;
  date: string;
  verified: boolean;
};

// Bin names from the HDB policy (policies/hdb_default.yaml), which the app scans with.
const BIN_LABELS: Record<string, string> = {
  blue_bin: "Blue recycling bin",
  general_waste: "General waste",
  e_waste_point: "E-waste collection point",
  donation: "Donate or use another channel",
};

const when = new Intl.DateTimeFormat("en-SG", {
  timeZone: "Asia/Singapore",
  day: "numeric",
  month: "short",
  hour: "numeric",
  minute: "2-digit",
});

/** The profile's history list, newest first as given; `verified` holds scan ids with a passed check. */
export function historyItems(scans: ScanSummary[], verified: ReadonlySet<string>): HistoryItem[] {
  return scans.map((scan) => ({
    id: scan.id,
    material: scan.material ? materialLabel(scan.material) : "Unknown item",
    bin:
      scan.channel === "abstain"
        ? "No verdict"
        : ((scan.bin_key && BIN_LABELS[scan.bin_key]) ?? scan.bin_key ?? "No verdict"),
    date: when.format(new Date(scan.created_at)),
    verified: verified.has(scan.id),
  }));
}
