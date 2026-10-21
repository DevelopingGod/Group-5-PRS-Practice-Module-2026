// Display helpers for the profile page. Pure, so they can be tested.

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
