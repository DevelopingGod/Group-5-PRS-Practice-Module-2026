import { describe, expect, it } from "vitest";

import { rankRows } from "@/lib/leaderboard";

describe("rankRows", () => {
  const rows = [
    { pseudonym: "Calm Heron 20", total_points: 3 },
    { pseudonym: "Tidy Otter 42", total_points: 7 },
    { pseudonym: "Bold Gecko 11", total_points: 3 },
    { pseudonym: "Keen Egret 64", total_points: 0 },
    { pseudonym: "Swift Sunbird 90", total_points: 1 },
  ];

  it("sorts by points, shares ranks on ties and drops zero-point users", () => {
    expect(rankRows(rows, null).map((r) => [r.rank, r.pseudonym])).toEqual([
      [1, "Tidy Otter 42"],
      [2, "Bold Gecko 11"],
      [2, "Calm Heron 20"],
      [4, "Swift Sunbird 90"],
    ]);
  });

  it("marks the signed-in user", () => {
    const ranked = rankRows(rows, "Calm Heron 20");
    expect(ranked.filter((r) => r.isMe).map((r) => r.pseudonym)).toEqual(["Calm Heron 20"]);
  });

  it("handles an empty board", () => {
    expect(rankRows([], "Tidy Otter 42")).toEqual([]);
  });
});
