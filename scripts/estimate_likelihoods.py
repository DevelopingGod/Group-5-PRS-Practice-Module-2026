"""Count answer likelihoods P(answer | class) from the Q1-Q3 answers in data/labels.csv (WS3).

Uses the train and val rows only (never test or field). Each row of the tables in
services/inference/app/likelihoods.py is counted with Laplace smoothing,
P(answer | class) = (count + 1) / (n + number of answers), then floored at LIKELIHOOD_FLOOR.
Rows with fewer than --min-count answers keep their hand-set values. "na" answers (no food or
drink involved) are skipped: they say nothing about the class.

    uv run python scripts/estimate_likelihoods.py            # compare counted vs hand-set
    uv run python scripts/estimate_likelihoods.py --write    # write likelihoods_counted.json

The model API loads likelihoods_counted.json at start-up; commit it with the PR that adds it.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

# The ml helpers and the model API are not installed as packages; put them on the path like the
# tests do.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "services" / "inference"))

from app import likelihoods
from binbuddy_contracts.dataset import LabelRow
from binbuddy_contracts.enums import Condition, Split
from binbuddy_ml.labels import load_labels

TRAINING_SPLITS = (Split.TRAIN, Split.VAL)


def observations(rows: Iterable[LabelRow]) -> dict[str, list[str]]:
    """Answers grouped by likelihood row name (the keys of likelihoods.all_rows())."""
    seen: dict[str, list[str]] = {}
    for r in rows:
        seen.setdefault(f"q1/{r.material.value}", []).append(r.q1_recycling_symbol.value)
        if r.q2_cleaned != "na":
            seen.setdefault(f"q2/{r.condition.value}", []).append(str(r.q2_cleaned))
        if r.q3_sitting_time != "na":
            if r.condition is Condition.CLEAN:
                key = "q3/clean"
            elif r.material in likelihoods.PERISHABLE_CONTENTS:
                key = "q3/residue_perishable"
            else:
                key = "q3/residue_dry"
            seen.setdefault(key, []).append(str(r.q3_sitting_time))
    return seen


def smoothed(answers: Sequence[str], options: Sequence[str]) -> dict[str, float]:
    """Laplace-smoothed frequencies, rounded, with every answer at least LIKELIHOOD_FLOOR.

    Renormalising after flooring would push small answers back under the floor, so the most
    common answer pays instead: it absorbs the amount added by flooring and the rounding remainder.
    """
    n, k = len(answers), len(options)
    row = {
        a: max(round((answers.count(a) + 1) / (n + k), 4), likelihoods.LIKELIHOOD_FLOOR)
        for a in options
    }
    top = max(row, key=row.__getitem__)
    row[top] = round(row[top] + 1.0 - sum(row.values()), 4)
    return row


def estimate(rows: Iterable[LabelRow], min_count: int) -> dict[str, dict[str, float]]:
    hand_set = likelihoods.all_rows()
    return {
        name: smoothed(answers, list(hand_set[name]))
        for name, answers in sorted(observations(rows).items())
        if len(answers) >= min_count
    }


def report(counted: dict[str, dict[str, float]], seen: dict[str, list[str]]) -> str:
    lines = []
    for name, hand in likelihoods.all_rows().items():
        n = len(seen.get(name, []))
        new = counted.get(name)
        fmt = ", ".join
        lines.append(
            f"{name:28s} n={n:4d}  hand-set: {fmt(f'{a}={p:.2f}' for a, p in hand.items())}"
        )
        if new:
            lines.append(f"{'':35s}counted:  {fmt(f'{a}={p:.2f}' for a, p in new.items())}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--min-count", type=int, default=5, help="answers needed to replace a row")
    ap.add_argument("--write", action="store_true", help="write likelihoods_counted.json")
    args = ap.parse_args(argv)

    rows = [r for r in load_labels() if r.split in TRAINING_SPLITS]
    seen = observations(rows)
    counted = estimate(rows, args.min_count)
    print(f"{len(rows)} train/val photos; {len(counted)} rows have >= {args.min_count} answers")
    print(report(counted, seen))
    if args.write:
        if not counted:
            print("nothing to write: no row has enough answers yet")
            return 1
        for name, row in counted.items():
            likelihoods.check_row(name, row, likelihoods.all_rows()[name])
        payload = {"source": "data/labels.csv (train + val)", "rows": counted}
        likelihoods.COUNTED_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {likelihoods.COUNTED_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
