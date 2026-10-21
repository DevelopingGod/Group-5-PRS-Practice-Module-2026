"""Verify a canonical WS2 validation manifest before any ensemble comparison.

uv run python scripts/ws2_check_validation_manifest.py
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def check(predictions: Path, manifest: Path) -> dict[str, object]:
    with predictions.open(newline="", encoding="utf-8") as handle:
        expected = {row["path"]: row for row in csv.DictReader(handle) if row["split"] == "val"}
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(expected) or len({r["path"] for r in rows}) != len(rows):
        raise ValueError("Manifest must contain each validation image exactly once")
    if set(expected) != {r["path"] for r in rows}:
        raise ValueError("Manifest paths do not match original validation paths")
    groups: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        original = expected[row["path"]]
        for key in ("source", "group", "material"):
            if row[key] != original[key]:
                raise ValueError(f"Metadata mismatch for {row['path']}: {key}")
        if row["partition"] not in {"calibration", "meta_train", "diagnostic"}:
            raise ValueError("Unknown partition")
        groups[(row["source"], row["group"])].add(row["partition"])
    if any(len(partitions) != 1 for partitions in groups.values()):
        raise ValueError("Physical-item group crosses partitions")
    counts = Counter(row["partition"] for row in rows)
    local_counts = Counter(row["partition"] for row in rows if row["source"] == "local")
    if len(counts) != 3 or any(local_counts[p] < 20 for p in counts):
        raise ValueError("Missing partition or insufficient local examples")
    return {
        "status": "passed",
        "n_validation": len(rows),
        "n_groups": len(groups),
        "partition_counts": dict(sorted(counts.items())),
        "local_counts": dict(sorted(local_counts.items())),
        "group_overlap": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions", type=Path, default=Path("data/ws2/v0.2-local-data/predictions.csv")
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/ws2/validation_partitions_v1.csv")
    )
    args = parser.parse_args()
    print(json.dumps(check(args.predictions, args.manifest), indent=2))


if __name__ == "__main__":
    main()
