"""Create the canonical WS2 validation partition manifest.

Deterministic by physical-item group, independent of CSV row order.
Source/material stratification is approximate because an item group may
contain several photos and labels. No test rows are included.

uv run python scripts/ws2_make_validation_manifest.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

PARTITIONS = ("calibration", "meta_train", "diagnostic")
TARGET = (0.4, 0.3, 0.3)
SEED = 42


def group_hash(source: str, group: str) -> str:
    return hashlib.sha256(f"ws2-v1:{SEED}:{source}:{group}".encode()).hexdigest()


def make_manifest(predictions: Path, output: Path) -> dict[str, object]:
    with predictions.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"path", "source", "split", "material", "group"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Missing required prediction columns")
        rows = [row for row in reader if row["split"] == "val"]
    if not rows or len({row["path"] for row in rows}) != len(rows):
        raise ValueError("Missing validation rows or duplicate paths")
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if not row["group"] or not row["source"]:
            raise ValueError("Missing source/group identifier")
        groups[(row["source"], row["group"])].append(row)

    # Greedy group allocation within source + dominant material strata.
    # Sorted cryptographic hashes ensure input row ordering never changes assignments.
    strata: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    for key, members in groups.items():
        counts = Counter(row["material"] for row in members)
        dominant = sorted(counts, key=lambda name: (-counts[name], name))[0]
        strata[(key[0], dominant)].append(key)
    assignment: dict[tuple[str, str], str] = {}
    for keys in strata.values():
        keys.sort(key=lambda key: group_hash(*key))
        counts = [0, 0, 0]
        total = sum(len(groups[key]) for key in keys)
        for key in keys:
            # Allocate to the partition furthest below its target count.
            deficits = [TARGET[i] * total - counts[i] for i in range(3)]
            selected = max(range(3), key=lambda i: (deficits[i], -i))
            assignment[key] = PARTITIONS[selected]
            counts[selected] += len(groups[key])

    records = [
        {
            "path": row["path"],
            "source": row["source"],
            "group": row["group"],
            "material": row["material"],
            "partition": assignment[(row["source"], row["group"])],
        }
        for row in rows
    ]
    records.sort(key=lambda row: row["path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    counts = Counter(row["partition"] for row in records)
    local = Counter(row["partition"] for row in records if row["source"] == "local")
    by_class = {
        part: dict(
            sorted(
                Counter(
                    row["material"]
                    for row in records
                    if row["source"] == "local" and row["partition"] == part
                ).items()
            )
        )
        for part in PARTITIONS
    }
    summary: dict[str, object] = {
        "seed": SEED,
        "protocol": "ws2-v1-group-source-dominant-material",
        "n_validation": len(records),
        "n_groups": len(groups),
        "partition_counts": dict(sorted(counts.items())),
        "local_partition_counts": dict(sorted(local.items())),
        "local_material_counts": by_class,
        "manifest": str(output),
    }
    (output.parent / "validation_manifest_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions", type=Path, default=Path("data/ws2/v0.2-local-data/predictions.csv")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/ws2/validation_partitions_v1.csv")
    )
    args = parser.parse_args()
    print(json.dumps(make_manifest(args.predictions, args.output), indent=2))


if __name__ == "__main__":
    main()
