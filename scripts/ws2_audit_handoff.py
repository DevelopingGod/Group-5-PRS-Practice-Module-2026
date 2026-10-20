"""Audit WS1 prediction/embedding handoff without fitting or evaluating a model.

Run from repository root:
    uv run python scripts/ws2_audit_handoff.py
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

DEFAULT_ROOT = Path("data/ws2")
EXPECTED_EMBEDDING_DIM = 256


def audit(predictions: Path, embeddings_path: Path) -> dict[str, object]:
    """Check alignment, probabilities, labels and item-group split isolation."""
    with predictions.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        rows = list(reader)
    required = {"path", "source", "split", "material", "group", "predicted", "confidence"}
    missing = required.difference(columns)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")
    probability_columns = [name for name in columns if name.startswith("p_")]
    if len(probability_columns) != 10:
        raise ValueError(f"Expected 10 material probabilities, got {probability_columns}")
    if not rows:
        raise ValueError("Prediction CSV is empty")

    embeddings = np.load(embeddings_path, allow_pickle=False, mmap_mode="r")
    if embeddings.shape != (len(rows), EXPECTED_EMBEDDING_DIM):
        raise ValueError(
            f"Embedding shape {embeddings.shape} != ({len(rows)}, {EXPECTED_EMBEDDING_DIM})"
        )
    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain NaN or infinity")
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        raise ValueError("Embeddings are not L2-normalised")

    groups: dict[str, set[str]] = defaultdict(set)
    split_counts: Counter[str] = Counter()
    source_split_counts: Counter[str] = Counter()
    labels = {name[2:] for name in probability_columns}
    for index, row in enumerate(rows):
        split = row["split"]
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Row {index}: unknown split {split!r}")
        if row["material"] not in labels or row["predicted"] not in labels:
            raise ValueError(f"Row {index}: label not among probability columns")
        if not row["path"] or not row["group"]:
            raise ValueError(f"Row {index}: missing path or group")
        probs = np.array([float(row[name]) for name in probability_columns])
        if not np.isfinite(probs).all() or np.any(probs < 0) or np.any(probs > 1):
            raise ValueError(f"Row {index}: invalid probabilities")
        if not np.isclose(probs.sum(), 1.0, atol=2e-5):
            raise ValueError(f"Row {index}: probabilities do not sum to one")
        if row["predicted"] != probability_columns[int(probs.argmax())][2:]:
            raise ValueError(f"Row {index}: predicted label disagrees with argmax")
        if not np.isclose(float(row["confidence"]), probs.max(), atol=2e-5):
            raise ValueError(f"Row {index}: confidence disagrees with maximum probability")
        # Include source in the group key to avoid conflating unrelated dataset IDs.
        groups[f"{row['source']}::{row['group']}"].add(split)
        split_counts[split] += 1
        source_split_counts[f"{row['source']}::{split}"] += 1

    leaked = [group for group, splits in groups.items() if len(splits) > 1]
    if leaked:
        raise ValueError(f"Physical-item groups cross splits: {leaked[:10]}")
    return {
        "status": "passed",
        "n_images": len(rows),
        "embedding_shape": list(embeddings.shape),
        "probability_class_order": [name[2:] for name in probability_columns],
        "split_counts": dict(sorted(split_counts.items())),
        "source_split_counts": dict(sorted(source_split_counts.items())),
        "n_groups": len(groups),
        "groups_crossing_splits": 0,
        "note": "Row-order alignment cannot be independently proven without stable IDs in the NPY.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions", type=Path, default=DEFAULT_ROOT / "v0.2-local-data/predictions.csv"
    )
    parser.add_argument(
        "--embeddings", type=Path, default=DEFAULT_ROOT / "v0.2-local-data/embeddings.npy"
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON report path")
    args = parser.parse_args()
    result = audit(args.predictions, args.embeddings)
    report = json.dumps(result, indent=2)
    print(report)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
