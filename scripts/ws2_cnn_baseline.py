"""Reproduce WS1's ConvNeXt-T material baseline from the frozen prediction export.

No fitting or threshold selection is performed. The local test subset is used only
to reproduce the already reported WS1 benchmark, not to choose WS2 models.

    uv run python scripts/ws2_cnn_baseline.py
    uv run python scripts/ws2_cnn_baseline.py --split val --output results/ws2/cnn_val.json
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

DEFAULT_PREDICTIONS = Path("data/ws2/v0.2-local-data/predictions.csv")
DEFAULT_OUTPUT = Path("results/ws2/cnn_local_test.json")


def score(predictions: Path, split: str, source: str) -> dict[str, object]:
    """Calculate per-class classification metrics and confusion counts without sklearn."""
    with predictions.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        required = {"split", "source", "material", "predicted"}
        if not required.issubset(columns):
            raise ValueError(f"Missing columns: {sorted(required.difference(columns))}")
        classes = [column[2:] for column in columns if column.startswith("p_")]
        if len(classes) != 10:
            raise ValueError(f"Expected ten material classes; found {classes}")
        rows = [row for row in reader if row["split"] == split and row["source"] == source]
    if not rows:
        raise ValueError(f"No rows for split={split!r}, source={source!r}")

    counts: Counter[tuple[str, str]] = Counter()
    for row in rows:
        truth, predicted = row["material"], row["predicted"]
        if truth not in classes or predicted not in classes:
            raise ValueError(f"Unknown class in row: {truth!r}, {predicted!r}")
        counts[(truth, predicted)] += 1

    class_metrics: dict[str, dict[str, float | int]] = {}
    for label in classes:
        tp = counts[(label, label)]
        support = sum(counts[(label, pred)] for pred in classes)
        predicted_total = sum(counts[(true, label)] for true in classes)
        precision = tp / predicted_total if predicted_total else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        class_metrics[label] = {
            "support": support,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
        }
    present = [label for label in classes if class_metrics[label]["support"] > 0]
    macro_f1 = sum(float(class_metrics[label]["f1"]) for label in present) / len(present)
    correct = sum(counts[(label, label)] for label in classes)
    confusions = [
        {"true": true, "predicted": pred, "count": count}
        for (true, pred), count in counts.items()
        if true != pred
    ]
    confusions.sort(key=lambda entry: (-int(entry["count"]), str(entry["true"])))
    matrix = [[counts[(true, pred)] for pred in classes] for true in classes]
    return {
        "model": "WS1 ConvNeXt-T v0.2-local-data (precomputed predictions)",
        "split": split,
        "source": source,
        "n_images": len(rows),
        "class_order": classes,
        "macro_f1": round(macro_f1, 6),
        "accuracy": round(correct / len(rows), 6),
        "per_class": class_metrics,
        "confusion_matrix": matrix,
        "top_confusions": confusions[:15],
        "note": "Vision material metrics only; not contamination recall or policy verdict accuracy.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--source", default="local")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = score(args.predictions, args.split, args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    summary = {key: result[key] for key in ("split", "source", "n_images", "macro_f1", "accuracy")}
    print(json.dumps(summary, indent=2))
    print("Top confusions:")
    for entry in result["top_confusions"][:8]:
        print(f"  {entry['true']} -> {entry['predicted']}: {entry['count']}")
    print(f"Full report: {args.output}")


if __name__ == "__main__":
    main()
