"""WS2-11: group-separated local vs pooled OOD threshold calibration.

References use TRAIN embeddings only. Local validation physical-item groups are
split into calibration and diagnostic partitions. No test rows are used.

Run:
    uv run python scripts/ws2_ood_local_thresholds.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path("data/ws2")


def unit(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError("Zero-length embedding")
    return x / norm


def knn_scores(query: np.ndarray, reference: np.ndarray, k: int = 5) -> np.ndarray:
    result = np.empty(len(query), dtype=np.float32)
    for start in range(0, len(query), 128):
        sim = query[start : start + 128] @ reference.T
        nearest = np.partition(sim, -k, axis=1)[:, -k:]
        result[start : start + 128] = 1 - nearest.mean(axis=1)
    return result


def split_group(source: str, group: str, seed: int = 42) -> bool:
    """Stable item-group assignment: True is calibration, False diagnostic."""
    digest = hashlib.sha256(f"{seed}:{source}:{group}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % 2 == 0


def rejection(scores: np.ndarray, threshold: float) -> dict[str, float | int]:
    rejected = int(np.sum(scores > threshold))
    return {
        "n": len(scores),
        "rejected": rejected,
        "rejection_rate": rejected / len(scores) if len(scores) else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions", type=Path, default=ROOT / "v0.2-local-data/predictions.csv"
    )
    parser.add_argument("--embeddings", type=Path, default=ROOT / "v0.2-local-data/embeddings.npy")
    parser.add_argument(
        "--output", type=Path, default=Path("results/ws2/ood/local_thresholds.json")
    )
    parser.add_argument("--target-rejection", type=float, default=0.05)
    args = parser.parse_args()
    if not 0 < args.target_rejection < 1:
        raise ValueError("Target rejection must lie strictly between 0 and 1")

    with args.predictions.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    raw = np.load(args.embeddings, allow_pickle=False, mmap_mode="r")
    if raw.shape != (len(rows), 256) or not np.isfinite(raw).all():
        raise ValueError("Invalid embedding shape or non-finite values")

    train_idx = np.array([i for i, r in enumerate(rows) if r["split"] == "train"])
    val_idx = np.array([i for i, r in enumerate(rows) if r["split"] == "val"])
    if not len(train_idx) or not len(val_idx):
        raise ValueError("Missing training or validation images")
    train_groups = {(rows[int(i)]["source"], rows[int(i)]["group"]) for i in train_idx}
    val_groups = {(rows[int(i)]["source"], rows[int(i)]["group"]) for i in val_idx}
    if train_groups & val_groups:
        raise ValueError("Physical item group crosses train/validation")

    ref = unit(np.asarray(raw[train_idx], dtype=np.float32))
    query = unit(np.asarray(raw[val_idx], dtype=np.float32))
    train_labels = np.array([rows[int(i)]["material"] for i in train_idx])
    classes = sorted(set(train_labels))
    centres = unit(np.stack([ref[train_labels == label].mean(axis=0) for label in classes]))
    val_rows = [rows[int(i)] for i in val_idx]
    local = np.array([r["source"] == "local" for r in val_rows])
    cal = np.array([split_group(r["source"], r["group"]) for r in val_rows])
    diagnostic = ~cal
    if any((mask & local).sum() < 20 for mask in (cal, diagnostic)):
        raise ValueError("Insufficient local images in calibration or diagnostic subset")
    scores = {
        "cosine_knn5": knn_scores(query, ref),
        "cosine_centroid": 1 - (query @ centres.T).max(axis=1),
    }
    report: dict[str, object] = {
        "reference_train": len(train_idx),
        "validation_total": len(val_idx),
        "local_calibration": int((local & cal).sum()),
        "local_diagnostic": int((local & diagnostic).sum()),
        "target_rejection": args.target_rejection,
        "group_split": "stable SHA256 group assignment, seed 42",
        "methods": {},
        "limitations": [
            "Known-image rejection only; OOD recall and AUROC require unfamiliar images.",
            "Pooled calibration reflects public-dataset class and source composition.",
            "Small local diagnostic set gives uncertain rejection-rate estimates.",
            "Thresholds are provisional and must not be selected using diagnostic outcomes.",
        ],
    }
    for name, score in scores.items():
        comparisons: dict[str, object] = {}
        for calibration_name, fit_mask in (
            ("pooled", cal),
            ("local_only", cal & local),
        ):
            threshold = float(
                np.quantile(score[fit_mask], 1 - args.target_rejection, method="higher")
            )
            comparisons[calibration_name] = {
                "threshold": threshold,
                "calibration": rejection(score[fit_mask], threshold),
                "diagnostic_local": rejection(score[diagnostic & local], threshold),
                "diagnostic_public": rejection(score[diagnostic & ~local], threshold),
            }
        report["methods"][name] = comparisons
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
