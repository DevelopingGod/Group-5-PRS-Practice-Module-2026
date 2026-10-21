"""WS2-10: fit cosine-distance OOD baselines and ID rejection thresholds.

Uses training embeddings as references and validation embeddings to set
known-item false-rejection thresholds. Does NOT claim OOD detection performance
without a separate unknown-object evaluation set.

Run:
    uv run python scripts/ws2_ood_baselines.py
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path("data/ws2")
OUT = Path("results/ws2/ood")


def normalise(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(norms < 1e-8):
        raise ValueError("Zero-length embedding")
    return x / norms


def knn_distance(query: np.ndarray, reference: np.ndarray, k: int = 5) -> np.ndarray:
    """Mean cosine distance to k nearest training examples, in bounded batches."""
    if len(reference) < k:
        raise ValueError("Insufficient training reference vectors")
    scores = np.empty(len(query), dtype=np.float32)
    for start in range(0, len(query), 128):
        end = min(start + 128, len(query))
        similarities = query[start:end] @ reference.T
        top = np.partition(similarities, kth=len(reference) - k, axis=1)[:, -k:]
        scores[start:end] = 1.0 - top.mean(axis=1)
    return scores


def centroid_distance(query: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Cosine distance to nearest material centroid; no ground-truth query label needed."""
    return 1.0 - (query @ centroids.T).max(axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions", type=Path, default=ROOT / "v0.2-local-data/predictions.csv"
    )
    parser.add_argument("--embeddings", type=Path, default=ROOT / "v0.2-local-data/embeddings.npy")
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--target-fpr", type=float, default=0.05)
    args = parser.parse_args()
    if not 0 < args.target_fpr < 1:
        raise ValueError("target-fpr must be between zero and one")
    with args.predictions.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    embedding = np.load(args.embeddings, allow_pickle=False, mmap_mode="r")
    if embedding.shape != (len(rows), 256):
        raise ValueError(f"Unexpected embedding shape {embedding.shape}")
    if not np.isfinite(embedding).all():
        raise ValueError("Non-finite embeddings")
    train_ids = np.array([i for i, row in enumerate(rows) if row["split"] == "train"])
    val_ids = np.array([i for i, row in enumerate(rows) if row["split"] == "val"])
    if not len(train_ids) or not len(val_ids):
        raise ValueError("Training and validation rows required")
    train_groups = {(rows[int(i)]["source"], rows[int(i)]["group"]) for i in train_ids}
    val_groups = {(rows[int(i)]["source"], rows[int(i)]["group"]) for i in val_ids}
    if train_groups & val_groups:
        raise ValueError("Physical-item leakage between train and validation")
    train = normalise(np.asarray(embedding[train_ids], dtype=np.float32))
    val = normalise(np.asarray(embedding[val_ids], dtype=np.float32))
    classes = [key[2:] for key in rows[0] if key.startswith("p_")]
    train_labels = np.array([rows[int(i)]["material"] for i in train_ids])
    centroids = []
    for label in classes:
        members = train[train_labels == label]
        if not len(members):
            raise ValueError(f"Missing training material: {label}")
        centroids.append(normalise(members.mean(axis=0, keepdims=True))[0])
    centres = np.stack(centroids).astype(np.float32)
    scores = {
        "cosine_knn5": knn_distance(val, train),
        "cosine_centroid": centroid_distance(val, centres),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "n_train": len(train),
        "n_validation": len(val),
        "target_known_rejection": args.target_fpr,
        "methods": {},
        "limitations": [
            "Known-item false rejection is not OOD recall or AUROC.",
            "Thresholds selected on known validation examples only.",
            "Public and Singapore validation distributions differ.",
            "A separate unknown-object set is required before deployment claims.",
        ],
    }
    local = np.array([rows[int(i)]["source"] == "local" for i in val_ids])
    for name, distance in scores.items():
        # Conservative empirical threshold; reject only distances strictly above it.
        threshold = float(np.quantile(distance, 1 - args.target_fpr, method="higher"))
        rejected = distance > threshold
        report["methods"][name] = {
            "threshold": threshold,
            "known_rejection_all": float(rejected.mean()),
            "known_rejection_local": float(rejected[local].mean()) if local.any() else None,
            "local_rejected": int(rejected[local].sum()),
            "local_total": int(local.sum()),
            "score_median_local": float(np.median(distance[local])) if local.any() else None,
            "score_median_public": float(np.median(distance[~local])) if (~local).any() else None,
        }
    np.savez_compressed(
        args.output / "ood_reference.npz",
        train_embeddings=train,
        centroids=centres,
        class_names=np.asarray(classes),
    )
    (args.output / "thresholds.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    print(f"Saved reference: {args.output / 'ood_reference.npz'}")


if __name__ == "__main__":
    main()
