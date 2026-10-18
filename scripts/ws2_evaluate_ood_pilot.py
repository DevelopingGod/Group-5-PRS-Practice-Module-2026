"""Evaluate OOD novelty detection using fresh unfamiliar-object photographs.

First collect 30-50 images under data/ws2/ood_pilot/<physical_item_id>/*.jpg.
The directory names represent unique physical objects, not material classes.

Usage:
  uv run python scripts/ws2_evaluate_ood_pilot.py --model-dir data/ws2/models/convnext_export

Requires the same WS1 ConvNeXt ONNX export used to generate embeddings.npy:
config.json, model.onnx, head.npz. Downloads nothing automatically.
Local calibration thresholds are frozen from WS2-11, not fitted on OOD photos.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "services/inference"))
from app.vision import VisionModel  # noqa: E402

ROOT = Path("data/ws2")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
METHODS = ("cosine_knn5", "cosine_centroid")


def normalise(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError("Zero-length embedding")
    return x / norm


def scores(x: np.ndarray, reference: np.ndarray, centroids: np.ndarray) -> dict[str, np.ndarray]:
    nearest = np.empty(len(x), dtype=np.float32)
    for start in range(0, len(x), 128):
        similarity = x[start : start + 128] @ reference.T
        top = np.partition(similarity, -5, axis=1)[:, -5:]
        nearest[start : start + 128] = 1 - top.mean(axis=1)
    return {
        "cosine_knn5": nearest,
        "cosine_centroid": 1 - (x @ centroids.T).max(axis=1),
    }


def diagnostic_group(source: str, group: str) -> bool:
    digest = hashlib.sha256(f"42:{source}:{group}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % 2 != 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--ood-images", type=Path, default=ROOT / "ood_pilot")
    parser.add_argument("--predictions", type=Path, default=ROOT / "v0.2-local-data/predictions.csv")
    parser.add_argument("--embeddings", type=Path, default=ROOT / "v0.2-local-data/embeddings.npy")
    parser.add_argument("--reference", type=Path, default=Path("results/ws2/ood/ood_reference.npz"))
    parser.add_argument("--thresholds", type=Path, default=Path("results/ws2/ood/local_thresholds.json"))
    parser.add_argument("--output", type=Path, default=Path("results/ws2/ood/pilot_evaluation.json"))
    args = parser.parse_args()

    files = sorted(
        path for path in args.ood_images.glob("*/*")
        if path.suffix.lower() in IMAGE_EXTENSIONS and path.is_file()
    )
    if not files:
        raise ValueError("No OOD images: expected ood_pilot/<physical_item_id>/<image>.jpg")
    item_ids = [path.parent.name for path in files]
    if len(set(item_ids)) < 5:
        raise ValueError("Collect at least five distinct physical objects for the pilot")
    model = VisionModel.load(args.model_dir)
    if model.pca_components.shape[0] != 256:
        raise ValueError("Expected 256D ConvNeXt export")
    unknown_embeddings = []
    for i, path in enumerate(files):
        with Image.open(path) as image:
            x, _ = model.preprocess(image.convert("RGB"))
        _, pooled, _ = model.session.run(None, {"image": x})
        unknown_embeddings.append(model.embed(np.asarray(pooled[0], dtype=np.float32)))
        if (i + 1) % 10 == 0:
            print(f"Embedded {i + 1}/{len(files)} OOD photos", flush=True)
    unknown = normalise(np.asarray(unknown_embeddings, dtype=np.float32))

    with args.predictions.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    known_embeddings = np.load(args.embeddings, allow_pickle=False, mmap_mode="r")
    if known_embeddings.shape != (len(rows), 256):
        raise ValueError("Known embedding row mismatch")
    known_ids = [
        i for i, row in enumerate(rows)
        if row["split"] == "val" and row["source"] == "local"
        and diagnostic_group(row["source"], row["group"])
    ]
    if not known_ids:
        raise ValueError("No local diagnostic known samples")
    known = normalise(np.asarray(known_embeddings[known_ids], dtype=np.float32))
    with np.load(args.reference, allow_pickle=False) as archive:
        reference = normalise(archive["train_embeddings"].astype(np.float32))
        centroids = normalise(archive["centroids"].astype(np.float32))
    thresholds = json.loads(args.thresholds.read_text(encoding="utf-8"))
    known_scores = scores(known, reference, centroids)
    unknown_scores = scores(unknown, reference, centroids)
    results: dict[str, object] = {
        "n_ood_photos": len(files),
        "n_ood_physical_items": len(set(item_ids)),
        "n_known_local_diagnostic": len(known),
        "model_dir": str(args.model_dir),
        "methods": {},
        "limitations": [
            "Pilot OOD items were selected by the collector and may not represent realistic unknowns.",
            "Multiple photos of one object are correlated; photo-level AUROC is descriptive.",
            "These pilot results must not be used to tune thresholds and claim independent evaluation.",
            "A known waste image that is visually difficult is not automatically OOD.",
        ],
    }
    for name in METHODS:
        threshold = float(thresholds["methods"][name]["local_only"]["threshold"])
        kscore = known_scores[name]
        uscore = unknown_scores[name]
        auc = float(
            roc_auc_score(
                np.r_[np.zeros(len(kscore)), np.ones(len(uscore))],
                np.r_[kscore, uscore],
            )
        )
        results["methods"][name] = {
            "threshold": threshold,
            "known_false_rejection": float(np.mean(kscore > threshold)),
            "known_false_rejected_n": int(np.sum(kscore > threshold)),
            "ood_recall": float(np.mean(uscore > threshold)),
            "ood_rejected_n": int(np.sum(uscore > threshold)),
            "auroc": auc,
            "median_known_score": float(np.median(kscore)),
            "median_ood_score": float(np.median(uscore)),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    with (args.output.parent / "pilot_scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["item_id", "file", *METHODS])
        for i, path in enumerate(files):
            writer.writerow([
                item_ids[i], str(path.relative_to(args.ood_images)),
                *(float(unknown_scores[name][i]) for name in METHODS),
            ])
    print(json.dumps(results, indent=2))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
