"""WS2-05: compare SVM classifiers on frozen ConvNeXt embeddings.

Only training rows are used to fit models and scalers; only local validation rows
are used for hyperparameter selection. Test rows are never scored.

Run:
    uv run --with scikit-learn python scripts/ws2_embedding_svm.py
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path("data/ws2")


def evaluate(predictions: Path, embeddings_path: Path, hsv_path: Path) -> dict[str, object]:
    """Evaluate frozen-feature SVM candidates on the same local validation rows."""
    with predictions.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Empty prediction CSV")
    embeddings = np.load(embeddings_path, mmap_mode="r")
    hsv = np.load(hsv_path, mmap_mode="r")
    if embeddings.shape != (len(rows), 256) or hsv.shape != (len(rows), 48):
        raise ValueError("Feature matrices must align with predictions.csv")
    train = np.array([i for i, row in enumerate(rows) if row["split"] == "train"])
    val = np.array(
        [i for i, row in enumerate(rows) if row["split"] == "val" and row["source"] == "local"]
    )
    if not len(train) or not len(val):
        raise ValueError("Missing training or local validation samples")
    groups_train = {(rows[int(i)]["source"], rows[int(i)]["group"]) for i in train}
    groups_val = {(rows[int(i)]["source"], rows[int(i)]["group"]) for i in val}
    if groups_train & groups_val:
        raise ValueError("Physical-item groups overlap train and validation")

    labels = np.asarray([row["material"] for row in rows])
    cnn = np.asarray([row["predicted"] for row in rows])
    classes = [key[2:] for key in rows[0] if key.startswith("p_")]
    y_val = labels[val]
    cnn_val = cnn[val]
    cnn_correct = cnn_val == y_val
    baseline = {
        "macro_f1": float(
            f1_score(y_val, cnn_val, labels=classes, average="macro", zero_division=0)
        ),
        "accuracy": float(accuracy_score(y_val, cnn_val)),
    }
    trials: list[dict[str, object]] = []
    for representation, matrix in (("hsv", hsv), ("cnn_embedding", embeddings)):
        for kernel in ("linear", "rbf"):
            for c_value in (0.1, 1.0, 10.0):
                pipeline = Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        (
                            "svm",
                            SVC(
                                kernel=kernel,
                                C=c_value,
                                gamma="scale",
                                class_weight="balanced",
                            ),
                        ),
                    ]
                )
                pipeline.fit(matrix[train], labels[train])
                predicted = pipeline.predict(matrix[val])
                correct = predicted == y_val
                result = {
                    "representation": representation,
                    "kernel": kernel,
                    "C": c_value,
                    "macro_f1": float(
                        f1_score(y_val, predicted, labels=classes, average="macro", zero_division=0)
                    ),
                    "accuracy": float(accuracy_score(y_val, predicted)),
                    "both_correct": int(np.sum(correct & cnn_correct)),
                    "cnn_only_correct": int(np.sum(~correct & cnn_correct)),
                    "svm_only_correct": int(np.sum(correct & ~cnn_correct)),
                    "both_wrong": int(np.sum(~correct & ~cnn_correct)),
                    "prediction_disagreement": int(np.sum(predicted != cnn_val)),
                }
                trials.append(result)
                print(
                    f"{representation:13s} {kernel:6s} C={c_value:<4} "
                    f"macro-F1={result['macro_f1']:.4f} accuracy={result['accuracy']:.4f} "
                    f"rescues={result['svm_only_correct']}",
                    flush=True,
                )
    best_embedding = max(
        (trial for trial in trials if trial["representation"] == "cnn_embedding"),
        key=lambda trial: (float(trial["macro_f1"]), float(trial["accuracy"])),
    )
    return {
        "data_version": "v0.2-local-data",
        "split": "local validation only",
        "train_n": len(train),
        "local_val_n": len(val),
        "cnn_baseline": baseline,
        "trials": trials,
        "best_embedding_svm": best_embedding,
        "limitations": [
            "CNN embeddings share a learned backbone with the CNN classifier.",
            "SVM selection and reported validation metrics use the same local validation set.",
            "Training embeddings are in-sample CNN features; no stacking model is fitted.",
            "Do not interpret this as an independent handcrafted-feature baseline.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions", type=Path, default=ROOT / "v0.2-local-data/predictions.csv"
    )
    parser.add_argument("--embeddings", type=Path, default=ROOT / "v0.2-local-data/embeddings.npy")
    parser.add_argument("--hsv", type=Path, default=ROOT / "features/hsv.npy")
    parser.add_argument(
        "--output", type=Path, default=Path("results/ws2/embedding_svm_validation.json")
    )
    args = parser.parse_args()
    result = evaluate(args.predictions, args.embeddings, args.hsv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("\nCNN validation baseline:", json.dumps(result["cnn_baseline"]))
    print("Best embedding SVM:", json.dumps(result["best_embedding_svm"], indent=2))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
