"""Train leakage-controlled SVM material baselines on WS2 handcrafted features.

Select hyperparameters using local validation only. This script never reads test
labels for model selection and deliberately does not report test performance.

    uv run --with scikit-learn python scripts/ws2_train_svm.py
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
FEATURE_SETS = {
    "hsv": ("hsv",),
    "hog": ("hog",),
    "lbp": ("lbp",),
    "combined": ("hsv", "hog", "lbp"),
}
C_VALUES = (0.1, 1.0, 10.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions", type=Path, default=ROOT / "v0.2-local-data/predictions.csv"
    )
    parser.add_argument("--features", type=Path, default=ROOT / "features")
    parser.add_argument("--output", type=Path, default=Path("results/ws2/svm_validation.json"))
    args = parser.parse_args()

    with args.predictions.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    train_idx = np.array([i for i, row in enumerate(rows) if row["split"] == "train"])
    val_idx = np.array(
        [i for i, row in enumerate(rows) if row["split"] == "val" and row["source"] == "local"]
    )
    if not len(train_idx) or not len(val_idx):
        raise ValueError("Expected nonempty training and local validation splits")
    y = np.array([row["material"] for row in rows])
    groups = [row["source"] + "::" + row["group"] for row in rows]
    if set(groups[i] for i in train_idx) & set(groups[i] for i in val_idx):
        raise ValueError("Item leakage between training and validation")

    matrices = {
        name: np.load(args.features / f"{name}.npy", mmap_mode="r")
        for name in ("hsv", "hog", "lbp")
    }
    for name, matrix in matrices.items():
        if matrix.shape[0] != len(rows):
            raise ValueError(f"{name} rows {matrix.shape[0]} != predictions {len(rows)}")
        if not np.isfinite(matrix).all():
            raise ValueError(f"{name} has nonfinite values")
    trials: list[dict[str, object]] = []
    for feature_set, names in FEATURE_SETS.items():
        x = np.concatenate([matrices[name] for name in names], axis=1)
        for kernel in ("linear", "rbf"):
            for c_value in C_VALUES:
                pipeline = Pipeline(
                    [
                        ("scale", StandardScaler()),
                        (
                            "svm",
                            SVC(kernel=kernel, C=c_value, gamma="scale", class_weight="balanced"),
                        ),
                    ]
                )
                pipeline.fit(x[train_idx], y[train_idx])
                predicted = pipeline.predict(x[val_idx])
                result = {
                    "features": feature_set,
                    "kernel": kernel,
                    "C": c_value,
                    "train_n": len(train_idx),
                    "local_val_n": len(val_idx),
                    "local_val_macro_f1": float(
                        f1_score(y[val_idx], predicted, average="macro", zero_division=0)
                    ),
                    "local_val_accuracy": float(accuracy_score(y[val_idx], predicted)),
                }
                trials.append(result)
                print(
                    f"{feature_set:8s} {kernel:6s} C={c_value:<4} "
                    f"macro-F1={result['local_val_macro_f1']:.4f} "
                    f"accuracy={result['local_val_accuracy']:.4f}",
                    flush=True,
                )
    best = max(
        trials, key=lambda r: (float(r["local_val_macro_f1"]), float(r["local_val_accuracy"]))
    )
    report = {
        "data_version": "v0.2-local-data",
        "selection_split": "local validation only",
        "scaling": "StandardScaler fit on training rows only",
        "class_weight": "balanced",
        "trials": trials,
        "best_validation_configuration": best,
        "note": "Validation results only. Do not use test set to choose features or hyperparameters.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("\nBest validation configuration:")
    print(json.dumps(best, indent=2))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
