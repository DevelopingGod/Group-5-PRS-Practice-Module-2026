"""Compare frozen CNN and selected handcrafted SVM on identical local validation rows.

This is exploratory: SVM hyperparameters were selected on these same validation rows.
No test labels are loaded and no ensemble weights are fitted.

    uv run --with scikit-learn python scripts/ws2_compare_validation.py
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path("data/ws2")


def compare(predictions: Path, features: Path) -> dict[str, object]:
    """Train the selected SVM on train only and compare its errors with CNN on local val."""
    with predictions.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    classes = [name[2:] for name in rows[0] if name.startswith("p_")]
    train = np.array([i for i, r in enumerate(rows) if r["split"] == "train"])
    val = np.array(
        [i for i, r in enumerate(rows) if r["split"] == "val" and r["source"] == "local"]
    )
    if not len(train) or not len(val):
        raise ValueError("Missing training or local validation images")
    group = lambda i: (rows[int(i)]["source"], rows[int(i)]["group"])
    if {group(i) for i in train} & {group(i) for i in val}:
        raise ValueError("Physical-item group leakage between train and validation")
    hsv = np.load(features / "hsv.npy", mmap_mode="r")
    if hsv.shape != (len(rows), 48):
        raise ValueError(f"Unexpected HSV matrix shape: {hsv.shape}")
    truth = np.asarray([r["material"] for r in rows])
    cnn = np.asarray([r["predicted"] for r in rows])
    svm = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", C=10.0, gamma="scale", class_weight="balanced")),
        ]
    )
    svm.fit(hsv[train], truth[train])
    svm_val = svm.predict(hsv[val])
    y = truth[val]
    cnn_val = cnn[val]
    cnn_ok = cnn_val == y
    svm_ok = svm_val == y
    both_correct = int(np.sum(cnn_ok & svm_ok))
    cnn_only = int(np.sum(cnn_ok & ~svm_ok))
    svm_only = int(np.sum(~cnn_ok & svm_ok))
    both_wrong = int(np.sum(~cnn_ok & ~svm_ok))
    if both_correct + cnn_only + svm_only + both_wrong != len(val):
        raise AssertionError("Disagreement partition does not cover validation rows")

    confusion_pairs = Counter(
        (str(true), str(pred))
        for true, pred in zip(y, svm_val, strict=True)
        if true != pred
    )
    rescued = [
        {
            "path": rows[int(i)]["path"],
            "material": str(y[j]),
            "cnn_prediction": str(cnn_val[j]),
            "svm_prediction": str(svm_val[j]),
            "cnn_confidence": float(rows[int(i)]["confidence"]),
        }
        for j, i in enumerate(val)
        if svm_ok[j] and not cnn_ok[j]
    ]
    report = {
        "scope": "Local validation only; exploratory and SVM-selected-on-validation",
        "n_local_validation": int(len(val)),
        "cnn": {
            "macro_f1": float(f1_score(y, cnn_val, labels=classes, average="macro", zero_division=0)),
            "accuracy": float(accuracy_score(y, cnn_val)),
            "per_class": classification_report(
                y, cnn_val, labels=classes, output_dict=True, zero_division=0
            ),
        },
        "svm": {
            "features": "HSV (48 dimensions)",
            "kernel": "rbf",
            "C": 10.0,
            "macro_f1": float(f1_score(y, svm_val, labels=classes, average="macro", zero_division=0)),
            "accuracy": float(accuracy_score(y, svm_val)),
            "per_class": classification_report(
                y, svm_val, labels=classes, output_dict=True, zero_division=0
            ),
        },
        "agreement": {
            "both_correct": both_correct,
            "cnn_only_correct": cnn_only,
            "svm_only_correct": svm_only,
            "both_wrong": both_wrong,
            "prediction_disagreement": int(np.sum(cnn_val != svm_val)),
        },
        "svm_top_confusions": [
            {"true": true, "predicted": pred, "count": count}
            for (true, pred), count in confusion_pairs.most_common(12)
        ],
        "svm_rescues": rescued,
        "limitations": [
            "SVM configuration was selected on local validation; its score is selection-biased.",
            "No image-quality metadata is available in this export; visual causes are hypotheses.",
            "This does not estimate ensemble performance or test-set generalisation.",
        ],
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions", type=Path, default=ROOT / "v0.2-local-data/predictions.csv"
    )
    parser.add_argument("--features", type=Path, default=ROOT / "features")
    parser.add_argument("--output", type=Path, default=Path("results/ws2/validation_comparison.json"))
    args = parser.parse_args()
    result = compare(args.predictions, args.features)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Local validation images: {result['n_local_validation']}")
    for name in ("cnn", "svm"):
        metrics = result[name]
        print(f"{name.upper()}: macro-F1={metrics['macro_f1']:.4f}, accuracy={metrics['accuracy']:.4f}")
    print("Disagreement breakdown:")
    print(json.dumps(result["agreement"], indent=2))
    print("SVM's most frequent confusions:")
    print(json.dumps(result["svm_top_confusions"][:6], indent=2))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
