"""WS2-09: exploratory leakage-controlled CNN stacking and hybrid voting.

Three disjoint validation-group partitions: temperature calibration, meta-fit,
and evaluation. CNN base models are frozen. The SVM is fitted on original train
embeddings, never on validation labels. This is a small development-set study,
not independent final-test evidence.

uv run --with scikit-learn --with scipy python scripts/ws2_stacking.py
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

CLASSES = (
    "paper",
    "plastic",
    "glass",
    "metal",
    "beverage_carton",
    "styrofoam",
    "e_waste",
    "textile",
    "food_waste",
    "other_general",
)
BLUE = frozenset(CLASSES[:5])
ROOT = Path("data/ws2")


def load(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len({row["path"] for row in rows}) != len(rows):
        raise ValueError(f"Duplicate paths: {path}")
    return {row["path"]: row for row in rows}


def probs(rows: list[dict[str, str]]) -> np.ndarray:
    result = np.array(
        [[float(row[f"p_{label}"]) for label in CLASSES] for row in rows], dtype=float
    )
    if not np.isfinite(result).all() or not np.allclose(result.sum(axis=1), 1, atol=2e-4):
        raise ValueError("Invalid probabilities")
    return result


def scale(p: np.ndarray, temperature: float) -> np.ndarray:
    z = np.log(np.clip(p, 1e-12, 1)) / temperature
    z -= z.max(axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / exp.sum(axis=1, keepdims=True)


def calibrate(p: np.ndarray, y: np.ndarray) -> float:
    result = minimize_scalar(
        lambda log_t: log_loss(y, scale(p, float(np.exp(log_t))), labels=np.arange(len(CLASSES))),
        method="bounded",
        bounds=(np.log(0.2), np.log(10)),
    )
    return float(np.exp(result.x))


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float | int]:
    pred = p.argmax(axis=1)
    not_blue = np.array([CLASSES[int(i)] not in BLUE for i in y])
    blue_pred = np.array([CLASSES[int(i)] in BLUE for i in pred])
    return {
        "n": len(y),
        "macro_f1": float(
            f1_score(y, pred, labels=np.arange(10), average="macro", zero_division=0)
        ),
        "accuracy": float(accuracy_score(y, pred)),
        "costly_error_count": int((not_blue & blue_pred).sum()),
        "non_blue_count": int(not_blue.sum()),
        "costly_error_rate": float(blue_pred[not_blue].mean()) if not_blue.any() else 0.0,
        "nll": float(log_loss(y, p, labels=np.arange(10))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cnn", type=Path, default=ROOT / "v0.2-local-data/predictions.csv")
    parser.add_argument("--swin", type=Path, default=ROOT / "run3/swin_validation.csv")
    parser.add_argument("--embeddings", type=Path, default=ROOT / "v0.2-local-data/embeddings.npy")
    parser.add_argument("--output", type=Path, default=Path("results/ws2/stacking_validation.json"))
    args = parser.parse_args()

    cnn = load(args.cnn)
    swin = load(args.swin)
    all_cnn = list(cnn.values())
    embeddings = np.load(args.embeddings, mmap_mode="r", allow_pickle=False)
    if embeddings.shape != (len(all_cnn), 256):
        raise ValueError("CNN CSV and embeddings are not aligned")
    # Dict insertion order follows the original CSV; explicitly align validation paths.
    paths = sorted(path for path, row in swin.items() if row["split"] == "val")
    if len(paths) != len(swin):
        raise ValueError("Swin file must contain validation rows only")
    if any(path not in cnn or cnn[path]["split"] != "val" for path in paths):
        raise ValueError("Missing matching CNN validation rows")
    for path in paths:
        if any(cnn[path][field] != swin[path][field] for field in ("material", "source", "group")):
            raise ValueError(f"Metadata mismatch: {path}")
    c_rows = [cnn[path] for path in paths]
    s_rows = [swin[path] for path in paths]
    pc = probs(c_rows)
    ps_raw = probs(s_rows)
    y = np.array([CLASSES.index(row["material"]) for row in c_rows])
    index = {row["path"]: i for i, row in enumerate(all_cnn)}
    train_ids = np.array([i for i, row in enumerate(all_cnn) if row["split"] == "train"])
    train_y = np.array([CLASSES.index(all_cnn[int(i)]["material"]) for i in train_ids])
    svm = make_pipeline(
        StandardScaler(),
        SVC(kernel="linear", C=0.1, class_weight="balanced", probability=True, random_state=42),
    )
    svm.fit(embeddings[train_ids], train_y)
    svm_raw = np.zeros((len(paths), len(CLASSES)), dtype=float)
    svm_classes = svm.named_steps["svc"].classes_.astype(int)
    svm_raw[:, svm_classes] = svm.predict_proba(embeddings[[index[p] for p in paths]])

    # Split by physical-item group, within source. Do not change original dataset splits.
    groups_by_source: dict[str, list[str]] = {}
    for row in c_rows:
        key = row["source"] + "::" + row["group"]
        groups_by_source.setdefault(row["source"], []).append(key)
    rng = np.random.default_rng(42)
    partition: dict[str, int] = {}
    for groups in groups_by_source.values():
        unique = sorted(set(groups))
        for j, idx in enumerate(rng.permutation(len(unique))):
            fraction = j / len(unique)
            partition[unique[int(idx)]] = 0 if fraction < 0.4 else (1 if fraction < 0.7 else 2)
    fold = np.array([partition[row["source"] + "::" + row["group"]] for row in c_rows])
    cal, fit, hold = fold == 0, fold == 1, fold == 2
    local = np.array([row["source"] == "local" for row in c_rows])
    if any((mask & local).sum() < 20 for mask in (cal, fit, hold)):
        raise ValueError("Too few local samples in a partition")

    swin_t = calibrate(ps_raw[cal], y[cal])
    svm_t = calibrate(svm_raw[cal], y[cal])
    ps = scale(ps_raw, swin_t)
    pm = scale(svm_raw, svm_t)

    # Meta-models use only out-of-sample CNN predictions, never CNN training predictions.
    # Keep regularisation fixed rather than searching on the small holdout.
    features = {
        "cnn_swin": np.hstack((pc, ps)),
        "cnn_swin_svm": np.hstack((pc, ps, pm)),
    }
    results: dict[str, object] = {
        "seed": 42,
        "n_validation": len(paths),
        "partition_counts": {
            name: {"all": int(mask.sum()), "local": int((mask & local).sum())}
            for name, mask in (("calibration", cal), ("meta_fit", fit), ("holdout", hold))
        },
        "temperatures": {"swin": swin_t, "embedding_svm": svm_t},
        "methods": {},
        "limitations": [
            "Holdout is a partition of previously used validation data, not independent field test.",
            "SVM probability=True calibrates internally on original training data; temperature is fitted separately.",
            "CNN and embedding-SVM share ConvNeXt representation and may have correlated errors.",
            "Stacking may be unstable with few local images; no hyperparameters tuned on holdout.",
        ],
    }
    predictions: dict[str, np.ndarray] = {
        "convnext": pc,
        "swin_calibrated": ps,
        "embedding_svm_calibrated": pm,
        "soft_vote_50_50": 0.5 * pc + 0.5 * ps,
        "soft_vote_cnn_svm": 0.75 * pc + 0.25 * pm,
    }
    for name, x in features.items():
        meta = LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced")
        meta.fit(x[fit], y[fit])
        p = np.zeros((len(paths), len(CLASSES)))
        p[:, meta.classes_] = meta.predict_proba(x)
        predictions["stack_" + name] = p

    for name, p in predictions.items():
        results["methods"][name] = {
            "local_holdout": metrics(y[hold & local], p[hold & local]),
            "all_holdout": metrics(y[hold], p[hold]),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print("Partition counts:", json.dumps(results["partition_counts"]))
    print("Temperature parameters:", json.dumps(results["temperatures"]))
    print("Local held-out development subset:")
    for name, value in results["methods"].items():
        m = value["local_holdout"]
        print(
            f"{name:24s} F1={m['macro_f1']:.4f} accuracy={m['accuracy']:.4f} "
            f"costly={m['costly_error_count']}/{m['non_blue_count']}"
        )
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
