"""WS2-14: canonical-manifest hybrid voting and regularised stacking ablation.

Calibration rows fit Swin and SVM temperatures. Meta-training rows fit stackers
and select a voting weight. Diagnostic rows are scored once with frozen choices.
The embedding SVM uses only original TRAIN embeddings and labels.

uv run --with scikit-learn --with scipy python scripts/ws2_hybrid_stack_v1.py
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

CLASSES = ("paper", "plastic", "glass", "metal", "beverage_carton",
           "styrofoam", "e_waste", "textile", "food_waste", "other_general")
BLUE = set(CLASSES[:5])
ROOT = Path("data/ws2")


def load(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != len({r["path"] for r in rows}):
        raise ValueError(f"Duplicate paths in {path}")
    return {r["path"]: r for r in rows}


def matrix(rows: list[dict[str, str]]) -> np.ndarray:
    p = np.array([[float(r[f"p_{c}"]) for c in CLASSES] for r in rows])
    if not np.isfinite(p).all() or np.any(p < 0) or not np.allclose(p.sum(1), 1, atol=2e-4):
        raise ValueError("Invalid probability matrix")
    return p


def temperature(p: np.ndarray, t: float) -> np.ndarray:
    z = np.log(np.clip(p, 1e-12, 1)) / t
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(p: np.ndarray, y: np.ndarray) -> float:
    fit = minimize_scalar(
        lambda z: log_loss(y, temperature(p, float(np.exp(z))), labels=np.arange(10)),
        bounds=(np.log(0.2), np.log(10)), method="bounded",
    )
    return float(np.exp(fit.x))


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float | int]:
    pred = p.argmax(axis=1)
    nonblue = np.array([CLASSES[int(v)] not in BLUE for v in y])
    predicted_blue = np.array([CLASSES[int(v)] in BLUE for v in pred])
    return {
        "n": len(y),
        "macro_f1": float(f1_score(y, pred, labels=np.arange(10), average="macro", zero_division=0)),
        "accuracy": float(np.mean(pred == y)),
        "costly_errors": int(np.sum(nonblue & predicted_blue)),
        "nonblue_n": int(np.sum(nonblue)),
        "nll": float(log_loss(y, p, labels=np.arange(10))),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=ROOT / "validation_partitions_v1.csv")
    ap.add_argument("--cnn", type=Path, default=ROOT / "v0.2-local-data/predictions.csv")
    ap.add_argument("--swin", type=Path, default=ROOT / "run3/swin_validation.csv")
    ap.add_argument("--embeddings", type=Path, default=ROOT / "v0.2-local-data/embeddings.npy")
    ap.add_argument("--output", type=Path, default=Path("results/ws2/hybrid_stack_v1.json"))
    args = ap.parse_args()

    manifest, cnn, swin = load(args.manifest), load(args.cnn), load(args.swin)
    if len(manifest) != 1556 or set(manifest) != set(swin):
        raise ValueError("Manifest and Swin validation paths differ")
    paths = sorted(manifest)
    for path in paths:
        if path not in cnn or any(
            manifest[path][key] != source[path][key]
            for source in (cnn, swin) for key in ("source", "group", "material")
        ):
            raise ValueError(f"Metadata mismatch: {path}")
    rows = list(cnn.values())  # original CSV order for embeddings
    idx = {row["path"]: i for i, row in enumerate(rows)}
    embedding = np.load(args.embeddings, mmap_mode="r", allow_pickle=False)
    if embedding.shape != (len(rows), 256):
        raise ValueError("Embedding row alignment mismatch")
    train_idx = np.array([i for i, row in enumerate(rows) if row["split"] == "train"])
    train_groups = {(rows[int(i)]["source"], rows[int(i)]["group"]) for i in train_idx}
    val_groups = {(manifest[p]["source"], manifest[p]["group"]) for p in paths}
    if train_groups & val_groups:
        raise ValueError("Physical-item leakage")
    train_y = np.array([CLASSES.index(rows[int(i)]["material"]) for i in train_idx])
    svm = make_pipeline(
        StandardScaler(),
        SVC(kernel="linear", C=0.1, class_weight="balanced", probability=True, random_state=42),
    )
    svm.fit(embedding[train_idx], train_y)
    pm = np.zeros((len(paths), 10))
    pm[:, svm.named_steps["svc"].classes_.astype(int)] = svm.predict_proba(
        embedding[[idx[p] for p in paths]]
    )
    pc = matrix([cnn[p] for p in paths])
    ps = matrix([swin[p] for p in paths])
    y = np.array([CLASSES.index(manifest[p]["material"]) for p in paths])
    partitions = np.array([manifest[p]["partition"] for p in paths])
    local = np.array([manifest[p]["source"] == "local" for p in paths])
    cal, meta, diag = (partitions == part for part in ("calibration", "meta_train", "diagnostic"))
    if tuple(int(np.sum(local & mask)) for mask in (cal, meta, diag)) != (99, 76, 72):
        raise ValueError("Unexpected canonical local partition counts")
    t_swin, t_svm = fit_temperature(ps[cal], y[cal]), fit_temperature(pm[cal], y[cal])
    ps, pm = temperature(ps, t_swin), temperature(pm, t_svm)

    # Select simple fusion weights on meta_train LOCAL only, never diagnostic.
    candidate_weights = (0.0, 0.25, 0.5, 0.75, 1.0)
    selected: dict[str, float] = {}
    methods: dict[str, np.ndarray] = {
        "convnext": pc,
        "swin": ps,
        "embedding_svm": pm,
        "cnn_swin_equal": 0.5 * (pc + ps),
        "cnn_svm_equal": 0.5 * (pc + pm),
    }
    for name, secondary in (("cnn_swin", ps), ("cnn_svm", pm)):
        trials = [
            (w, metrics(y[meta & local], (w * pc + (1 - w) * secondary)[meta & local]))
            for w in candidate_weights
        ]
        weight = max(trials, key=lambda item: (item[1]["macro_f1"], -item[1]["costly_errors"]))[0]
        selected[name] = weight
        methods[f"{name}_selected"] = weight * pc + (1 - weight) * secondary

    # Train the meta-model only on original validation meta_train rows.
    for name, features in (
        ("stack_cnn_swin", np.hstack((pc, ps))),
        ("stack_cnn_svm", np.hstack((pc, pm))),
        ("stack_all_three", np.hstack((pc, ps, pm))),
    ):
        model = LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced")
        model.fit(features[meta], y[meta])
        output = np.zeros((len(paths), 10))
        output[:, model.classes_] = model.predict_proba(features)
        methods[name] = output

    result = {
        "protocol": "ws2-v1-group-source-dominant-material",
        "counts": {name: {"all": int(mask.sum()), "local": int(np.sum(mask & local))}
                   for name, mask in (("calibration", cal), ("meta_train", meta), ("diagnostic", diag))},
        "temperatures": {"swin": t_swin, "embedding_svm": t_svm},
        "selected_cnn_weights": selected,
        "diagnostic": {
            name: {
                "local": metrics(y[diag & local], p[diag & local]),
                "all": metrics(y[diag], p[diag]),
            }
            for name, p in methods.items()
        },
        "limitations": [
            "This diagnostic subset has been viewed in previous experiments.",
            "Small local classes make macro-F1 unstable.",
            "SVM probability=True fits an internal training-set calibration, followed by held-out temperature calibration.",
            "CNN and embedding SVM share representation; this is not an independent handcrafted-feature ensemble.",
            "Costly errors are vision-only material proxies, not final disposal verdict errors.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("Counts:", json.dumps(result["counts"]))
    print("Temperatures:", json.dumps(result["temperatures"]))
    print("Selected CNN weights:", json.dumps(selected))
    print("Local diagnostic:")
    for name, value in result["diagnostic"].items():
        m = value["local"]
        print(f"{name:24s} F1={m['macro_f1']:.4f} accuracy={m['accuracy']:.4f} costly={m['costly_errors']}/{m['nonblue_n']}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
