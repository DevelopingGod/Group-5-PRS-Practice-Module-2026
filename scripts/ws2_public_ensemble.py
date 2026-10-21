"""WS2-15: frozen hybrid/stacking evaluation on public test images.

Training: original train embeddings only. Calibration: canonical validation
calibration partition. Meta-model fitting and weight selection: canonical
validation meta_train partition. Evaluation: public test, never used for fitting.

Requires Run 3 Swin public test CSV in data/ws2/run3/.
uv run --with scikit-learn --with scipy python scripts/ws2_public_ensemble.py
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, log_loss, precision_recall_fscore_support
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
BLUE = set(CLASSES[:5])
ROOT = Path("data/ws2")


def read(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != len({r["path"] for r in rows}):
        raise ValueError(f"Duplicate paths: {path}")
    return {r["path"]: r for r in rows}


def vector(rows: list[dict[str, str]], prefixed: bool) -> np.ndarray:
    p = np.asarray(
        [[float(row[("p_" if prefixed else "") + c]) for c in CLASSES] for row in rows],
        dtype=np.float64,
    )
    if not np.isfinite(p).all() or np.any(p < 0) or not np.allclose(p.sum(1), 1, atol=2e-4):
        raise ValueError("Invalid probabilities")
    return p


def temp_scale(p: np.ndarray, t: float) -> np.ndarray:
    z = np.log(np.clip(p, 1e-12, 1)) / t
    z -= z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def fit_temperature(p: np.ndarray, y: np.ndarray) -> float:
    result = minimize_scalar(
        lambda z: log_loss(y, temp_scale(p, float(np.exp(z))), labels=np.arange(10)),
        bounds=(np.log(0.2), np.log(10.0)),
        method="bounded",
    )
    return float(np.exp(result.x))


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, object]:
    pred = p.argmax(1)
    precision, recall, f1, support = precision_recall_fscore_support(
        y, pred, labels=np.arange(10), zero_division=0
    )
    nonblue = np.array([CLASSES[int(i)] not in BLUE for i in y])
    predblue = np.array([CLASSES[int(i)] in BLUE for i in pred])
    return {
        "n": len(y),
        "macro_f1": float(
            f1_score(y, pred, labels=np.arange(10), average="macro", zero_division=0)
        ),
        "accuracy": float(np.mean(y == pred)),
        "correct": int(np.sum(y == pred)),
        "costly_error_count": int(np.sum(nonblue & predblue)),
        "nonblue_count": int(nonblue.sum()),
        "costly_error_rate": float(predblue[nonblue].mean()) if nonblue.any() else None,
        "nll": float(log_loss(y, p, labels=np.arange(10))),
        "per_class": {
            name: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, name in enumerate(CLASSES)
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", type=Path, default=ROOT / "v0.2-local-data/predictions.csv")
    ap.add_argument("--embeddings", type=Path, default=ROOT / "v0.2-local-data/embeddings.npy")
    ap.add_argument("--manifest", type=Path, default=ROOT / "validation_partitions_v1.csv")
    ap.add_argument("--swin-val", type=Path, default=ROOT / "run3/swin_validation.csv")
    ap.add_argument(
        "--swin-public",
        type=Path,
        default=ROOT / "run3/pred_E2_public_plus_local_swin_t_public_test.csv",
    )
    ap.add_argument("--output", type=Path, default=Path("results/ws2/public_ensemble_v1.json"))
    args = ap.parse_args()

    with args.predictions.open(newline="", encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))
    if len({r["path"] for r in all_rows}) != len(all_rows):
        raise ValueError("Duplicate paths in CNN export")
    cnn = {r["path"]: r for r in all_rows}
    emb = np.load(args.embeddings, allow_pickle=False, mmap_mode="r")
    if emb.shape != (len(all_rows), 256):
        raise ValueError("Embedding/CSV row mismatch")
    idx = {r["path"]: i for i, r in enumerate(all_rows)}
    manifest, swin_val, swin_public = (
        read(p) for p in (args.manifest, args.swin_val, args.swin_public)
    )
    if len(manifest) != 1556 or set(manifest) != set(swin_val):
        raise ValueError("Canonical validation and Swin predictions mismatch")
    val_paths = sorted(manifest)
    public_paths = sorted(
        r["path"] for r in all_rows if r["split"] == "test" and r["source"] != "local"
    )
    if set(public_paths) != set(swin_public):
        raise ValueError(
            f"Public test path mismatch: CNN={len(public_paths)} Swin={len(swin_public)}. "
            "Check that the Swin CSV is Run 3 data v2."
        )
    if set(val_paths) & set(public_paths):
        raise ValueError("Validation/test overlap")
    for path in val_paths:
        if path not in cnn or any(
            manifest[path][key] != other[path][key]
            for other in (cnn, swin_val)
            for key in ("material", "source", "group")
        ):
            raise ValueError(f"Validation metadata mismatch: {path}")
    for path in public_paths:
        if cnn[path]["material"] != swin_public[path]["true"]:
            raise ValueError(f"Public test label mismatch: {path}")
    val_y = np.array([CLASSES.index(manifest[p]["material"]) for p in val_paths])
    test_y = np.array([CLASSES.index(cnn[p]["material"]) for p in public_paths])
    partition = np.array([manifest[p]["partition"] for p in val_paths])
    local = np.array([manifest[p]["source"] == "local" for p in val_paths])
    cal, meta = partition == "calibration", partition == "meta_train"
    if tuple(
        int(np.sum(local & (partition == s))) for s in ("calibration", "meta_train", "diagnostic")
    ) != (99, 76, 72):
        raise ValueError("Canonical local partition counts changed")
    train_ids = np.array([i for i, r in enumerate(all_rows) if r["split"] == "train"])
    train_groups = {(all_rows[int(i)]["source"], all_rows[int(i)]["group"]) for i in train_ids}
    val_groups = {(manifest[p]["source"], manifest[p]["group"]) for p in val_paths}
    if train_groups & val_groups:
        raise ValueError("Physical item leakage")
    train_y = np.array([CLASSES.index(all_rows[int(i)]["material"]) for i in train_ids])

    # Reproduce WS2-14 SVM configuration, including its probability interface.
    svm = make_pipeline(
        StandardScaler(),
        SVC(kernel="linear", C=0.1, class_weight="balanced", probability=True, random_state=42),
    )
    svm.fit(emb[train_ids], train_y)

    def svm_proba(paths: list[str]) -> np.ndarray:
        result = np.zeros((len(paths), 10), dtype=np.float64)
        result[:, svm.named_steps["svc"].classes_.astype(int)] = svm.predict_proba(
            emb[[idx[p] for p in paths]]
        )
        return result

    pc_val = vector([cnn[p] for p in val_paths], True)
    pc_test = vector([cnn[p] for p in public_paths], True)
    ps_val_raw = vector([swin_val[p] for p in val_paths], True)
    ps_test_raw = vector([swin_public[p] for p in public_paths], False)
    pm_val_raw = svm_proba(val_paths)
    pm_test_raw = svm_proba(public_paths)
    t_swin = fit_temperature(ps_val_raw[cal], val_y[cal])
    t_svm = fit_temperature(pm_val_raw[cal], val_y[cal])
    ps_val, ps_test = temp_scale(ps_val_raw, t_swin), temp_scale(ps_test_raw, t_swin)
    pm_val, pm_test = temp_scale(pm_val_raw, t_svm), temp_scale(pm_test_raw, t_svm)

    # Select weights on local meta_train only, as in WS2-14.
    selected: dict[str, float] = {}
    methods: dict[str, np.ndarray] = {
        "convnext": pc_test,
        "swin": ps_test,
        "embedding_svm": pm_test,
        "cnn_swin_equal": 0.5 * (pc_test + ps_test),
        "cnn_svm_equal": 0.5 * (pc_test + pm_test),
    }
    for name, secondary_val, secondary_test in (
        ("cnn_swin", ps_val, ps_test),
        ("cnn_svm", pm_val, pm_test),
    ):
        trials = []
        for weight in (0.0, 0.25, 0.5, 0.75, 1.0):
            p = weight * pc_val + (1 - weight) * secondary_val
            m = metrics(val_y[meta & local], p[meta & local])
            trials.append((weight, m))
        weight = max(
            trials, key=lambda item: (item[1]["macro_f1"], -item[1]["costly_error_count"])
        )[0]
        selected[name] = weight
        methods[name + "_selected"] = weight * pc_test + (1 - weight) * secondary_test

    for name, x_val, x_test in (
        ("stack_cnn_swin", np.hstack((pc_val, ps_val)), np.hstack((pc_test, ps_test))),
        ("stack_cnn_svm", np.hstack((pc_val, pm_val)), np.hstack((pc_test, pm_test))),
        (
            "stack_all_three",
            np.hstack((pc_val, ps_val, pm_val)),
            np.hstack((pc_test, ps_test, pm_test)),
        ),
    ):
        stacker = LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced")
        stacker.fit(x_val[meta], val_y[meta])
        p = np.zeros((len(public_paths), 10), dtype=np.float64)
        p[:, stacker.classes_.astype(int)] = stacker.predict_proba(x_test)
        methods[name] = p

    report = {
        "protocol": "ws2-v1-group-source-dominant-material",
        "evaluation": "public_test; exploratory, test previously inspected",
        "n_public_test": len(public_paths),
        "calibration_n": int(cal.sum()),
        "meta_train_n": int(meta.sum()),
        "temperatures": {"swin": t_swin, "embedding_svm": t_svm},
        "selected_cnn_weights": selected,
        "results": {name: metrics(test_y, p) for name, p in methods.items()},
        "limitations": [
            "Test set was previously inspected; do not tune models on these results.",
            "ConvNeXt uses the calibrated WS1 export while Swin uses Run 3 raw softmax with held-out temperature.",
            "SVM is a CNN-embedding classifier, not an independent handcrafted feature model.",
            "Public test macro-F1 is sensitive to very low-support classes.",
            "Costly errors are material-based proxies, not end-to-end policy verdicts.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Public test:", len(public_paths), "images")
    print("Temperatures:", json.dumps(report["temperatures"]))
    print("Selected CNN weights:", json.dumps(selected))
    for name, result in report["results"].items():
        print(
            f"{name:24s} F1={result['macro_f1']:.4f} "
            f"accuracy={result['accuracy']:.4f} "
            f"costly={result['costly_error_count']}/{result['nonblue_count']}"
        )
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
