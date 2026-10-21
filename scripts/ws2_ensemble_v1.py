"""Compare fixed CNN fusion baselines on canonical WS2 validation partitions.

Uses calibration for Swin temperature, meta_train for optional weight selection,
and diagnostic for reporting. Never re-partitions images or touches test rows.

uv run --with scikit-learn --with scipy python scripts/ws2_ensemble_v1.py
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import f1_score, log_loss

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


def read(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len({r["path"] for r in rows}):
        raise ValueError(f"Duplicate paths in {path}")
    return {r["path"]: r for r in rows}


def probabilities(rows: list[dict[str, str]]) -> np.ndarray:
    p = np.array([[float(r[f"p_{c}"]) for c in CLASSES] for r in rows])
    if not np.isfinite(p).all() or not np.allclose(p.sum(axis=1), 1, atol=2e-4):
        raise ValueError("Invalid probabilities")
    return p


def scale(p: np.ndarray, t: float) -> np.ndarray:
    z = np.log(np.clip(p, 1e-12, 1)) / t
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float | int]:
    pred = p.argmax(axis=1)
    nonblue = np.array([CLASSES[int(i)] not in BLUE for i in y])
    blue = np.array([CLASSES[int(i)] in BLUE for i in pred])
    return {
        "n": len(y),
        "macro_f1": float(
            f1_score(y, pred, labels=np.arange(10), average="macro", zero_division=0)
        ),
        "accuracy": float(np.mean(pred == y)),
        "costly_error_count": int(np.sum(nonblue & blue)),
        "nonblue_count": int(np.sum(nonblue)),
        "costly_error_rate": float(np.mean(blue[nonblue])) if nonblue.any() else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=Path("data/ws2/validation_partitions_v1.csv"))
    ap.add_argument(
        "--convnext", type=Path, default=Path("data/ws2/v0.2-local-data/predictions.csv")
    )
    ap.add_argument("--swin", type=Path, default=Path("data/ws2/run3/swin_validation.csv"))
    ap.add_argument("--output", type=Path, default=Path("results/ws2/ensemble_v1.json"))
    args = ap.parse_args()

    manifest, conv, swin = read(args.manifest), read(args.convnext), read(args.swin)
    if len(manifest) != 1556 or set(manifest) != set(swin):
        raise ValueError("Canonical manifest must match all 1556 Swin validation rows")
    paths = sorted(manifest)
    for path in paths:
        if path not in conv or any(
            manifest[path][k] != conv[path][k] or manifest[path][k] != swin[path][k]
            for k in ("source", "group", "material")
        ):
            raise ValueError(f"Metadata mismatch: {path}")
    y = np.array([CLASSES.index(manifest[p]["material"]) for p in paths])
    pc = probabilities([conv[p] for p in paths])
    ps_raw = probabilities([swin[p] for p in paths])
    partition = np.array([manifest[p]["partition"] for p in paths])
    local = np.array([manifest[p]["source"] == "local" for p in paths])
    cal, meta, diag = (partition == name for name in ("calibration", "meta_train", "diagnostic"))
    if len(set(paths)) != len(paths) or not all(mask.any() for mask in (cal, meta, diag)):
        raise ValueError("Invalid partitions")
    groups = [manifest[p]["source"] + "::" + manifest[p]["group"] for p in paths]
    if any(
        len({partition[i] for i, g in enumerate(groups) if g == group}) != 1
        for group in set(groups)
    ):
        raise ValueError("Item group crosses partitions")

    opt = minimize_scalar(
        lambda z: log_loss(y[cal], scale(ps_raw[cal], float(np.exp(z))), labels=np.arange(10)),
        bounds=(np.log(0.2), np.log(10)),
        method="bounded",
    )
    t = float(np.exp(opt.x))
    ps = scale(ps_raw, t)
    weights = (0.0, 0.25, 0.5, 0.75, 1.0)
    # Weight selection is on meta_train only; never select on diagnostic.
    scores = []
    for w in weights:
        mixed = w * pc + (1 - w) * ps
        scores.append((w, metrics(y[meta & local], mixed[meta & local])))
    selected = max(scores, key=lambda item: (item[1]["macro_f1"], -item[1]["costly_error_count"]))[
        0
    ]
    methods = {
        "convnext": pc,
        "swin_calibrated": ps,
        "equal_vote": 0.5 * pc + 0.5 * ps,
        "meta_selected_vote": selected * pc + (1 - selected) * ps,
    }
    report = {
        "protocol": "ws2-v1-group-source-dominant-material",
        "counts": {
            name: {"all": int(mask.sum()), "local": int((mask & local).sum())}
            for name, mask in (("calibration", cal), ("meta_train", meta), ("diagnostic", diag))
        },
        "swin_temperature": t,
        "meta_selected_convnext_weight": selected,
        "meta_weight_trials": [{"convnext_weight": w, **m} for w, m in scores],
        "diagnostic": {
            name: {
                "local": metrics(y[diag & local], p[diag & local]),
                "all": metrics(y[diag], p[diag]),
            }
            for name, p in methods.items()
        },
        "limitations": [
            "Previously inspected validation images; diagnostic is a development benchmark.",
            "Small local diagnostic class counts cause high metric variance.",
            "Vision-only material proxy costly errors, not final policy verdict errors.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "counts": report["counts"],
                "swin_temperature": t,
                "selected_weight": selected,
                "local_diagnostic": {
                    name: value["local"] for name, value in report["diagnostic"].items()
                },
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
