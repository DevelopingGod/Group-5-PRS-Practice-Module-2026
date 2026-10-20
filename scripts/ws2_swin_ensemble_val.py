"""Compare ConvNeXt-T and Swin-T on the original validation images.

Run:
    uv run --with scikit-learn python scripts/ws2_swin_ensemble_val.py

Swin-T is temperature-scaled on a separate calibration subset of validation
groups. Weights are selected on disjoint selection groups; results are
development-set estimates, not independent test estimates.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import accuracy_score, f1_score, log_loss

CLASSES = (
    "paper", "plastic", "glass", "metal", "beverage_carton",
    "styrofoam", "e_waste", "textile", "food_waste", "other_general",
)
RECYCLABLE = {"paper", "plastic", "glass", "metal", "beverage_carton"}
WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)


def read_predictions(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len({row["path"] for row in rows}):
        raise ValueError(f"Duplicate image paths in {path}")
    return {row["path"]: row for row in rows}


def probabilities(rows: list[dict[str, str]]) -> np.ndarray:
    values = np.asarray(
        [[float(row[f"p_{label}"]) for label in CLASSES] for row in rows],
        dtype=np.float64,
    )
    if not np.isfinite(values).all() or not np.allclose(values.sum(axis=1), 1, atol=2e-4):
        raise ValueError("Invalid probability matrix")
    return values


def temperature_scale(probs: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.log(np.clip(probs, 1e-12, 1.0)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def evaluate(y: np.ndarray, probs: np.ndarray) -> dict[str, float | int]:
    predicted = probs.argmax(axis=1)
    non_blue = np.asarray([CLASSES[int(i)] not in RECYCLABLE for i in y])
    predicted_blue = np.asarray([CLASSES[int(i)] in RECYCLABLE for i in predicted])
    return {
        "n": len(y),
        "macro_f1": float(f1_score(y, predicted, labels=list(range(10)), average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y, predicted)),
        "costly_error_rate": float(predicted_blue[non_blue].mean()) if non_blue.any() else 0.0,
        "costly_error_count": int((predicted_blue & non_blue).sum()),
        "non_blue_count": int(non_blue.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--convnext", type=Path, default=Path("data/ws2/v0.2-local-data/predictions.csv"))
    parser.add_argument("--swin", type=Path, default=Path("data/ws2/run3/swin_validation.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/ws2/swin_ensemble_validation.json"))
    args = parser.parse_args()

    conv = read_predictions(args.convnext)
    swin = read_predictions(args.swin)
    common = sorted(
        path for path in swin
        if path in conv and conv[path]["split"] == "val" and swin[path]["split"] == "val"
    )
    if len(common) != len(swin):
        raise ValueError("Swin validation export must align exactly to ConvNeXt validation rows")
    for path in common:
        if any(conv[path][key] != swin[path][key] for key in ("material", "source", "group")):
            raise ValueError(f"Metadata mismatch: {path}")
    rows_c = [conv[path] for path in common]
    rows_s = [swin[path] for path in common]
    y = np.asarray([CLASSES.index(row["material"]) for row in rows_c])
    pc = probabilities(rows_c)
    ps = probabilities(rows_s)
    # Deterministic group-level 50/50 partition, stratified approximately by source.
    groups = sorted({(row["source"], row["group"]) for row in rows_c})
    rng = np.random.default_rng(42)
    by_source: dict[str, list[tuple[str, str]]] = {}
    for group in groups:
        by_source.setdefault(group[0], []).append(group)
    calibration_groups: set[tuple[str, str]] = set()
    for group_list in by_source.values():
        indices = rng.permutation(len(group_list))[: max(1, len(group_list) // 2)]
        calibration_groups.update(group_list[int(i)] for i in indices)
    is_cal = np.asarray(
        [(row["source"], row["group"]) in calibration_groups for row in rows_c]
    )
    is_select = ~is_cal
    if not is_cal.any() or not is_select.any():
        raise ValueError("Calibration and selection subsets must both be nonempty")
    fit = minimize_scalar(
        lambda t: log_loss(
            y[is_cal], temperature_scale(ps[is_cal], float(np.exp(t))),
            labels=list(range(10)),
        ),
        bounds=(np.log(0.2), np.log(10.0)),
        method="bounded",
    )
    temp = float(np.exp(fit.x))
    ps_calibrated = temperature_scale(ps, temp)
    report: dict[str, object] = {
        "validation_images": len(common),
        "calibration_images": int(is_cal.sum()),
        "selection_images": int(is_select.sum()),
        "swin_temperature": temp,
        "local_selection": {},
        "all_selection": {},
        "disagreement": {},
        "note": "Exploratory validation split; fixed test predictions not used for weight selection.",
    }
    for name, mask in (
        ("local_selection", is_select & np.asarray([r["source"] == "local" for r in rows_c])),
        ("all_selection", is_select),
    ):
        yy = y[mask]
        results = []
        for weight in WEIGHTS:
            mixed = weight * pc[mask] + (1 - weight) * ps_calibrated[mask]
            results.append({"convnext_weight": weight, **evaluate(yy, mixed)})
        report[name] = results
        if name == "local_selection":
            cc = pc[mask].argmax(axis=1) == yy
            ss = ps_calibrated[mask].argmax(axis=1) == yy
            report["disagreement"] = {
                "both_correct": int((cc & ss).sum()),
                "convnext_only": int((cc & ~ss).sum()),
                "swin_only": int((~cc & ss).sum()),
                "both_wrong": int((~cc & ~ss).sum()),
            }
    selection = report["local_selection"]
    best = max(
        selection,
        key=lambda r: (float(r["macro_f1"]), -float(r["costly_error_rate"]), float(r["accuracy"])),
    )
    report["selected_weight"] = best["convnext_weight"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Matched validation images: {len(common)}")
    print(f"Calibration: {is_cal.sum()} | selection: {is_select.sum()} | Swin T={temp:.4f}")
    print("Local selection comparisons (ConvNeXt weight / macro-F1 / accuracy / costly-error):")
    for result in selection:
        print(
            f"  {result['convnext_weight']:.2f}: {result['macro_f1']:.4f} / "
            f"{result['accuracy']:.4f} / {result['costly_error_rate']:.4f} "
            f"({result['costly_error_count']}/{result['non_blue_count']})"
        )
    print("Local disagreement:", json.dumps(report["disagreement"]))
    print(f"Selected validation weight: {report['selected_weight']}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
