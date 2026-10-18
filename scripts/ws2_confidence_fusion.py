"""WS2: fixed-rule fusion ablation on group-held-out local validation images.

No fitting of weights on the diagnostic subset. Swin temperature and SVM
probabilities are not assumed calibrated here; only ConvNeXt and Swin are used.
Compare fixed confidence gating with fixed voting on the SAME subset.

uv run python scripts/ws2_confidence_fusion.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

CLASSES = (
    "paper", "plastic", "glass", "metal", "beverage_carton",
    "styrofoam", "e_waste", "textile", "food_waste", "other_general",
)
BLUE = set(CLASSES[:5])


def load(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len({r["path"] for r in rows}):
        raise ValueError(f"Duplicate image paths: {path}")
    return {r["path"]: r for r in rows}


def vector(row: dict[str, str]) -> np.ndarray:
    p = np.array([float(row[f"p_{c}"]) for c in CLASSES])
    if not np.isfinite(p).all() or not np.isclose(p.sum(), 1, atol=2e-4):
        raise ValueError("Invalid probabilities")
    return p


def temperature(p: np.ndarray, t: float) -> np.ndarray:
    z = np.log(np.clip(p, 1e-12, 1)) / t
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def score(y: np.ndarray, p: np.ndarray) -> dict[str, float | int]:
    from sklearn.metrics import f1_score

    pred = p.argmax(axis=1)
    nonblue = np.array([CLASSES[int(i)] not in BLUE for i in y])
    predblue = np.array([CLASSES[int(i)] in BLUE for i in pred])
    return {
        "n": int(len(y)),
        "macro_f1": float(f1_score(y, pred, labels=list(range(10)), average="macro", zero_division=0)),
        "accuracy": float(np.mean(pred == y)),
        "costly_error_count": int((nonblue & predblue).sum()),
        "nonblue_count": int(nonblue.sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cnn", type=Path, default=Path("data/ws2/v0.2-local-data/predictions.csv"))
    ap.add_argument("--swin", type=Path, default=Path("data/ws2/run3/swin_validation.csv"))
    ap.add_argument("--temperature", type=float, default=0.6862418803557301)
    ap.add_argument("--output", type=Path, default=Path("results/ws2/confidence_fusion.json"))
    args = ap.parse_args()
    cnn, swin = load(args.cnn), load(args.swin)
    paths = sorted(
        p for p, r in swin.items()
        if r["source"] == "local" and r["split"] == "val"
    )
    if len(paths) != 247:
        raise ValueError(f"Expected 247 local validation rows, got {len(paths)}")
    for path in paths:
        if path not in cnn or any(
            cnn[path][key] != swin[path][key] for key in ("material", "group", "source", "split")
        ):
            raise ValueError(f"Metadata mismatch: {path}")
    # Match the earlier WS2-09 held-out partition (40/30/30 within source,
    # sorted unique groups, seeded permutation).
    all_swin = [r for r in swin.values() if r["split"] == "val"]
    groups_by_source: dict[str, list[str]] = {}
    for row in all_swin:
        key = row["source"] + "::" + row["group"]
        groups_by_source.setdefault(row["source"], []).append(key)
    rng = np.random.default_rng(42)
    partition: dict[str, int] = {}
    for groups in groups_by_source.values():
        unique = sorted(set(groups))
        for j, idx in enumerate(rng.permutation(len(unique))):
            fraction = j / len(unique)
            partition[unique[int(idx)]] = 0 if fraction < 0.4 else (1 if fraction < 0.7 else 2)
    paths = [p for p in paths if partition[swin[p]["source"] + "::" + swin[p]["group"]] == 2]
    y = np.array([CLASSES.index(cnn[p]["material"]) for p in paths])
    pc = np.stack([vector(cnn[p]) for p in paths])
    ps = temperature(np.stack([vector(swin[p]) for p in paths]), args.temperature)

    # Predefined rules; no threshold search on this evaluation subset.
    conf_c, conf_s = pc.max(axis=1), ps.max(axis=1)
    selected = np.where((conf_s > conf_c + 0.10)[:, None], ps, pc)
    disagreement = pc.argmax(axis=1) != ps.argmax(axis=1)
    cautious = np.where(
        (disagreement & (conf_c < 0.70))[:, None], 0.5 * (pc + ps), pc
    )
    methods = {
        "convnext": pc,
        "swin": ps,
        "equal_vote": 0.5 * (pc + ps),
        "confidence_margin_0_10": selected,
        "disagreement_low_confidence": cautious,
    }
    report = {
        "split": "WS2-09 local validation development holdout",
        "n": len(paths),
        "swin_temperature": args.temperature,
        "methods": {name: score(y, p) for name, p in methods.items()},
        "limitations": [
            "Rules are predefined exploratory baselines, not optimised ensemble weights.",
            "Confidence from different models may remain imperfectly calibrated.",
            "This validation holdout has already been examined in prior experiments.",
            "No independent final-test claim can be made from these results.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
