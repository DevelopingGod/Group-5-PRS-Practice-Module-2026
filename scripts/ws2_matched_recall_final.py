"""WS2-20: final matched-blue-recall comparison and per-image audit.

Fits SVM on original train, temperatures on canonical calibration, stackers
on canonical meta_train. Chooses a common decision threshold on meta_train
local examples, then evaluates on canonical diagnostic local and public test.

uv run --with scikit-learn --with scipy python scripts/ws2_matched_recall_final.py
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ws2_public_ensemble import CLASSES, ROOT, fit_temperature, read, temp_scale, vector

BLUE_IDS = np.arange(5)
THRESHOLDS = np.round(np.arange(0.5, 1.001, 0.05), 2)


def gate_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float | int | None]:
    blue_truth = np.isin(y, BLUE_IDS)
    blue_probability = p[:, BLUE_IDS].sum(axis=1)
    blue_decision = blue_probability >= threshold
    harmful = int(np.sum(blue_decision & ~blue_truth))
    nonblue = int(np.sum(~blue_truth))
    eligible = int(np.sum(blue_truth))
    return {
        "n": len(y),
        "threshold": float(threshold),
        "costly_errors": harmful,
        "nonblue_n": nonblue,
        "costly_error_rate": harmful / nonblue if nonblue else None,
        "blue_recall": int(np.sum(blue_decision & blue_truth)) / eligible if eligible else None,
        "blue_precision": int(np.sum(blue_decision & blue_truth)) / int(np.sum(blue_decision)) if blue_decision.any() else None,
        "blue_recommendation_rate": float(np.mean(blue_decision)),
        "deferred_n": int(np.sum(~blue_decision)),
        "deferred_fraction": float(np.mean(~blue_decision)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", type=Path, default=ROOT / "v0.2-local-data/predictions.csv")
    ap.add_argument("--embeddings", type=Path, default=ROOT / "v0.2-local-data/embeddings.npy")
    ap.add_argument("--manifest", type=Path, default=ROOT / "validation_partitions_v1.csv")
    ap.add_argument("--swin-val", type=Path, default=ROOT / "run3/swin_validation.csv")
    ap.add_argument("--swin-public", type=Path, default=ROOT / "run3/pred_E2_public_plus_local_swin_t_public_test.csv")
    ap.add_argument("--output", type=Path, default=Path("results/ws2/matched_recall_final.json"))
    args = ap.parse_args()

    with args.predictions.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(set(r["path"] for r in rows)) != len(rows):
        raise ValueError("Duplicate CNN paths")
    cnn = {r["path"]: r for r in rows}
    manifest, swin_val, swin_public = (read(p) for p in (args.manifest, args.swin_val, args.swin_public))
    if len(manifest) != 1556 or set(manifest) != set(swin_val):
        raise ValueError("Validation manifest and Swin rows do not match")
    val_paths = sorted(manifest)
    public_paths = sorted(r["path"] for r in rows if r["split"] == "test" and r["source"] != "local")
    if set(public_paths) != set(swin_public):
        raise ValueError("Public test paths do not match")
    for path in val_paths:
        if path not in cnn or any(manifest[path][k] != other[path][k] for other in (cnn, swin_val) for k in ("material", "source", "group")):
            raise ValueError(f"Validation mismatch: {path}")
    for path in public_paths:
        if cnn[path]["material"] != swin_public[path]["true"]:
            raise ValueError(f"Public test label mismatch: {path}")

    embedding = np.load(args.embeddings, allow_pickle=False, mmap_mode="r")
    if embedding.shape != (len(rows), 256):
        raise ValueError("Embedding shape mismatch")
    row_idx = {r["path"]: i for i, r in enumerate(rows)}
    train_idx = np.array([i for i, r in enumerate(rows) if r["split"] == "train"])
    train_groups = {(rows[int(i)]["source"], rows[int(i)]["group"]) for i in train_idx}
    val_groups = {(manifest[p]["source"], manifest[p]["group"]) for p in val_paths}
    if train_groups & val_groups:
        raise ValueError("Train/validation group leakage")
    train_y = np.array([CLASSES.index(rows[int(i)]["material"]) for i in train_idx])
    svm = make_pipeline(
        StandardScaler(),
        SVC(kernel="linear", C=0.1, class_weight="balanced", probability=True, random_state=42),
    )
    svm.fit(embedding[train_idx], train_y)

    def svm_probs(paths: list[str]) -> np.ndarray:
        result = np.zeros((len(paths), 10))
        result[:, svm.named_steps["svc"].classes_.astype(int)] = svm.predict_proba(
            embedding[[row_idx[p] for p in paths]]
        )
        return result

    yv = np.array([CLASSES.index(manifest[p]["material"]) for p in val_paths])
    yp = np.array([CLASSES.index(cnn[p]["material"]) for p in public_paths])
    partition = np.array([manifest[p]["partition"] for p in val_paths])
    local = np.array([manifest[p]["source"] == "local" for p in val_paths])
    cal, meta, diag = (partition == name for name in ("calibration", "meta_train", "diagnostic"))
    if tuple(int((local & mask).sum()) for mask in (cal, meta, diag)) != (99, 76, 72):
        raise ValueError("Canonical partition counts differ")

    pcv, pcp = vector([cnn[p] for p in val_paths], True), vector([cnn[p] for p in public_paths], True)
    psv, psp = vector([swin_val[p] for p in val_paths], True), vector([swin_public[p] for p in public_paths], False)
    pmv, pmp = svm_probs(val_paths), svm_probs(public_paths)
    ts, tm = fit_temperature(psv[cal], yv[cal]), fit_temperature(pmv[cal], yv[cal])
    psv, psp = temp_scale(psv, ts), temp_scale(psp, ts)
    pmv, pmp = temp_scale(pmv, tm), temp_scale(pmp, tm)

    models: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "convnext": (pcv, pcp),
        "cnn_svm_equal": (0.5 * (pcv + pmv), 0.5 * (pcp + pmp)),
    }
    xv = np.hstack((pcv, psv, pmv))
    xp = np.hstack((pcp, psp, pmp))
    stack = LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced")
    stack.fit(xv[meta], yv[meta])
    pv_stack = np.zeros((len(val_paths), 10))
    pp_stack = np.zeros((len(public_paths), 10))
    pv_stack[:, stack.classes_.astype(int)] = stack.predict_proba(xv)
    pp_stack[:, stack.classes_.astype(int)] = stack.predict_proba(xp)
    models["stack_all_three"] = (pv_stack, pp_stack)

    def decision_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, object]:
        truth = np.isin(y, BLUE_IDS)
        prob = p[:, BLUE_IDS].sum(axis=1)
        approve = prob >= threshold
        n_bad, n_good = int((~truth).sum()), int(truth.sum())
        tp = int((approve & truth).sum())
        fp = int((approve & ~truth).sum())
        return {
            "n": len(y), "threshold": float(threshold),
            "costly_errors": fp, "nonblue_n": n_bad,
            "costly_error_rate": fp / n_bad if n_bad else None,
            "blue_recall": tp / n_good if n_good else None,
            "blue_precision": tp / int(approve.sum()) if approve.any() else None,
            "deferred_n": int((~approve).sum()),
            "deferred_fraction": float((~approve).mean()),
        }

    def choose_threshold(y: np.ndarray, p: np.ndarray, target: float) -> dict[str, object]:
        # Exact observed probability cutoffs, rather than a coarse 0.05 grid.
        # Maximise threshold while satisfying recall on meta_train LOCAL.
        truth = np.isin(y, BLUE_IDS)
        scores = p[:, BLUE_IDS].sum(axis=1)
        positives = np.sort(scores[truth])
        if len(positives) == 0:
            raise ValueError("No blue material examples in threshold selection")
        required = int(np.ceil(target * len(positives)))
        threshold = float(positives[len(positives) - required])
        result = decision_metrics(y, p, threshold)
        if result["blue_recall"] + 1e-12 < target:
            raise AssertionError("Recall target not reached")
        return result

    targets = (0.80, 0.85, 0.90)
    report = {
        "protocol": "ws2-v1-group-source-dominant-material",
        "n_meta_train_local": int((meta & local).sum()),
        "n_diagnostic_local": int((diag & local).sum()),
        "n_public_test": len(public_paths),
        "temperatures": {"swin": ts, "embedding_svm": tm},
        "selection": "Highest observed P_blue threshold meeting target recall on local meta_train",
        "models": {},
        "limitations": [
            "Blue-bin eligibility is approximated by five material classes; contamination and policy are excluded.",
            "Threshold selection on 76 local meta-training images is highly variable.",
            "Local diagnostic and public test data were examined previously.",
            "Thresholds can yield different recall on evaluation data; target recall is not guaranteed.",
            "Deferred predictions are not equivalent to correct general-waste verdicts.",
        ],
    }
    audit = []
    for name, (pv, pp) in models.items():
        report["models"][name] = {}
        for target in targets:
            selected = choose_threshold(yv[meta & local], pv[meta & local], target)
            threshold = float(selected["threshold"])
            key = f"{target:.2f}"
            report["models"][name][key] = {
                "selection_local": selected,
                "diagnostic_local": decision_metrics(yv[diag & local], pv[diag & local], threshold),
                "public_test": decision_metrics(yp, pp, threshold),
            }
            for scope, paths, y, p, mask in (
                ("diagnostic_local", val_paths, yv, pv, diag & local),
                ("public_test", public_paths, yp, pp, np.ones(len(public_paths), dtype=bool)),
            ):
                for i in np.flatnonzero(mask):
                    probability = float(p[i, BLUE_IDS].sum())
                    audit.append({
                        "model": name, "target_recall": key, "split": scope,
                        "path": paths[i], "physical_group": (
                            manifest[paths[i]]["source"] + "::" + manifest[paths[i]]["group"]
                            if scope == "diagnostic_local" else cnn[paths[i]]["source"] + "::" + cnn[paths[i]]["group"]
                        ),
                        "true_material": CLASSES[int(y[i])],
                        "blue_material_truth": int(y[i] in BLUE_IDS),
                        "p_blue": f"{probability:.8f}",
                        "threshold": f"{threshold:.8f}",
                        "approved": int(probability >= threshold),
                        "costly_error": int(probability >= threshold and y[i] not in BLUE_IDS),
                    })

    # Paired decision audit: same images, fixed thresholds chosen independently
    # on meta_train. Counts reveal actual rescued vs introduced harmful errors.
    for target in targets:
        key = f"{target:.2f}"
        for scope in ("diagnostic_local", "public_test"):
            subset = [r for r in audit if r["target_recall"] == key and r["split"] == scope]
            by_model = {
                name: {r["path"]: r for r in subset if r["model"] == name}
                for name in models
            }
            baseline = by_model["convnext"]
            for name in ("cnn_svm_equal", "stack_all_three"):
                candidate = by_model[name]
                if set(candidate) != set(baseline):
                    raise ValueError("Unmatched audit paths")
                prevented = sum(
                    baseline[p]["costly_error"] == 1 and candidate[p]["costly_error"] == 0
                    for p in baseline
                )
                introduced = sum(
                    baseline[p]["costly_error"] == 0 and candidate[p]["costly_error"] == 1
                    for p in baseline
                )
                report["models"][name][key][scope]["versus_convnext"] = {
                    "harmful_prevented": prevented, "harmful_introduced": introduced,
                    "net_harmful_prevented": prevented - introduced,
                }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    audit_path = args.output.with_name("matched_recall_audit.csv")
    with audit_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit[0]))
        writer.writeheader()
        writer.writerows(audit)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Local diagnostic: {int((diag & local).sum())}; public test: {len(public_paths)}")
    for target in targets:
        key = f"{target:.2f}"
        print(f"\\nTarget local meta-training blue recall: {key}")
        for name in models:
            result = report["models"][name][key]
            m, p = result["diagnostic_local"], result["public_test"]
            print(
                f"{name:18s} threshold={result['selection_local']['threshold']:.4f} "
                f"LOCAL costly={m['costly_errors']}/{m['nonblue_n']} recall={m['blue_recall']:.3f} defer={m['deferred_fraction']:.3f} | "
                f"PUBLIC costly={p['costly_errors']}/{p['nonblue_n']} recall={p['blue_recall']:.3f} defer={p['deferred_fraction']:.3f}"
            )
    print(f"Saved: {args.output}")
    print(f"Audit: {audit_path}")
    return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Local diagnostic: {int((diag & local).sum())}; public test: {len(public_paths)}")
    for name, result in report["models"].items():
        print(f"\n{name}: selected threshold {result['selected_threshold']:.2f}")
        for split in ("local_diagnostic", "public_test"):
            m = result[split]
            print(
                f"  {split}: costly={m['costly_errors']}/{m['nonblue_n']} "
                f"blue_recall={m['blue_recall']:.3f} deferred={m['deferred_fraction']:.3f}"
            )
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
