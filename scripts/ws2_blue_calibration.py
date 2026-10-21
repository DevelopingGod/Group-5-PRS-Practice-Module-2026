"""WS2-19: diagnose binary blue-material calibration and risk-coverage.

Fits SVM on original train, temperatures on canonical calibration, stackers
on canonical meta_train. Chooses a common decision threshold on meta_train
local examples, then evaluates on canonical diagnostic local and public test.

uv run --with scikit-learn --with scipy python scripts/ws2_blue_calibration.py
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
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
        "blue_precision": int(np.sum(blue_decision & blue_truth)) / int(np.sum(blue_decision))
        if blue_decision.any()
        else None,
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
    ap.add_argument(
        "--swin-public",
        type=Path,
        default=ROOT / "run3/pred_E2_public_plus_local_swin_t_public_test.csv",
    )
    ap.add_argument("--output", type=Path, default=Path("results/ws2/blue_calibration_v1.json"))
    args = ap.parse_args()

    with args.predictions.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(set(r["path"] for r in rows)) != len(rows):
        raise ValueError("Duplicate CNN paths")
    cnn = {r["path"]: r for r in rows}
    manifest, swin_val, swin_public = (
        read(p) for p in (args.manifest, args.swin_val, args.swin_public)
    )
    if len(manifest) != 1556 or set(manifest) != set(swin_val):
        raise ValueError("Validation manifest and Swin rows do not match")
    val_paths = sorted(manifest)
    public_paths = sorted(
        r["path"] for r in rows if r["split"] == "test" and r["source"] != "local"
    )
    if set(public_paths) != set(swin_public):
        raise ValueError("Public test paths do not match")
    for path in val_paths:
        if path not in cnn or any(
            manifest[path][k] != other[path][k]
            for other in (cnn, swin_val)
            for k in ("material", "source", "group")
        ):
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

    pcv, pcp = (
        vector([cnn[p] for p in val_paths], True),
        vector([cnn[p] for p in public_paths], True),
    )
    psv, psp = (
        vector([swin_val[p] for p in val_paths], True),
        vector([swin_public[p] for p in public_paths], False),
    )
    pmv, pmp = svm_probs(val_paths), svm_probs(public_paths)
    ts, tm = fit_temperature(psv[cal], yv[cal]), fit_temperature(pmv[cal], yv[cal])
    psv, psp = temp_scale(psv, ts), temp_scale(psp, ts)
    pmv, pmp = temp_scale(pmv, tm), temp_scale(pmp, tm)

    models: dict[str, tuple[np.ndarray, np.ndarray]] = {"convnext": (pcv, pcp)}
    for name, xv, xp in (
        ("stack_cnn_svm", np.hstack((pcv, pmv)), np.hstack((pcp, pmp))),
        ("stack_all_three", np.hstack((pcv, psv, pmv)), np.hstack((pcp, psp, pmp))),
    ):
        stack = LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced")
        stack.fit(xv[meta], yv[meta])
        v = np.zeros((len(val_paths), 10))
        p = np.zeros((len(public_paths), 10))
        v[:, stack.classes_.astype(int)] = stack.predict_proba(xv)
        p[:, stack.classes_.astype(int)] = stack.predict_proba(xp)
        models[name] = (v, p)

    # Select one threshold per model on meta_train local, with a safety-first
    # objective and a nontrivial recycling-recall constraint. This is a
    # predefined decision rule, not a post-hoc public-test optimisation.
    min_blue_recall = 0.70
    report: dict[str, object] = {
        "protocol": "ws2-v1-group-source-dominant-material",
        "threshold_candidates": THRESHOLDS.tolist(),
        "min_meta_blue_recall": min_blue_recall,
        "calibration_n": int(cal.sum()),
        "meta_train_local_n": int((meta & local).sum()),
        "diagnostic_local_n": int((diag & local).sum()),
        "public_test_n": len(public_paths),
        "temperatures": {"swin": ts, "embedding_svm": tm},
        "models": {},
        "limitations": [
            "Eligibility is a material-only proxy; contamination and policy are not represented.",
            "Deferred items are not counted as correct general-waste recommendations.",
            "Public test has already been examined in earlier experiments.",
            "Threshold is selected on just 76 local meta-training images.",
            "No genuine OOD recall can be estimated from these known-class examples.",
        ],
    }
    for name, (pv, pp) in models.items():
        meta_trials = [
            gate_metrics(yv[meta & local], pv[meta & local], float(t)) for t in THRESHOLDS
        ]
        feasible = [
            r
            for r in meta_trials
            if r["blue_recall"] is not None and r["blue_recall"] >= min_blue_recall
        ]
        if not feasible:
            raise ValueError(f"No threshold satisfies minimum blue recall for {name}")
        chosen = min(
            feasible,
            key=lambda r: (r["costly_error_rate"], -r["blue_recall"], r["threshold"]),
        )
        threshold = float(chosen["threshold"])
        report["models"][name] = {
            "selected_threshold": threshold,
            "meta_train_local": chosen,
            "local_diagnostic": gate_metrics(yv[diag & local], pv[diag & local], threshold),
            "public_test": gate_metrics(yp, pp, threshold),
            "local_risk_coverage_curve": [
                gate_metrics(yv[diag & local], pv[diag & local], float(t)) for t in THRESHOLDS
            ],
            "public_risk_coverage_curve": [gate_metrics(yp, pp, float(t)) for t in THRESHOLDS],
        }

    # Diagnostic only: no calibration or threshold fitting on diagnostic/test labels.
    def calibration_report(y: np.ndarray, p: np.ndarray) -> dict[str, object]:
        truth = np.isin(y, BLUE_IDS).astype(int)
        prob = np.clip(p[:, BLUE_IDS].sum(axis=1), 0, 1)
        edges = np.linspace(0, 1, 11)
        bins = []
        ece = 0.0
        for j in range(10):
            mask = (prob >= edges[j]) & (prob < edges[j + 1] if j < 9 else prob <= 1)
            if not mask.any():
                continue
            mean_conf = float(prob[mask].mean())
            fraction = float(truth[mask].mean())
            ece += float(mask.mean()) * abs(mean_conf - fraction)
            bins.append(
                {"bin": j, "n": int(mask.sum()), "confidence": mean_conf, "frequency": fraction}
            )
        curve = []
        for t in np.linspace(0, 1, 21):
            accepted = prob >= t
            nonblue = truth == 0
            blue = truth == 1
            curve.append(
                {
                    "threshold": float(t),
                    "costly_errors": int((accepted & nonblue).sum()),
                    "nonblue_n": int(nonblue.sum()),
                    "costly_error_rate": float((accepted & nonblue).sum() / nonblue.sum())
                    if nonblue.any()
                    else None,
                    "blue_recall": float((accepted & blue).sum() / blue.sum())
                    if blue.any()
                    else None,
                    "deferred_fraction": float((~accepted).mean()),
                }
            )
        return {
            "n": len(y),
            "brier": float(brier_score_loss(truth, prob)),
            "ece_10_bin": float(ece),
            "average_precision": float(average_precision_score(truth, prob))
            if len(set(truth)) > 1
            else None,
            "roc_auc": float(roc_auc_score(truth, prob)) if len(set(truth)) > 1 else None,
            "bins": bins,
            "risk_coverage": curve,
        }

    # A fixed, group-aware paired bootstrap. Confidence intervals are descriptive,
    # not a license to select a new winner based on this diagnostic subset.
    def paired_bootstrap(
        y: np.ndarray,
        base: np.ndarray,
        candidate: np.ndarray,
        group_keys: list[str],
        seed: int = 42,
    ) -> dict[str, object]:
        unique = sorted(set(group_keys))
        group_indices = {g: np.flatnonzero(np.asarray(group_keys) == g) for g in unique}
        rng = np.random.default_rng(seed)
        differences = []
        for _ in range(1000):
            sampled = rng.choice(unique, size=len(unique), replace=True)
            ids = np.concatenate([group_indices[g] for g in sampled])
            truth = np.isin(y[ids], BLUE_IDS).astype(int)
            pb = base[ids][:, BLUE_IDS].sum(axis=1)
            pc = candidate[ids][:, BLUE_IDS].sum(axis=1)
            differences.append(float(brier_score_loss(truth, pc) - brier_score_loss(truth, pb)))
        lo, hi = np.percentile(differences, [2.5, 97.5])
        return {
            "metric": "candidate_minus_convnext_brier",
            "ci95": [float(lo), float(hi)],
            "n_groups": len(unique),
            "bootstrap_replicates": 1000,
        }

    # Include equal-weight hybrid, without altering any trained models.
    models["cnn_svm_equal"] = (0.5 * (pcv + pmv), 0.5 * (pcp + pmp))
    diagnostic_mask = diag & local
    diagnostic_groups = [
        manifest[path]["source"] + "::" + manifest[path]["group"]
        for path, keep in zip(val_paths, diagnostic_mask, strict=True)
        if keep
    ]
    analysis = {
        "local_diagnostic": {},
        "public_test": {},
        "paired_local_brier_difference": {},
        "limitations": [
            "Blue eligibility here is material-only, not contamination-aware policy verdict.",
            "Diagnostic and public test sets have been inspected in prior experiments.",
            "ECE is sensitive to bin choice and small class counts.",
            "Bootstrap groups are physical items; intervals describe sample uncertainty only.",
            "Stacker training uses meta_train; its in-sample meta_train predictions must not be used as independent calibration evaluation.",
        ],
    }
    for name, (pv, pp) in models.items():
        analysis["local_diagnostic"][name] = calibration_report(
            yv[diagnostic_mask], pv[diagnostic_mask]
        )
        analysis["public_test"][name] = calibration_report(yp, pp)
        if name != "convnext":
            analysis["paired_local_brier_difference"][name] = paired_bootstrap(
                yv[diagnostic_mask], pcv[diagnostic_mask], pv[diagnostic_mask], diagnostic_groups
            )
    report["blue_calibration_analysis"] = analysis
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
    print("\\nBlue-bin-material calibration diagnostics:")
    for name in models:
        local_metrics = analysis["local_diagnostic"][name]
        public_metrics = analysis["public_test"][name]
        print(
            f"{name:20s} local Brier={local_metrics['brier']:.4f} ECE={local_metrics['ece_10_bin']:.4f} AP={local_metrics['average_precision']:.4f} | public Brier={public_metrics['brier']:.4f} ECE={public_metrics['ece_10_bin']:.4f}"
        )
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
