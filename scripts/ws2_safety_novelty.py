"""WS2-17: compare confidence-only against confidence + novelty gating.

Fits SVM on original train, temperatures on canonical calibration, stackers
on canonical meta_train. Chooses a common decision threshold on meta_train
local examples, then evaluates on canonical diagnostic local and public test.

uv run --with scikit-learn --with scipy python scripts/ws2_safety_novelty.py
"""
from __future__ import annotations

import argparse
import csv
import json
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


def novelty_scores(query: np.ndarray, reference: np.ndarray, centroids: np.ndarray) -> dict[str, np.ndarray]:
    def unit(a: np.ndarray) -> np.ndarray:
        a = np.asarray(a, dtype=np.float32)
        norm = np.linalg.norm(a, axis=1, keepdims=True)
        if np.any(norm < 1e-8):
            raise ValueError("Zero embedding")
        return a / norm
    q, ref, centers = unit(query), unit(reference), unit(centroids)
    knn = np.empty(len(q), dtype=np.float32)
    for start in range(0, len(q), 128):
        similarity = q[start:start + 128] @ ref.T
        knn[start:start + 128] = 1 - np.partition(similarity, -5, axis=1)[:, -5:].mean(axis=1)
    return {"cosine_knn5": knn, "cosine_centroid": 1 - (q @ centers.T).max(axis=1)}


def gated_metrics(y: np.ndarray, p: np.ndarray, confidence_threshold: float,
                  novelty: np.ndarray | None = None, novelty_threshold: float | None = None) -> dict[str, object]:
    blue = np.isin(y, BLUE_IDS)
    candidate = p[:, BLUE_IDS].sum(axis=1) >= confidence_threshold
    accepted = candidate if novelty is None else candidate & (novelty <= float(novelty_threshold))
    harmful = int((accepted & ~blue).sum())
    eligible = int(blue.sum())
    return {
        "n": len(y),
        "costly_errors": harmful,
        "nonblue_n": int((~blue).sum()),
        "costly_error_rate": harmful / int((~blue).sum()) if (~blue).any() else None,
        "blue_recall": float((accepted & blue).sum() / eligible) if eligible else None,
        "blue_precision": float((accepted & blue).sum() / accepted.sum()) if accepted.any() else None,
        "deferred_n": int((~accepted).sum()),
        "deferred_fraction": float((~accepted).mean()),
        "additional_deferred": int((candidate & ~accepted).sum()),
        "additional_harmful_blocked": int((candidate & ~accepted & ~blue).sum()),
        "additional_eligible_blocked": int((candidate & ~accepted & blue).sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", type=Path, default=ROOT / "v0.2-local-data/predictions.csv")
    ap.add_argument("--embeddings", type=Path, default=ROOT / "v0.2-local-data/embeddings.npy")
    ap.add_argument("--manifest", type=Path, default=ROOT / "validation_partitions_v1.csv")
    ap.add_argument("--swin-val", type=Path, default=ROOT / "run3/swin_validation.csv")
    ap.add_argument("--swin-public", type=Path, default=ROOT / "run3/pred_E2_public_plus_local_swin_t_public_test.csv")
    ap.add_argument("--output", type=Path, default=Path("results/ws2/safety_novelty_v1.json"))
    ap.add_argument("--ood-reference", type=Path, default=Path("results/ws2/ood/ood_reference.npz"))
    ap.add_argument("--ood-thresholds", type=Path, default=Path("results/ws2/ood/local_thresholds.json"))
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
        meta_trials = [gate_metrics(yv[meta & local], pv[meta & local], float(t)) for t in THRESHOLDS]
        feasible = [r for r in meta_trials if r["blue_recall"] is not None and r["blue_recall"] >= min_blue_recall]
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
    with np.load(args.ood_reference, allow_pickle=False) as ref_archive:
        reference = np.asarray(ref_archive["train_embeddings"], dtype=np.float32)
        centers = np.asarray(ref_archive["centroids"], dtype=np.float32)
    thresholds = json.loads(args.ood_thresholds.read_text(encoding="utf-8"))
    val_novelty = novelty_scores(embedding[[row_idx[p] for p in val_paths]], reference, centers)
    pub_novelty = novelty_scores(embedding[[row_idx[p] for p in public_paths]], reference, centers)
    novelty_report = {}
    for model_name, (pv, pp) in models.items():
        t = float(report["models"][model_name]["selected_threshold"])
        novelty_report[model_name] = {}
        for method in ("cosine_knn5", "cosine_centroid"):
            novelty_t = float(thresholds["methods"][method]["local_only"]["threshold"])
            novelty_report[model_name][method] = {
                "novelty_threshold": novelty_t,
                "local_diagnostic": gated_metrics(
                    yv[diag & local], pv[diag & local], t,
                    val_novelty[method][diag & local], novelty_t,
                ),
                "public_test": gated_metrics(yp, pp, t, pub_novelty[method], novelty_t),
            }
        novelty_report[model_name]["confidence_only"] = {
            "local_diagnostic": gated_metrics(yv[diag & local], pv[diag & local], t),
            "public_test": gated_metrics(yp, pp, t),
        }
    report["novelty_comparison"] = novelty_report
    report["limitations"].append(
        "Novelty here is tested on known-class images, not on a genuine unknown-object dataset."
    )
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
    print("\\nIncremental novelty-gate comparison:")
    for name, methods in novelty_report.items():
        print(name)
        for method, scopes in methods.items():
            for scope in ("local_diagnostic", "public_test"):
                m = scopes[scope]
                print(
                    f"  {method:16s} {scope:16s} "
                    f"costly={m['costly_errors']}/{m['nonblue_n']} "
                    f"blue_recall={m['blue_recall']:.3f} "
                    f"deferred={m['deferred_fraction']:.3f} "
                    f"extra_blocked_bad={m['additional_harmful_blocked']} "
                    f"extra_blocked_good={m['additional_eligible_blocked']}"
                )
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
