"""Offline analyses of the WS3 decision flow, for the final report (WS3).

Runs the real decision flow (app/decision.py) in-process on the vision model's predictions, with
simulated users as in simulate_gate.py, and asks eight questions of it:

  1. threshold trade-off, with 95% bootstrap intervals (resampling physical items, not photos);
  2. question selection: expected entropy reduction vs a fixed order, and with D-003 switched off;
  3. answer noise: users who answer at random some of the time, adaptive vs static;
  4. wrong tables: users who follow different likelihood tables from the ones the flow assumes,
     and a different share of clean items;
  5. calibration: vision scores made over- or under-confident (temperature), and whether the
     confidence a verdict shows matches how often it is right;
  6. per material: where the questions and the errors go;
  7. held-out check: the chosen settings on the test split (tuned on val, checked once on test);
  8. the multi-stream campus policy;
  9. the condition prior: the flow starts every item at P(clean) = 0.25 (uniform over the four
     conditions); what if it started at the share of clean items users actually bring?

The photos have no recorded Q1-Q3 answers, so conditions and answers are simulated. Analyses 3 and
4 are the ones that let the simulated users disagree with the tables the flow uses.

    uv run python scripts/analyse_decisions.py \\
        --predictions ~/BinBuddy_Data/results/v0.2-local-data/predictions.csv \\
        --out docs/workstreams/ws3_analysis

Writes results.json (every number), tables.md (Markdown tables) and the figures as SVG.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, replace
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "services" / "inference"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from simulate_gate import (
    Case,
    Item,
    Outcome,
    condition_prior,
    load_items,
    make_cases,
    run_case,
    summarise,
)
from svg_charts import Panel, Series, chart

from app import decision, likelihoods
from binbuddy_contracts.constants import DEFAULT_SITE_POLICY
from binbuddy_contracts.enums import Material, QuestionId
from binbuddy_contracts.policy import SitePolicy, load_policies

REPO_ROOT = Path(__file__).resolve().parents[1]
Tables = dict[str, likelihoods.Row]
Json = dict[str, Any]

METRICS = (
    "avg_questions",
    "no_question_share",
    "verdict_accuracy",
    "costly_error_rate",
    "fallback_rate",
)
COSTLY_TARGET = 0.05  # proposal Section 9
RECOMMENDED = 0.75  # the gate the report recommends (figures and the per-material table)
SHORT = {
    QuestionId.Q1_RECYCLING_SYMBOL: "Q1",
    QuestionId.Q2_CLEANED: "Q2",
    QuestionId.Q3_SITTING_TIME: "Q3",
}


# ------------------------------------------------------------------------------ helpers


def outcomes(
    cases: Sequence[Case], policy: SitePolicy, gate: decision.Gate, static: bool = False
) -> list[Outcome]:
    return [run_case(c, policy, gate, static) for c in cases]


def metrics(outs: Sequence[Outcome], gate: decision.Gate, static: bool = False) -> Json:
    r = asdict(summarise(outs, gate, static))
    return {k: r[k] for k in ("runs", "costly_error_base", *METRICS)}


def _group_key(case: Case) -> str:
    return case.item.group or str(id(case.item))


def item_count(cases: Sequence[Case]) -> int:
    return len({_group_key(c) for c in cases})


def _group_sums(
    cases: Sequence[Case], outs: Sequence[Outcome]
) -> tuple[NDArray[np.float64], list[str]]:
    """Per physical item: runs, questions, no-question runs, correct, not-recycling, costly,
    fallback."""
    index: dict[str, int] = {}
    rows: list[list[float]] = []
    for case, o in zip(cases, outs, strict=True):
        key = _group_key(case)
        if key not in index:
            index[key] = len(rows)
            rows.append([0.0] * 7)
        row = rows[index[key]]
        for i, v in enumerate(
            (1, len(o.asked), not o.asked, o.correct, o.not_recycling, o.costly, o.fallback)
        ):
            row[i] += float(v)
    return np.array(rows), list(index)


def _ratios(sums: NDArray[np.float64]) -> dict[str, NDArray[np.float64]]:
    n = np.maximum(sums[..., 0], 1)
    return {
        "avg_questions": sums[..., 1] / n,
        "no_question_share": sums[..., 2] / n,
        "verdict_accuracy": sums[..., 3] / n,
        "costly_error_rate": sums[..., 5] / np.maximum(sums[..., 4], 1),
        "fallback_rate": sums[..., 6] / n,
    }


def bootstrap_weights(groups: int, resamples: int, seed: int) -> NDArray[np.float64]:
    """How often each item is drawn in each bootstrap resample (resampling items, not photos)."""
    rng = np.random.default_rng(seed)
    return rng.multinomial(groups, np.full(groups, 1.0 / groups), size=resamples).astype(float)


def intervals(
    cases: Sequence[Case], outs: Sequence[Outcome], weights: NDArray[np.float64]
) -> dict[str, list[float]]:
    sums, _ = _group_sums(cases, outs)
    return {
        k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
        for k, v in _ratios(weights @ sums).items()
    }


def paired_difference(
    cases: Sequence[Case],
    a: Sequence[Outcome],
    b: Sequence[Outcome],
    weights: NDArray[np.float64],
) -> dict[str, list[float]]:
    """a minus b on the same resampled items: the estimate and its 95% interval."""
    sums_a, sums_b = _group_sums(cases, a)[0], _group_sums(cases, b)[0]
    point_a, point_b = _ratios(sums_a.sum(axis=0)), _ratios(sums_b.sum(axis=0))
    ra, rb = _ratios(weights @ sums_a), _ratios(weights @ sums_b)
    out = {}
    for k in METRICS:
        d = ra[k] - rb[k]
        point = float(point_a[k] - point_b[k])
        out[k] = [point, float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
    return out


def noisy(tables: Mapping[str, likelihoods.Row], rate: float) -> Tables:
    """Users who, with probability `rate`, pick an answer at random instead of the true one."""
    return {
        name: {a: (1.0 - rate) * p + rate / len(row) for a, p in row.items()}
        for name, row in tables.items()
    }


def perturbed(tables: Mapping[str, likelihoods.Row], sigma: float, rng: random.Random) -> Tables:
    """Tables that differ from ours: each cell times exp(N(0, sigma)), then renormalised."""
    world: Tables = {}
    for name, row in tables.items():
        raw = {
            a: max(p * math.exp(rng.gauss(0.0, sigma)), likelihoods.LIKELIHOOD_FLOOR)
            for a, p in row.items()
        }
        total = sum(raw.values())
        world[name] = {a: p / total for a, p in raw.items()}
    return world


def tempered(items: Sequence[Item], temperature: float) -> list[Item]:
    """Material scores sharpened (T < 1) or flattened (T > 1): p ** (1 / T), renormalised."""
    out = []
    for item in items:
        raw = {m: p ** (1.0 / temperature) for m, p in item.probabilities.items()}
        total = sum(raw.values())
        out.append(replace(item, probabilities={m: p / total for m, p in raw.items()}))
    return out


def expected_calibration_error(confidence: Sequence[float], correct: Sequence[bool]) -> float:
    """ECE over ten equal-width confidence bins."""
    bins: dict[int, list[tuple[float, bool]]] = {}
    for c, ok in zip(confidence, correct, strict=True):
        bins.setdefault(min(int(c * 10), 9), []).append((c, ok))
    n = len(confidence)
    return sum(
        len(b) / n * abs(sum(c for c, _ in b) / len(b) - sum(ok for _, ok in b) / len(b))
        for b in bins.values()
    )


def reliability(outs: Sequence[Outcome], edges: Sequence[float]) -> list[Json]:
    rows = []
    for lo, hi in pairwise(edges):
        sel = [o for o in outs if lo <= o.confidence < hi or (hi == 1.0 and o.confidence == 1.0)]
        if sel:
            rows.append(
                {
                    "bin": f"{lo:.0%}-{hi:.0%}",
                    "runs": len(sel),
                    "mean_confidence": sum(o.confidence for o in sel) / len(sel),
                    "accuracy": sum(o.correct for o in sel) / len(sel),
                }
            )
    return rows


def best_threshold(rows: Sequence[Json]) -> float | None:
    """Highest accuracy among the thresholds whose costly-error rate meets the target."""
    ok = [r for r in rows if r["costly_error_rate"] <= COSTLY_TARGET]
    return max(ok, key=lambda r: r["verdict_accuracy"])["threshold"] if ok else None


# ---------------------------------------------------------------------------- analyses


def threshold_tradeoff(
    cases: Sequence[Case], policy: SitePolicy, thresholds: Sequence[float], resamples: int
) -> Json:
    weights = bootstrap_weights(item_count(cases), resamples, 0)
    rows: list[Json] = []
    adaptive: dict[float, list[Outcome]] = {}
    for t in thresholds:
        gate = decision.Gate(threshold=t)
        for static in (False, True):
            outs = outcomes(cases, policy, gate, static)
            if not static:
                adaptive[t] = outs
            rows.append(
                {
                    "system": "static" if static else "adaptive",
                    "threshold": t,
                    **metrics(outs, gate, static),
                    "ci95": intervals(cases, outs, weights),
                }
            )
    comparisons = {}
    for x, y in ((0.7, 0.8), (0.75, 0.8), (0.7, 0.75)):
        if x in adaptive and y in adaptive:
            comparisons[f"adaptive {x:.2f} minus {y:.2f}"] = paired_difference(
                cases, adaptive[x], adaptive[y], weights
            )
    static_08 = outcomes(cases, policy, decision.Gate(threshold=0.8), static=True)
    for x in (0.7, 0.75, 0.8):
        if x in adaptive:
            comparisons[f"adaptive {x:.2f} minus static 0.80"] = paired_difference(
                cases, adaptive[x], static_08, weights
            )
    return {"rows": rows, "paired_differences": comparisons, "resamples": resamples}


VARIANTS: tuple[tuple[str, decision.Gate], ...] = (
    ("as designed: entropy choice, D-003 on", decision.Gate()),
    ("fixed order Q2 then Q3, D-003 on", decision.Gate(follow_up="q2_first")),
    ("fixed order Q3 then Q2, D-003 on", decision.Gate(follow_up="q3_first")),
    ("entropy choice, D-003 off", decision.Gate(require_q2=False)),
    ("fixed order Q2 then Q3, D-003 off", decision.Gate(require_q2=False, follow_up="q2_first")),
    ("fixed order Q3 then Q2, D-003 off", decision.Gate(require_q2=False, follow_up="q3_first")),
)


def selection_ablation(
    cases: Sequence[Case], policy: SitePolicy, thresholds: Sequence[float]
) -> Json:
    rows: list[Json] = []
    paths: dict[str, dict[str, float]] = {}
    for t in thresholds:
        for name, base in VARIANTS:
            gate = replace(base, threshold=t)
            outs = outcomes(cases, policy, gate)
            rows.append({"variant": name, "threshold": t, **metrics(outs, gate)})
            if name.startswith("as designed") or name == "entropy choice, D-003 off":
                counts = Counter(" > ".join(SHORT[q] for q in o.asked) or "none" for o in outs)
                paths[f"{name} @ {t:.2f}"] = {k: v / len(outs) for k, v in counts.most_common()}
    return {"rows": rows, "question_paths": paths}


def answer_noise(
    items: Sequence[Item],
    policy: SitePolicy,
    rates: Sequence[float],
    thresholds: Sequence[float],
    repeats: int,
    clean_share: float,
    seed: int,
) -> Json:
    rows: list[Json] = []
    for rate in rates:
        cases = make_cases(items, repeats, clean_share, seed, noisy(likelihoods.all_rows(), rate))
        for t in thresholds:
            gate = decision.Gate(threshold=t)
            for static in (False, True):
                outs = outcomes(cases, policy, gate, static)
                rows.append(
                    {
                        "noise": rate,
                        "system": "static" if static else "adaptive",
                        "threshold": t,
                        **metrics(outs, gate, static),
                    }
                )
    return {"rows": rows}


def wrong_tables(
    items: Sequence[Item],
    policy: SitePolicy,
    sigmas: Sequence[float],
    worlds: int,
    thresholds: Sequence[float],
    clean_shares: Sequence[float],
    repeats: int,
    clean_share: float,
    seed: int,
) -> Json:
    rows: list[Json] = []
    winners: dict[str, Counter[str]] = {}
    for sigma in sigmas:
        rng = random.Random(seed + 1)
        per_threshold: dict[float, list[Json]] = {t: [] for t in thresholds}
        winners[f"{sigma}"] = Counter()
        for w in range(worlds):
            world = perturbed(likelihoods.all_rows(), sigma, rng)
            cases = make_cases(items, repeats, clean_share, seed + w, world)
            world_rows = []
            for t in thresholds:
                gate = decision.Gate(threshold=t)
                m = {"threshold": t, **metrics(outcomes(cases, policy, gate), gate)}
                per_threshold[t].append(m)
                world_rows.append(m)
            best = best_threshold(world_rows)
            winners[f"{sigma}"][f"{best:.2f}" if best is not None else "none"] += 1
        for t, ms in per_threshold.items():
            acc = [m["verdict_accuracy"] for m in ms]
            costly = [m["costly_error_rate"] for m in ms]
            rows.append(
                {
                    "sigma": sigma,
                    "threshold": t,
                    "worlds": worlds,
                    "accuracy_mean": sum(acc) / len(acc),
                    "accuracy_min": min(acc),
                    "accuracy_max": max(acc),
                    "costly_mean": sum(costly) / len(costly),
                    "costly_max": max(costly),
                    "worlds_within_target": sum(c <= COSTLY_TARGET for c in costly) / len(costly),
                    "avg_questions_mean": sum(m["avg_questions"] for m in ms) / len(ms),
                }
            )
    shares = []
    for share in clean_shares:
        cases = make_cases(items, repeats, share, seed)
        for t in thresholds:
            gate = decision.Gate(threshold=t)
            outs = outcomes(cases, policy, gate)
            shares.append({"clean_share": share, "threshold": t, **metrics(outs, gate)})
    return {
        "rows": rows,
        "best_threshold_counts": {k: dict(v) for k, v in winners.items()},
        "clean_share_rows": shares,
    }


def calibration(
    items: Sequence[Item],
    policy: SitePolicy,
    temperatures: Sequence[float],
    thresholds: Sequence[float],
    repeats: int,
    clean_share: float,
    seed: int,
) -> Json:
    top_conf = [max(i.probabilities.values()) for i in items]
    top_ok = [max(i.probabilities, key=i.probabilities.__getitem__) is i.material for i in items]
    vision = {
        "photos": len(items),
        "top1_accuracy": sum(top_ok) / len(items),
        "mean_top1_confidence": sum(top_conf) / len(items),
        "ece": expected_calibration_error(top_conf, top_ok),
    }
    rows: list[Json] = []
    for temp in temperatures:
        cases = make_cases(tempered(items, temp), repeats, clean_share, seed)
        for t in thresholds:
            gate = decision.Gate(threshold=t)
            outs = outcomes(cases, policy, gate)
            rows.append(
                {
                    "temperature": temp,
                    "threshold": t,
                    **metrics(outs, gate),
                    "verdict_ece": expected_calibration_error(
                        [o.confidence for o in outs], [o.correct for o in outs]
                    ),
                }
            )
    edges = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
    verdict = {}
    cases = make_cases(items, repeats, clean_share, seed)
    for t in thresholds:
        outs = outcomes(cases, policy, decision.Gate(threshold=t))
        verdict[f"{t:.2f}"] = {
            "ece": expected_calibration_error(
                [o.confidence for o in outs], [o.correct for o in outs]
            ),
            "bins": reliability(outs, edges),
        }
    return {"vision_material_scores": vision, "temperature_rows": rows, "verdict": verdict}


def per_material(cases: Sequence[Case], policy: SitePolicy, threshold: float) -> Json:
    gate = decision.Gate(threshold=threshold)
    outs = outcomes(cases, policy, gate)
    rows = []
    for material in Material:
        pairs = [(c, o) for c, o in zip(cases, outs, strict=True) if c.item.material is material]
        if not pairs:
            continue
        sel = [o for _, o in pairs]
        rows.append(
            {
                "material": material.value,
                "photos": len({id(c.item) for c, _ in pairs}),
                **metrics(sel, gate),
                "costly_runs": sum(o.costly for o in sel),
                "wrong_runs": sum(not o.correct for o in sel),
            }
        )
    return {"threshold": threshold, "rows": rows}


def held_out(
    items: Sequence[Item],
    policy: SitePolicy,
    thresholds: Sequence[float],
    repeats: int,
    clean_share: float,
    seed: int,
    resamples: int,
) -> Json:
    cases = make_cases(items, repeats, clean_share, seed)
    weights = bootstrap_weights(item_count(cases), resamples, 1)
    rows = []
    for t in thresholds:
        gate = decision.Gate(threshold=t)
        for static in (False, True):
            outs = outcomes(cases, policy, gate, static)
            rows.append(
                {
                    "system": "static" if static else "adaptive",
                    "threshold": t,
                    **metrics(outs, gate, static),
                    "ci95": intervals(cases, outs, weights),
                }
            )
    return {"photos": len(items), "rows": rows}


def prior_check(
    items: Sequence[Item],
    policy: SitePolicy,
    priors: Sequence[float],
    clean_shares: Sequence[float],
    thresholds: Sequence[float],
    repeats: int,
    seed: int,
) -> Json:
    rows = []
    for share in clean_shares:
        cases = make_cases(items, repeats, share, seed)
        for prior in priors:
            scores = condition_prior(prior)
            for t in thresholds:
                gate = decision.Gate(threshold=t)
                outs = [run_case(c, policy, gate, prior=scores) for c in cases]
                rows.append(
                    {
                        "clean_share": share,
                        "prior_clean": prior,
                        "threshold": t,
                        **metrics(outs, gate),
                    }
                )
    return {"rows": rows}


def other_policy(cases: Sequence[Case], policy: SitePolicy, thresholds: Sequence[float]) -> Json:
    rows = []
    for t in thresholds:
        gate = decision.Gate(threshold=t)
        for static in (False, True):
            outs = outcomes(cases, policy, gate, static)
            rows.append(
                {
                    "system": "static" if static else "adaptive",
                    "threshold": t,
                    **metrics(outs, gate, static),
                }
            )
    return {"site": policy.site, "rows": rows}


# ------------------------------------------------------------------------------ output


def _pct(v: float) -> str:
    return f"{v:.1%}"


def _ci(r: Json, k: str) -> str:
    lo, hi = r["ci95"][k]
    fmt: Callable[[float], str] = (lambda v: f"{v:.2f}") if k == "avg_questions" else _pct
    return f"{fmt(r[k])} ({fmt(lo)}-{fmt(hi)})"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def _std_cells(r: Json) -> list[str]:
    return [
        f"{r['avg_questions']:.2f}",
        _pct(r["no_question_share"]),
        _pct(r["verdict_accuracy"]),
        _pct(r["costly_error_rate"]),
        _pct(r["fallback_rate"]),
    ]


STD_HEAD = ["Avg questions", "No questions", "Accuracy", "Costly errors", "Fallback"]


def markdown(results: Json) -> str:
    out = ["# WS3 decision-flow analysis: tables", "", f"Settings: `{results['settings']}`", ""]

    a = results["1_threshold_tradeoff"]
    out += ["## 1. Threshold trade-off (val, 95% bootstrap intervals over items)", ""]
    out.append(
        _table(
            ["System", "Gate", "Avg questions", "Accuracy", "Costly errors", "Fallback"],
            [
                [
                    r["system"],
                    f"{r['threshold']:.2f}",
                    _ci(r, "avg_questions"),
                    _ci(r, "verdict_accuracy"),
                    _ci(r, "costly_error_rate"),
                    _ci(r, "fallback_rate"),
                ]
                for r in a["rows"]
            ],
        )
    )
    out += ["", "Paired differences (same resampled items; estimate and 95% interval):", ""]
    out.append(
        _table(
            ["Comparison", "Accuracy", "Costly errors", "Avg questions"],
            [
                [
                    name,
                    *(
                        f"{d[k][0]:+.1%} ({d[k][1]:+.1%} to {d[k][2]:+.1%})"
                        for k in ("verdict_accuracy", "costly_error_rate")
                    ),
                    f"{d['avg_questions'][0]:+.2f} ({d['avg_questions'][1]:+.2f} to "
                    f"{d['avg_questions'][2]:+.2f})",
                ]
                for name, d in a["paired_differences"].items()
            ],
        )
    )

    s = results["2_question_selection"]
    out += ["", "## 2. Question selection and D-003", ""]
    out.append(
        _table(
            ["Variant", "Gate", *STD_HEAD],
            [[r["variant"], f"{r['threshold']:.2f}", *_std_cells(r)] for r in s["rows"]],
        )
    )
    out += ["", "Question paths (share of runs):", ""]
    for name, paths in s["question_paths"].items():
        out.append(f"- {name}: " + ", ".join(f"{k} {_pct(v)}" for k, v in paths.items()))

    n = results["3_answer_noise"]
    out += ["", "## 3. Answer noise (share of answers given at random)", ""]
    out.append(
        _table(
            ["Noise", "System", "Gate", *STD_HEAD],
            [
                [_pct(r["noise"]), r["system"], f"{r['threshold']:.2f}", *_std_cells(r)]
                for r in n["rows"]
            ],
        )
    )

    w = results["4_wrong_tables"]
    out += ["", "## 4. Users who follow different tables (random worlds)", ""]
    out.append(
        _table(
            [
                "Sigma",
                "Gate",
                "Accuracy mean (min-max)",
                "Costly mean (max)",
                "Worlds within 5%",
                "Avg questions",
            ],
            [
                [
                    f"{r['sigma']:g}",
                    f"{r['threshold']:.2f}",
                    f"{_pct(r['accuracy_mean'])} ({_pct(r['accuracy_min'])}-"
                    f"{_pct(r['accuracy_max'])})",
                    f"{_pct(r['costly_mean'])} ({_pct(r['costly_max'])})",
                    _pct(r["worlds_within_target"]),
                    f"{r['avg_questions_mean']:.2f}",
                ]
                for r in w["rows"]
            ],
        )
    )
    out += ["", "Best threshold per world (highest accuracy with costly errors ≤ 5%):", ""]
    for sigma, counts in w["best_threshold_counts"].items():
        out.append(f"- sigma {sigma}: " + ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())))
    out += ["", "Share of clean items:", ""]
    out.append(
        _table(
            ["Clean share", "Gate", *STD_HEAD],
            [
                [_pct(r["clean_share"]), f"{r['threshold']:.2f}", *_std_cells(r)]
                for r in w["clean_share_rows"]
            ],
        )
    )

    c = results["5_calibration"]
    v = c["vision_material_scores"]
    out += [
        "",
        "## 5. Calibration",
        "",
        f"Vision material scores on these photos: top-1 accuracy {_pct(v['top1_accuracy'])}, "
        f"mean top-1 confidence {_pct(v['mean_top1_confidence'])}, ECE {v['ece']:.3f}.",
        "",
    ]
    out.append(
        _table(
            ["Temperature", "Gate", *STD_HEAD, "Verdict ECE"],
            [
                [
                    f"{r['temperature']:g}",
                    f"{r['threshold']:.2f}",
                    *_std_cells(r),
                    f"{r['verdict_ece']:.3f}",
                ]
                for r in c["temperature_rows"]
            ],
        )
    )
    for t, rel in c["verdict"].items():
        out += ["", f"Verdict confidence vs accuracy at gate {t} (ECE {rel['ece']:.3f}):", ""]
        out.append(
            _table(
                ["Confidence shown", "Runs", "Mean confidence", "Actually right"],
                [
                    [b["bin"], str(b["runs"]), _pct(b["mean_confidence"]), _pct(b["accuracy"])]
                    for b in rel["bins"]
                ],
            )
        )

    m = results["6_per_material"]
    out += ["", f"## 6. Per material (adaptive, gate {m['threshold']:.2f})", ""]
    out.append(
        _table(
            ["Material", "Photos", *STD_HEAD, "Costly runs"],
            [
                [r["material"], str(r["photos"]), *_std_cells(r), str(r["costly_runs"])]
                for r in m["rows"]
            ],
        )
    )

    h = results["7_held_out_test"]
    out += ["", f"## 7. Held-out check on the test split ({h['photos']} photos)", ""]
    out.append(
        _table(
            ["System", "Gate", "Avg questions", "Accuracy", "Costly errors", "Fallback"],
            [
                [
                    r["system"],
                    f"{r['threshold']:.2f}",
                    _ci(r, "avg_questions"),
                    _ci(r, "verdict_accuracy"),
                    _ci(r, "costly_error_rate"),
                    _ci(r, "fallback_rate"),
                ]
                for r in h["rows"]
            ],
        )
    )

    p = results["8_campus_policy"]
    out += ["", f"## 8. Multi-stream policy ({p['site']})", ""]
    out.append(
        _table(
            ["System", "Gate", *STD_HEAD],
            [[r["system"], f"{r['threshold']:.2f}", *_std_cells(r)] for r in p["rows"]],
        )
    )

    q = results["9_condition_prior"]
    out += ["", "## 9. Condition prior (P(clean) before any answer)", ""]
    out.append(
        _table(
            ["Clean share of items", "Prior P(clean)", "Gate", *STD_HEAD],
            [
                [
                    _pct(r["clean_share"]),
                    f"{r['prior_clean']:.2f}",
                    f"{r['threshold']:.2f}",
                    *_std_cells(r),
                ]
                for r in q["rows"]
            ],
        )
    )
    return "\n".join(out) + "\n"


def figures(results: Json) -> dict[str, str]:
    rows = results["1_threshold_tradeoff"]["rows"]
    ts = sorted({r["threshold"] for r in rows})

    def series(system: str, key: str) -> Series:
        sel = sorted((r for r in rows if r["system"] == system), key=lambda r: r["threshold"])
        return Series(
            "Adaptive" if system == "adaptive" else "Static (all 3)",
            [r[key] for r in sel],
            [r["ci95"][key][0] for r in sel],
            [r["ci95"][key][1] for r in sel],
        )

    def panel(title: str, key: str, y_min: float, y_max: float, **kw: Any) -> Panel:
        return Panel(
            title,
            ts,
            [series("adaptive", key), series("static", key)],
            "Confidence gate",
            y_min,
            y_max,
            **kw,
        )

    tradeoff = chart(
        "Confidence gate trade-off (val photos, 95% intervals)",
        [
            panel("Verdict accuracy", "verdict_accuracy", 0.5, 1.0, y_step=0.1),
            panel(
                "Costly errors",
                "costly_error_rate",
                0.0,
                0.08,
                y_step=0.02,
                target=(COSTLY_TARGET, "target 5%"),
            ),
            panel("Questions per item", "avg_questions", 0.0, 3.0, percent=False, y_step=0.5),
            panel("Fell back to general waste", "fallback_rate", 0.0, 0.6, y_step=0.15),
        ],
    )

    noise_rows = [r for r in results["3_answer_noise"]["rows"] if r["threshold"] == RECOMMENDED]
    rates = sorted({r["noise"] for r in noise_rows})

    def noise_series(system: str, key: str) -> Series:
        sel = sorted((r for r in noise_rows if r["system"] == system), key=lambda r: r["noise"])
        return Series(
            "Adaptive" if system == "adaptive" else "Static (all 3)", [r[key] for r in sel]
        )

    pct_x: Callable[[float], str] = lambda x: f"{x:.0%}"  # noqa: E731
    noise = chart(
        f"Users who answer at random some of the time (gate {RECOMMENDED:.2f})",
        [
            Panel(
                "Verdict accuracy",
                rates,
                [
                    noise_series("adaptive", "verdict_accuracy"),
                    noise_series("static", "verdict_accuracy"),
                ],
                "Share of answers given at random",
                0.6,
                1.0,
                y_step=0.1,
                x_format=pct_x,
            ),
            Panel(
                "Costly errors",
                rates,
                [
                    noise_series("adaptive", "costly_error_rate"),
                    noise_series("static", "costly_error_rate"),
                ],
                "Share of answers given at random",
                0.0,
                0.10,
                y_step=0.02,
                target=(COSTLY_TARGET, "target 5%"),
                x_format=pct_x,
            ),
        ],
    )

    temp_rows = [
        r for r in results["5_calibration"]["temperature_rows"] if r["threshold"] == RECOMMENDED
    ]
    temps = [r["temperature"] for r in temp_rows]
    temp_x = "Temperature (<1 sharper, >1 flatter)"

    def temp_panel(title: str, key: str, y_min: float, y_max: float, **kw: Any) -> Panel:
        series = [Series("Adaptive", [r[key] for r in temp_rows])]
        return Panel(title, temps, series, temp_x, y_min, y_max, **kw)

    calib = chart(
        f"Over- or under-confident vision scores (adaptive, gate {RECOMMENDED:.2f})",
        [
            temp_panel("Verdict accuracy", "verdict_accuracy", 0.6, 1.0, y_step=0.1),
            temp_panel(
                "Costly errors",
                "costly_error_rate",
                0.0,
                0.08,
                y_step=0.02,
                target=(COSTLY_TARGET, "target 5%"),
            ),
            temp_panel("Questions per item", "avg_questions", 0.0, 3.0, percent=False, y_step=0.5),
            temp_panel("Fell back to general waste", "fallback_rate", 0.0, 0.3, y_step=0.1),
        ],
    )
    return {
        "fig_threshold_tradeoff.svg": tradeoff,
        "fig_answer_noise.svg": noise,
        "fig_calibration.svg": calib,
    }


# -------------------------------------------------------------------------------- main


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--source", default="local", help="rows from this source (team photos)")
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.6, 0.7, 0.75, 0.8, 0.9, 0.95])
    ap.add_argument("--repeats", type=int, default=20, help="simulated users per photo")
    ap.add_argument("--clean-share", type=float, default=0.5)
    ap.add_argument("--worlds", type=int, default=20, help="random worlds per sigma (analysis 4)")
    ap.add_argument("--resamples", type=int, default=1000, help="bootstrap resamples")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True, help="output folder")
    args = ap.parse_args(argv)

    policies = load_policies(REPO_ROOT / "policies")
    hdb = policies[DEFAULT_SITE_POLICY]
    val = load_items(args.predictions, "val", args.source)
    test = load_items(args.predictions, "test", args.source)
    if not val or not test:
        print(f"need val and test rows from source '{args.source}' in {args.predictions}")
        return 1
    rep, share, seed = args.repeats, args.clean_share, args.seed
    cases = make_cases(val, rep, share, seed)
    focus = [t for t in (0.7, 0.75, 0.8) if t in args.thresholds] or args.thresholds[:2]

    print(f"{len(val)} val / {len(test)} test photos, {rep} simulated users each")
    results: Json = {"settings": {k: str(v) for k, v in vars(args).items()}}
    steps: list[tuple[str, Callable[[], Json]]] = [
        (
            "1_threshold_tradeoff",
            lambda: threshold_tradeoff(cases, hdb, args.thresholds, args.resamples),
        ),
        ("2_question_selection", lambda: selection_ablation(cases, hdb, focus)),
        (
            "3_answer_noise",
            lambda: answer_noise(val, hdb, [0.0, 0.1, 0.2, 0.3], focus, rep, share, seed),
        ),
        (
            "4_wrong_tables",
            lambda: wrong_tables(
                val,
                hdb,
                [0.25, 0.5],
                args.worlds,
                args.thresholds,
                [0.3, 0.5, 0.7],
                rep,
                share,
                seed,
            ),
        ),
        (
            "5_calibration",
            lambda: calibration(val, hdb, [0.5, 0.75, 1.0, 1.5, 2.0], focus, rep, share, seed),
        ),
        ("6_per_material", lambda: per_material(cases, hdb, RECOMMENDED)),
        ("7_held_out_test", lambda: held_out(test, hdb, focus, rep, share, seed, args.resamples)),
        ("8_campus_policy", lambda: other_policy(cases, policies["campus_example"], focus)),
        (
            "9_condition_prior",
            lambda: prior_check(val, hdb, [0.25, 0.5], [0.3, 0.5, 0.7], focus, rep, seed),
        ),
    ]
    for name, run in steps:
        print(f"  {name} ...", flush=True)
        results[name] = run()

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    (args.out / "tables.md").write_text(markdown(results), encoding="utf-8")
    for name, svg in figures(results).items():
        (args.out / name).write_text(svg, encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
