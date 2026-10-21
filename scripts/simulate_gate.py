"""Sweep the confidence gate offline on the vision model's predictions (WS3).

Reads predictions.csv (team Drive: BinBuddy_Data/results/v0.2-local-data/predictions.csv; columns
split, material and p_<material> for every material) and runs the decision flow in-process for
every row of one split, once per gate setting: each threshold with and without the e-waste rule,
plus baseline 2, the static questionnaire (all three questions every time, then the rules).

The vision model never judges cleanliness and the photos have no recorded Q1-Q3 answers, so both
are simulated: each run draws a true condition (clean with --clean-share, otherwise one of the
three contaminated conditions) and the user's answers from the likelihood tables in
app/likelihoods.py. Every gate setting sees the same draws. Because the answers follow the same
tables the decision flow uses, this compares gate settings; it does not test the tables.

    uv run python scripts/simulate_gate.py \\
        --predictions ~/BinBuddy_Data/results/v0.2-local-data/predictions.csv

Report: average questions per item, share answered without questions, verdict accuracy (against
the policy's bin for the true material and condition), costly-error rate and the general-waste
fallback rate. The costly-error rate is defined as in evaluate_baselines.py: of the runs whose true
bin is not a recycling bin, the share sent to one. Keep outputs out of git.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import UUID

# The model API is not installed as a package; put it on the path like the tests do.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "services" / "inference"))

from app import decision, likelihoods
from binbuddy_contracts.api import ConditionScore, DecisionState, MaterialScore
from binbuddy_contracts.constants import DEFAULT_SITE_POLICY
from binbuddy_contracts.enums import Channel, Condition, Material, QuestionId
from binbuddy_contracts.policy import SitePolicy, load_policies

REPO_ROOT = Path(__file__).resolve().parents[1]
CONTAMINATED = [c for c in Condition if c is not Condition.CLEAN]
UNIFORM_CONDITIONS = [ConditionScore(label=c, probability=1.0 / len(Condition)) for c in Condition]
NIL_UUID = UUID(int=0)


@dataclass(frozen=True)
class Item:
    material: Material
    probabilities: dict[Material, float]
    group: str = ""  # the physical item (photos of one item share it)


@dataclass(frozen=True)
class Case:
    item: Item
    condition: Condition
    answers: dict[QuestionId, str]


@dataclass(frozen=True)
class Outcome:
    """One simulated user's run through the flow."""

    asked: tuple[QuestionId, ...]
    correct: bool
    not_recycling: bool  # the true bin is not a recycling bin
    costly: bool  # ... and the verdict sent it to one anyway
    fallback: bool
    confidence: float  # the confidence the verdict showed


@dataclass
class GateResult:
    system: str  # "adaptive" or "static"
    threshold: float
    e_waste_rule: bool
    runs: int
    avg_questions: float
    no_question_share: float
    verdict_accuracy: float
    costly_error_rate: float
    costly_error_base: int  # runs whose true bin is not a recycling bin
    fallback_rate: float


def load_items(path: Path, split: str, source: str | None = None) -> list[Item]:
    items: list[Item] = []
    known = {m.value for m in Material}
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["split"] != split or row["material"] not in known:
                continue
            if source is not None and row.get("source") != source:
                continue
            probs = {m: max(0.0, float(row[f"p_{m.value}"])) for m in Material}
            total = sum(probs.values())
            if total <= 0.0:
                continue
            probabilities = {m: p / total for m, p in probs.items()}
            items.append(Item(Material(row["material"]), probabilities, row.get("group", "")))
    return items


def _draw(rng: random.Random, row: likelihoods.Row) -> str:
    answers = list(row)
    return rng.choices(answers, weights=[row[a] for a in answers])[0]


def simulate_answers(
    rng: random.Random,
    material: Material,
    condition: Condition,
    world: Mapping[str, likelihoods.Row] | None = None,
) -> dict[QuestionId, str]:
    """A user's answers for the true material and condition.

    Drawn from `world`, the tables real users follow (named as in likelihoods.all_rows()). It
    defaults to the tables the decision flow uses; scripts/analyse_decisions.py passes others to
    test what happens when users do not behave as the tables assume.
    """
    rows = likelihoods.all_rows() if world is None else world
    if condition is Condition.CLEAN:
        q3 = rows["q3/clean"]
    elif material in likelihoods.PERISHABLE_CONTENTS:
        q3 = rows["q3/residue_perishable"]
    else:
        q3 = rows["q3/residue_dry"]
    return {
        QuestionId.Q1_RECYCLING_SYMBOL: _draw(rng, rows[f"q1/{material.value}"]),
        QuestionId.Q2_CLEANED: _draw(rng, rows[f"q2/{condition.value}"]),
        QuestionId.Q3_SITTING_TIME: _draw(rng, q3),
    }


def make_cases(
    items: Sequence[Item],
    repeats: int,
    clean_share: float,
    seed: int,
    world: Mapping[str, likelihoods.Row] | None = None,
) -> list[Case]:
    """Simulated users. The same seed gives the same true conditions whatever the world."""
    rng = random.Random(seed)
    cases: list[Case] = []
    for item in items:
        for _ in range(repeats):
            clean = rng.random() < clean_share
            condition = Condition.CLEAN if clean else rng.choice(CONTAMINATED)
            answers = simulate_answers(rng, item.material, condition, world)
            cases.append(Case(item, condition, answers))
    return cases


def condition_prior(clean: float) -> list[ConditionScore]:
    """P(clean) = `clean`, the rest shared by the contaminated conditions (uniform is 0.25)."""
    rest = (1.0 - clean) / len(CONTAMINATED)
    return [
        ConditionScore(label=c, probability=clean if c is Condition.CLEAN else rest)
        for c in Condition
    ]


def _state(
    item: Item, site: str, prior: Sequence[ConditionScore] = UNIFORM_CONDITIONS
) -> DecisionState:
    return DecisionState(
        scan_id=NIL_UUID,
        site_policy=site,
        material_scores=[
            MaterialScore(label=m, probability=item.probabilities[m]) for m in Material
        ],
        condition_scores=list(prior),
        confidence=max(item.probabilities.values()),
        signature="offline",
    )


def run_case(
    case: Case,
    policy: SitePolicy,
    gate: decision.Gate,
    static: bool = False,
    prior: Sequence[ConditionScore] = UNIFORM_CONDITIONS,
) -> Outcome:
    run = decision.decide_static if static else decision.decide_offline
    verdict = run(_state(case.item, policy.site, prior), policy, case.answers, gate)
    true_bin = policy.first_match(case.item.material, case.condition).then
    not_recycling = policy.bin(true_bin).channel is not Channel.RECYCLE
    return Outcome(
        asked=tuple(a.question_id for a in verdict.questions_asked),
        correct=verdict.bin_key == true_bin,
        not_recycling=not_recycling,
        costly=not_recycling and verdict.channel is Channel.RECYCLE,
        fallback=verdict.rule_id == "uncertain_fallback",
        confidence=verdict.confidence,
    )


def summarise(outcomes: Sequence[Outcome], gate: decision.Gate, static: bool = False) -> GateResult:
    n = max(1, len(outcomes))
    not_recycling = sum(o.not_recycling for o in outcomes)
    return GateResult(
        system="static" if static else "adaptive",
        threshold=gate.threshold,
        e_waste_rule=gate.e_waste_alert is not None,
        runs=len(outcomes),
        avg_questions=sum(len(o.asked) for o in outcomes) / n,
        no_question_share=sum(not o.asked for o in outcomes) / n,
        verdict_accuracy=sum(o.correct for o in outcomes) / n,
        costly_error_rate=sum(o.costly for o in outcomes) / max(1, not_recycling),
        costly_error_base=not_recycling,
        fallback_rate=sum(o.fallback for o in outcomes) / n,
    )


def run_gate(
    cases: Sequence[Case], policy: SitePolicy, gate: decision.Gate, static: bool = False
) -> GateResult:
    return summarise([run_case(c, policy, gate, static) for c in cases], gate, static)


def sweep(
    cases: Sequence[Case], policy: SitePolicy, thresholds: Sequence[float]
) -> list[GateResult]:
    results: list[GateResult] = []
    for threshold in thresholds:
        for alert in (decision.E_WASTE_ALERT, None):
            gate = decision.Gate(threshold=threshold, e_waste_alert=alert)
            results.append(run_gate(cases, policy, gate))
        results.append(run_gate(cases, policy, decision.Gate(threshold=threshold), static=True))
    return results


def format_table(results: Sequence[GateResult]) -> str:
    lines = [
        "system    gate  e-waste rule  avg questions  no questions  accuracy  costly errors"
        "  fallback",
    ]
    for r in results:
        lines.append(
            f"{r.system:8s}  {r.threshold:4.2f}  {'on ' if r.e_waste_rule else 'off'}"
            f"           {r.avg_questions:5.2f}          {r.no_question_share:6.1%}"
            f"       {r.verdict_accuracy:6.1%}    {r.costly_error_rate:6.1%}"
            f"         {r.fallback_rate:6.1%}"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--split", default="val", help="tune on validation rows, never on test")
    ap.add_argument("--source", help="only rows from this source, e.g. local (team photos)")
    ap.add_argument("--policy", default=DEFAULT_SITE_POLICY)
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.6, 0.7, 0.8, 0.9])
    ap.add_argument("--repeats", type=int, default=4, help="simulated users per photo")
    ap.add_argument("--clean-share", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, help="also write the results as JSON")
    args = ap.parse_args(argv)

    policy = load_policies(REPO_ROOT / "policies")[args.policy]
    items = load_items(args.predictions, args.split, args.source)
    if not items:
        print(f"no rows in split '{args.split}' of {args.predictions}")
        return 1
    cases = make_cases(items, args.repeats, args.clean_share, args.seed)
    results = sweep(cases, policy, args.thresholds)
    where = f"source {args.source}" if args.source else "all sources"
    print(
        f"{len(items)} {args.split} photos ({where}) x {args.repeats} simulated users,"
        f" {policy.site}"
    )
    print(format_table(results))
    if args.out:
        meta = {k: str(v) for k, v in vars(args).items()}
        payload = {"settings": meta, "results": [asdict(r) for r in results]}
        args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
