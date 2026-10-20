"""Decision flow: confidence gate, adaptive questions and verdict (WS3 owns the logic).

Belief. DecisionState keeps two distributions: material (calibrated, from the vision model) and
condition (uniform from the vision model, which never judges cleanliness). Each answer updates
one of them by naive-Bayes evidence fusion with the likelihoods in app/likelihoods.py:
Q1 updates the material, Q2 and Q3 update the condition.

Confidence in a verdict is measured on the bin, not the material. Every (material, condition)
pair maps to a bin through the site policy, so P(bin) is the sum of P(material) * P(condition)
over the pairs that lead to it. The verdict names the most likely bin, with the rule of the most
likely pair inside that bin, and its confidence is that bin's probability (so an item that may or
may not be clean leans to general waste). `state.confidence` carries the same number, so the app's
confidence meter moves with each answer. A confident styrofoam is settled without questions (every
condition leads to general waste); a confident plastic is not (its condition decides the bin).

Gate (proposal 1.2 and the WS1 handoff):
  - settled when P(verdict bin) >= threshold, unless one of the safety rules below applies;
  - D-003: a recyclable top material is never settled before Q2 has been asked;
  - e-waste rule: not settled on another bin while P(e_waste) > E_WASTE_ALERT (redundant at a
    0.80 gate, which no other bin can reach then; it matters if the gate is tuned lower);
  - Q1 always opens; then the remaining question with the largest expected entropy reduction
    of the bin distribution; at most three questions;
  - still not settled when the bank is exhausted: general waste plus the NEA link.

Every verdict carries a plain-English rule trace (shown in the app as "How Bin Buddy decided").
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Final, Literal, TypeVar
from uuid import UUID

from app import likelihoods
from binbuddy_contracts.api import (
    ConditionScore,
    DecisionState,
    MaterialScore,
    Question,
    QuestionAnswer,
    VerdictResult,
    VisionAnalysis,
)
from binbuddy_contracts.constants import CONFIDENCE_THRESHOLD
from binbuddy_contracts.enums import Channel, Condition, Material, QuestionId
from binbuddy_contracts.policy import SitePolicy
from binbuddy_contracts.questions import OPENING_QUESTION, QUESTION_BANK

RECYCLABLE_MATERIALS: Final = frozenset(
    {Material.PAPER, Material.PLASTIC, Material.GLASS, Material.METAL, Material.BEVERAGE_CARTON}
)
# WS1 handoff: treat an item as possible e-waste when P(e_waste) > 0.2 (catches 15 of 17 e-waste
# items on the team test photos, with 6 false flags).
E_WASTE_ALERT: Final = 0.20
FOLLOW_UP_QUESTIONS: Final = (QuestionId.Q2_CLEANED, QuestionId.Q3_SITTING_TIME)

# Rule-trace wording (user-facing: short, plain, never blaming the user).
MATERIAL_WORDS: Final[dict[Material, str]] = {
    Material.PAPER: "paper or cardboard",
    Material.PLASTIC: "plastic",
    Material.GLASS: "glass",
    Material.METAL: "metal",
    Material.BEVERAGE_CARTON: "a drink carton",
    Material.STYROFOAM: "styrofoam",
    Material.E_WASTE: "an electronic item",
    Material.TEXTILE: "clothing or textile",
    Material.FOOD_WASTE: "food waste",
    Material.OTHER_GENERAL: "a general-waste item",
}
ANSWER_WORDS: Final[dict[tuple[QuestionId, str], str]] = {
    (QuestionId.Q1_RECYCLING_SYMBOL, "yes"): "You see a recycling symbol",
    (QuestionId.Q1_RECYCLING_SYMBOL, "no"): "No recycling symbol",
    (QuestionId.Q1_RECYCLING_SYMBOL, "not_sure"): "Not sure about a recycling symbol",
    (QuestionId.Q2_CLEANED, "yes"): "Rinsed",
    (QuestionId.Q2_CLEANED, "no"): "Not rinsed",
    (QuestionId.Q2_CLEANED, "not_sure"): "Not sure if it was rinsed",
    (QuestionId.Q3_SITTING_TIME, "under_1_hour"): "Used less than an hour ago",
    (QuestionId.Q3_SITTING_TIME, "same_day"): "Used earlier today",
    (QuestionId.Q3_SITTING_TIME, "more_than_1_day"): "Left for more than a day",
}

K = TypeVar("K")
Materials = dict[Material, float]
Conditions = dict[Condition, float]


@dataclass(frozen=True)
class Gate:
    """Gate settings. The API always uses DEFAULT_GATE; other values are for offline tuning."""

    threshold: float = CONFIDENCE_THRESHOLD
    e_waste_alert: float | None = E_WASTE_ALERT  # None switches the e-waste rule off
    # Ablations only (scripts/analyse_decisions.py): D-003 on or off, and how the follow-up after
    # Q1 is chosen ("entropy" as designed, or a fixed order).
    require_q2: bool = True
    follow_up: Literal["entropy", "q2_first", "q3_first"] = "entropy"


DEFAULT_GATE: Final = Gate()


class InvalidStateError(ValueError):
    pass


# ------------------------------------------------------------------------------ signing


def _payload(state: DecisionState) -> bytes:
    data = state.model_dump(mode="json", exclude={"signature"})
    return json.dumps(data, sort_keys=True, separators=(",", ":")).encode()


def sign(state: DecisionState, secret: bytes) -> DecisionState:
    sig = hmac.new(secret, _payload(state), hashlib.sha256).hexdigest()
    return state.model_copy(update={"signature": sig})


def verify_signature(state: DecisionState, secret: bytes) -> None:
    expected = hmac.new(secret, _payload(state), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, state.signature):
        raise InvalidStateError("state signature is invalid (state was modified)")


def new_state(
    scan_id: UUID, analysis: VisionAnalysis, policy: SitePolicy, secret: bytes
) -> DecisionState:
    materials = {s.label: s.probability for s in analysis.material_scores}
    conditions = {s.label: s.probability for s in analysis.condition_scores}
    unsigned = DecisionState(
        scan_id=scan_id,
        site_policy=policy.site,
        material_scores=analysis.material_scores,
        condition_scores=analysis.condition_scores,
        confidence=bin_confidence(materials, conditions, policy),
        signature="",
    )
    return sign(unsigned, secret)


# ------------------------------------------------------------------------------- belief


def _normalise(p: Mapping[K, float]) -> dict[K, float]:
    total = sum(p.values())
    if total <= 0.0:
        raise ValueError("belief has no probability mass left")
    return {k: v / total for k, v in p.items()}


def _entropy(probabilities: Iterable[float]) -> float:
    return -sum(p * math.log2(p) for p in probabilities if p > 0.0)


def beliefs(state: DecisionState) -> tuple[Materials, Conditions]:
    materials = {s.label: s.probability for s in state.material_scores}
    conditions = {s.label: s.probability for s in state.condition_scores}
    return materials, conditions


def _posterior(
    materials: Materials, conditions: Conditions, qid: QuestionId, answer: str
) -> tuple[Materials, Conditions]:
    """Naive-Bayes update: multiply by P(answer | class) and normalise."""
    if qid is QuestionId.Q1_RECYCLING_SYMBOL:
        return _normalise({m: p * likelihoods.q1(m, answer) for m, p in materials.items()}), (
            conditions
        )
    lik = likelihoods.condition_likelihood(qid, answer, materials)
    return materials, _normalise({c: p * lik[c] for c, p in conditions.items()})


def _answer_probability(
    materials: Materials, conditions: Conditions, qid: QuestionId, answer: str
) -> float:
    if qid is QuestionId.Q1_RECYCLING_SYMBOL:
        return sum(p * likelihoods.q1(m, answer) for m, p in materials.items())
    lik = likelihoods.condition_likelihood(qid, answer, materials)
    return sum(p * lik[c] for c, p in conditions.items())


def _update_belief(state: DecisionState, qa: QuestionAnswer, policy: SitePolicy) -> DecisionState:
    materials, conditions = _posterior(*beliefs(state), qa.question_id, qa.answer)
    return state.model_copy(
        update={
            "material_scores": [
                MaterialScore(label=s.label, probability=materials[s.label])
                for s in state.material_scores
            ],
            "condition_scores": [
                ConditionScore(label=s.label, probability=conditions[s.label])
                for s in state.condition_scores
            ],
            "confidence": bin_confidence(materials, conditions, policy),
            "asked": [*state.asked, qa],
        }
    )


# --------------------------------------------------------------------------------- bins


# Bin key for every (material, condition) pair, per policy object (the policy is kept in the value
# so its id cannot be reused while cached).
_BIN_TABLES: dict[int, tuple[SitePolicy, dict[tuple[Material, Condition], str]]] = {}


def _bin_table(policy: SitePolicy) -> dict[tuple[Material, Condition], str]:
    cached = _BIN_TABLES.get(id(policy))
    if cached is None or cached[0] is not policy:
        table = {(m, c): policy.first_match(m, c).then for m in Material for c in Condition}
        cached = _BIN_TABLES[id(policy)] = (policy, table)
    return cached[1]


def bin_distribution(
    materials: Materials, conditions: Conditions, policy: SitePolicy
) -> dict[str, float]:
    """P(bin): the policy's bin for every (material, condition) pair, weighted by its belief."""
    table = _bin_table(policy)
    dist: dict[str, float] = {}
    for m, pm in materials.items():
        for c, pc in conditions.items():
            key = table[m, c]
            dist[key] = dist.get(key, 0.0) + pm * pc
    return dist


@dataclass(frozen=True)
class Decision:
    bin_key: str
    confidence: float  # P(bin_key)
    material: Material  # the most likely (material, condition) pair leading to bin_key
    condition: Condition


def decide(materials: Materials, conditions: Conditions, policy: SitePolicy) -> Decision:
    """The most likely bin, and the most likely pair that leads to it (whose rule is shown)."""
    dist = bin_distribution(materials, conditions, policy)
    key = max(dist, key=dist.__getitem__)
    table = _bin_table(policy)
    material, condition = max(
        ((m, c) for m in materials for c in conditions if table[m, c] == key),
        key=lambda mc: materials[mc[0]] * conditions[mc[1]],
    )
    return Decision(key, min(1.0, dist[key]), material, condition)


def bin_confidence(materials: Materials, conditions: Conditions, policy: SitePolicy) -> float:
    return decide(materials, conditions, policy).confidence


def _expected_information_gain(
    materials: Materials, conditions: Conditions, policy: SitePolicy, qid: QuestionId
) -> float:
    """Entropy of P(bin) now, minus its expected entropy after hearing the answer."""
    before = _entropy(bin_distribution(materials, conditions, policy).values())
    after = 0.0
    for answer in QUESTION_BANK[qid].options:
        p_answer = _answer_probability(materials, conditions, qid, answer)
        if p_answer <= 0.0:
            continue
        post = _posterior(materials, conditions, qid, answer)
        after += p_answer * _entropy(bin_distribution(*post, policy).values())
    return before - after


def information_gain(state: DecisionState, policy: SitePolicy, qid: QuestionId) -> float:
    return _expected_information_gain(*beliefs(state), policy, qid)


# --------------------------------------------------------------------------------- gate


def _asked(state: DecisionState) -> set[QuestionId]:
    return {a.question_id for a in state.asked}


def _must_ask_q2(state: DecisionState, materials: Materials, gate: Gate) -> bool:
    """D-003: the camera cannot see whether a recyclable item is clean, so ask."""
    if not gate.require_q2:
        return False
    top = max(materials, key=materials.__getitem__)
    return top in RECYCLABLE_MATERIALS and QuestionId.Q2_CLEANED not in _asked(state)


def _possible_e_waste(
    materials: Materials, conditions: Conditions, policy: SitePolicy, gate: Gate
) -> bool:
    """WS1 handoff: e-waste is a likely runner-up, so do not settle on another bin yet.

    Only a winning e-waste bin is exempt. E-waste can be the top material while the other
    materials together send the item to another bin (e.g. general waste).
    """
    if gate.e_waste_alert is None or materials[Material.E_WASTE] <= gate.e_waste_alert:
        return False
    best = decide(materials, conditions, policy).bin_key
    return policy.bin(best).channel is not Channel.E_WASTE


def is_settled(state: DecisionState, policy: SitePolicy, gate: Gate = DEFAULT_GATE) -> bool:
    materials, conditions = beliefs(state)
    if _must_ask_q2(state, materials, gate):
        return False
    if _possible_e_waste(materials, conditions, policy, gate):
        return False
    return bin_confidence(materials, conditions, policy) >= gate.threshold


def _next_question(
    state: DecisionState, policy: SitePolicy, gate: Gate = DEFAULT_GATE
) -> QuestionId | None:
    if is_settled(state, policy, gate):
        return None
    asked = _asked(state)
    if not asked:
        return OPENING_QUESTION  # Q1 always opens (proposal 1.2)
    remaining = [q for q in FOLLOW_UP_QUESTIONS if q not in asked]
    if not remaining:
        return None
    materials, conditions = beliefs(state)
    if _must_ask_q2(state, materials, gate):
        return QuestionId.Q2_CLEANED
    if gate.follow_up == "q2_first":
        return remaining[0]
    if gate.follow_up == "q3_first":
        return remaining[-1]
    # max() keeps the first of equal gains, so ties go to Q2.
    return max(
        remaining, key=lambda q: _expected_information_gain(materials, conditions, policy, q)
    )


# ------------------------------------------------------------------------------ verdict


def _evidence_trace(
    materials: Materials,
    conditions: Conditions,
    material: Material,
    condition: Condition,
    asked: list[QuestionAnswer],
    policy: SitePolicy,
) -> list[str]:
    """What the photo showed and what the user said, in plain words."""
    trace = [f"Looks like {MATERIAL_WORDS[material]} ({materials[material]:.0%} sure)"]
    if len({_bin_table(policy)[material, c] for c in Condition}) == 1:
        trace.append("Clean or not, it goes in the same bin")
    elif condition is Condition.CLEAN:
        trace.append(f"Probably clean ({conditions[Condition.CLEAN]:.0%} likely)")
    else:
        trace.append(f"Probably not clean ({1.0 - conditions[Condition.CLEAN]:.0%} likely)")
    trace += [ANSWER_WORDS[a.question_id, a.answer] for a in asked]
    return trace


def verdict_for(state: DecisionState, policy: SitePolicy, *, uncertain: bool) -> VerdictResult:
    materials, conditions = beliefs(state)
    best = decide(materials, conditions, policy)
    material, condition, confidence = best.material, best.condition, best.confidence
    evidence = _evidence_trace(materials, conditions, material, condition, state.asked, policy)
    if uncertain:
        spec = policy.bin(policy.uncertain_fallback)
        # The confidence shown is for the bin recommended, not for the best guess that fell short.
        doubt = f"Only {confidence:.0%} sure after {len(state.asked)} questions"
        if best.bin_key != spec.key:
            doubt = f"Best guess {policy.bin(best.bin_key).label}, only {confidence:.0%} sure"
            confidence = bin_distribution(materials, conditions, policy).get(spec.key, 0.0)
        return VerdictResult(
            channel=spec.channel,
            bin_key=spec.key,
            bin_label=spec.label,
            bin_colour=spec.colour,
            reason="Still not sure after all the questions. General waste avoids contaminating "
            "the recycling.",
            rule_id="uncertain_fallback",
            rule_trace=[
                *evidence,
                doubt,
                f"When unsure, so: {spec.label}",
            ],
            source_url=policy.abstain_source_url,
            confidence=confidence,
            questions_asked=state.asked,
        )
    rule = policy.first_match(material, condition)
    spec = policy.bin(rule.then)
    rule_name = rule.id.replace("_", " ").capitalize()
    return VerdictResult(
        channel=spec.channel,
        bin_key=spec.key,
        bin_label=spec.label,
        bin_colour=spec.colour,
        reason=rule.reason,
        tip=rule.tip,
        rule_id=rule.id,
        rule_trace=[*evidence, f"{rule_name} rule, so: {spec.label}"],
        source_url=rule.source_url,
        confidence=confidence,
        questions_asked=state.asked,
    )


def abstain(policy: SitePolicy, confidence: float) -> VerdictResult:
    return VerdictResult(
        channel=Channel.ABSTAIN,
        bin_key=None,
        bin_label="No verdict",
        reason="This item is unlike anything the model was trained on.",
        rule_id="abstain_ood",
        rule_trace=["out-of-distribution"],
        source_url=policy.abstain_source_url,
        confidence=confidence,
    )


def step(
    state: DecisionState, policy: SitePolicy, gate: Gate = DEFAULT_GATE
) -> tuple[Question | None, VerdictResult | None]:
    """Either the next question to ask, or the final verdict."""
    qid = _next_question(state, policy, gate)
    if qid is not None:
        return Question.from_bank(qid), None
    return None, verdict_for(state, policy, uncertain=not is_settled(state, policy, gate))


def answer(
    state: DecisionState,
    qa: QuestionAnswer,
    secret: bytes,
    policy: SitePolicy,
    gate: Gate = DEFAULT_GATE,
) -> DecisionState:
    verify_signature(state, secret)
    expected = _next_question(state, policy, gate)
    if qa.question_id != expected:
        raise InvalidStateError(f"expected an answer to {expected}, got {qa.question_id}")
    return sign(_update_belief(state, qa, policy), secret)


def decide_offline(
    state: DecisionState,
    policy: SitePolicy,
    answers: Mapping[QuestionId, str],
    gate: Gate = DEFAULT_GATE,
) -> VerdictResult:
    """The /predict + /answer flow in one call, without signing, for offline evaluation.

    `answers` holds the answer the user would give to each question that gets asked.
    """
    while (qid := _next_question(state, policy, gate)) is not None:
        qa = QuestionAnswer(question_id=qid, answer=answers[qid])
        state = _update_belief(state, qa, policy)
    return verdict_for(state, policy, uncertain=not is_settled(state, policy, gate))


def decide_static(
    state: DecisionState,
    policy: SitePolicy,
    answers: Mapping[QuestionId, str],
    gate: Gate = DEFAULT_GATE,
) -> VerdictResult:
    """Baseline 2 (proposal Section 9): all three questions every time, then the rules.

    Same belief update, policy and uncertain fallback as the adaptive flow, so the comparison
    isolates adaptive question selection.
    """
    for qid in QUESTION_BANK:
        state = _update_belief(state, QuestionAnswer(question_id=qid, answer=answers[qid]), policy)
    return verdict_for(state, policy, uncertain=not is_settled(state, policy, gate))
