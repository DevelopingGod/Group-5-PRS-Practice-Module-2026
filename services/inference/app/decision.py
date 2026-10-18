"""Decision flow: confidence gate, adaptive questions and verdict (WS3 owns the logic).

Belief. DecisionState keeps two distributions: material (calibrated, from the vision model) and
condition (uniform from the vision model, which never judges cleanliness). Each answer updates
one of them by naive-Bayes evidence fusion with the likelihoods in app/likelihoods.py:
Q1 updates the material, Q2 and Q3 update the condition. `state.confidence` is the top material
probability.

Confidence in a verdict is measured on the bin, not the material. Every (material, condition)
pair maps to a bin through the site policy, so P(bin) is the sum of P(material) * P(condition)
over the pairs that lead to it. The verdict follows the most likely pair, and its confidence is
the probability of that pair's bin. A confident styrofoam is settled without questions (every
condition leads to general waste); a confident plastic is not (its condition decides the bin).

Gate (proposal 1.2 and the WS1 handoff):
  - settled when P(verdict bin) >= threshold, unless one of the safety rules below applies;
  - D-003: a recyclable top material is never settled before Q2 has been asked;
  - e-waste rule: not settled on another bin while P(e_waste) > E_WASTE_ALERT (redundant at a
    0.80 gate, which no other bin can reach then; it matters if the gate is tuned lower);
  - Q1 always opens; then the remaining question with the largest expected entropy reduction
    of the bin distribution; at most three questions;
  - still not settled when the bank is exhausted: general waste plus the NEA link.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Final, TypeVar
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

K = TypeVar("K")
Materials = dict[Material, float]
Conditions = dict[Condition, float]


@dataclass(frozen=True)
class Gate:
    """Gate settings. The API always uses DEFAULT_GATE; other values are for offline tuning."""

    threshold: float = CONFIDENCE_THRESHOLD
    e_waste_alert: float | None = E_WASTE_ALERT  # None switches the e-waste rule off


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


def new_state(scan_id: UUID, site: str, analysis: VisionAnalysis, secret: bytes) -> DecisionState:
    unsigned = DecisionState(
        scan_id=scan_id,
        site_policy=site,
        material_scores=analysis.material_scores,
        condition_scores=analysis.condition_scores,
        confidence=analysis.confidence,
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


def _update_belief(state: DecisionState, qa: QuestionAnswer) -> DecisionState:
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
            "confidence": max(materials.values()),
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


def most_likely_pair(materials: Materials, conditions: Conditions) -> tuple[Material, Condition]:
    return max(materials, key=materials.__getitem__), max(conditions, key=conditions.__getitem__)


def bin_confidence(materials: Materials, conditions: Conditions, policy: SitePolicy) -> float:
    """Probability of the bin the verdict would name (the most likely pair's bin)."""
    material, condition = most_likely_pair(materials, conditions)
    key = _bin_table(policy)[material, condition]
    return min(1.0, bin_distribution(materials, conditions, policy)[key])


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


def _must_ask_q2(state: DecisionState, materials: Materials) -> bool:
    """D-003: the camera cannot see whether a recyclable item is clean, so ask."""
    top = max(materials, key=materials.__getitem__)
    return top in RECYCLABLE_MATERIALS and QuestionId.Q2_CLEANED not in _asked(state)


def _possible_e_waste(materials: Materials, gate: Gate) -> bool:
    """WS1 handoff: e-waste is a likely runner-up, so do not settle on another bin yet."""
    if gate.e_waste_alert is None:
        return False
    top = max(materials, key=materials.__getitem__)
    return top is not Material.E_WASTE and materials[Material.E_WASTE] > gate.e_waste_alert


def is_settled(state: DecisionState, policy: SitePolicy, gate: Gate = DEFAULT_GATE) -> bool:
    materials, conditions = beliefs(state)
    if _must_ask_q2(state, materials):
        return False
    if _possible_e_waste(materials, gate):
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
    if _must_ask_q2(state, materials):
        return QuestionId.Q2_CLEANED
    # max() keeps the first of equal gains, so ties go to Q2.
    return max(
        remaining, key=lambda q: _expected_information_gain(materials, conditions, policy, q)
    )


# ------------------------------------------------------------------------------ verdict


def verdict_for(state: DecisionState, policy: SitePolicy, *, uncertain: bool) -> VerdictResult:
    materials, conditions = beliefs(state)
    material, condition = most_likely_pair(materials, conditions)
    confidence = bin_confidence(materials, conditions, policy)
    answers = [f"{a.question_id.value}={a.answer}" for a in state.asked]
    if uncertain:
        spec = policy.bin(policy.uncertain_fallback)
        return VerdictResult(
            channel=spec.channel,
            bin_key=spec.key,
            bin_label=spec.label,
            bin_colour=spec.colour,
            reason="Still uncertain after all questions; general waste avoids contaminating "
            "the blue bin.",
            rule_id="uncertain_fallback",
            rule_trace=[
                f"material={material.value} ({materials[material]:.2f})",
                f"condition={condition.value} ({conditions[condition]:.2f})",
                *answers,
                f"confidence={confidence:.2f}",
                "bank exhausted",
                "fallback",
            ],
            source_url=policy.abstain_source_url,
            confidence=confidence,
            questions_asked=state.asked,
        )
    rule = policy.first_match(material, condition)
    spec = policy.bin(rule.then)
    return VerdictResult(
        channel=spec.channel,
        bin_key=spec.key,
        bin_label=spec.label,
        bin_colour=spec.colour,
        reason=rule.reason,
        tip=rule.tip,
        rule_id=rule.id,
        rule_trace=[
            f"material={material.value} ({materials[material]:.2f})",
            f"condition={condition.value} ({conditions[condition]:.2f})",
            *answers,
            f"rule={rule.id}",
        ],
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
    return sign(_update_belief(state, qa), secret)


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
        state = _update_belief(state, QuestionAnswer(question_id=qid, answer=answers[qid]))
    return verdict_for(state, policy, uncertain=not is_settled(state, policy, gate))
