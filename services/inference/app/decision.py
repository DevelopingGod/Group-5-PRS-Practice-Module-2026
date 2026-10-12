"""Decision flow: confidence gate, adaptive questions and verdict (WS3 owns the logic).

STUB belief update: each answer raises confidence by a fixed step. WS3 replaces
`_update_belief` and `_next_question` with likelihood-based updates and entropy selection;
the signing, ordering rules and verdict building must keep satisfying the contract tests.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from uuid import UUID

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
from binbuddy_contracts.questions import OPENING_QUESTION

STUB_STEP = 0.15


class InvalidStateError(ValueError):
    pass


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


def _top_material(scores: list[MaterialScore]) -> Material:
    return max(scores, key=lambda s: s.probability).label


def _top_condition(scores: list[ConditionScore]) -> Condition:
    return max(scores, key=lambda s: s.probability).label


def _next_question(state: DecisionState) -> QuestionId | None:
    asked = {a.question_id for a in state.asked}
    if not asked:
        return OPENING_QUESTION  # Q1 always opens (proposal 1.2)
    for qid in (QuestionId.Q2_CLEANED, QuestionId.Q3_SITTING_TIME):
        if qid not in asked:
            return qid
    return None


def _update_belief(state: DecisionState, answer: QuestionAnswer) -> DecisionState:
    return state.model_copy(
        update={
            "asked": [*state.asked, answer],
            "confidence": min(1.0, state.confidence + STUB_STEP),
        }
    )


def verdict_for(state: DecisionState, policy: SitePolicy, *, uncertain: bool) -> VerdictResult:
    material = _top_material(state.material_scores)
    condition = _top_condition(state.condition_scores)
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
            rule_trace=[f"confidence={state.confidence:.2f}", "bank exhausted", "fallback"],
            source_url=policy.abstain_source_url,
            confidence=state.confidence,
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
            f"material={material.value}",
            f"condition={condition.value}",
            f"rule={rule.id}",
        ],
        source_url=rule.source_url,
        confidence=state.confidence,
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


def step(state: DecisionState, policy: SitePolicy) -> tuple[Question | None, VerdictResult | None]:
    """Either the next question to ask, or the final verdict."""
    if state.confidence >= CONFIDENCE_THRESHOLD:
        return None, verdict_for(state, policy, uncertain=False)
    qid = _next_question(state)
    if qid is None:
        return None, verdict_for(state, policy, uncertain=True)
    return Question.from_bank(qid), None


def answer(state: DecisionState, qa: QuestionAnswer, secret: bytes) -> DecisionState:
    verify_signature(state, secret)
    expected = _next_question(state)
    if qa.question_id != expected:
        raise InvalidStateError(f"expected an answer to {expected}, got {qa.question_id}")
    return sign(_update_belief(state, qa), secret)
