"""WS3 decision flow: belief updates, question order, the gate and verdicts."""

from __future__ import annotations

import random
from itertools import product
from pathlib import Path
from uuid import uuid4

import pytest

from app import decision, likelihoods
from binbuddy_contracts.api import (
    ConditionScore,
    DecisionState,
    MaterialScore,
    QuestionAnswer,
    VerdictResult,
)
from binbuddy_contracts.enums import Channel, Condition, Material, QuestionId
from binbuddy_contracts.policy import SitePolicy, load_policies
from binbuddy_contracts.questions import OPENING_QUESTION, QUESTION_BANK

SECRET = b"test-secret"
POLICIES = load_policies(Path(__file__).resolve().parents[3] / "policies")
HDB = POLICIES["hdb_default"]


def make_state(
    materials: dict[Material, float],
    conditions: dict[Condition, float] | None = None,
    site: str = "hdb_default",
) -> DecisionState:
    """A signed state; unlisted materials share the remaining probability."""
    rest = (1.0 - sum(materials.values())) / (len(Material) - len(materials))
    mp = {m: materials.get(m, rest) for m in Material}
    cp = conditions or {c: 1.0 / len(Condition) for c in Condition}
    return decision.sign(
        DecisionState(
            scan_id=uuid4(),
            site_policy=site,
            material_scores=[MaterialScore(label=m, probability=p) for m, p in mp.items()],
            condition_scores=[ConditionScore(label=c, probability=p) for c, p in cp.items()],
            confidence=decision.bin_confidence(mp, cp, POLICIES[site]),
            signature="",
        ),
        SECRET,
    )


def run(
    state: DecisionState, answers: dict[QuestionId, str], policy: SitePolicy = HDB
) -> tuple[list[QuestionId], VerdictResult]:
    """Answer every question from `answers` until a verdict."""
    asked: list[QuestionId] = []
    question, verdict = decision.step(state, policy)
    while question is not None:
        asked.append(question.id)
        qa = QuestionAnswer(question_id=question.id, answer=answers[question.id])
        state = decision.answer(state, qa, SECRET, policy)
        question, verdict = decision.step(state, policy)
    assert verdict is not None
    return asked, verdict


Q1, Q2, Q3 = QuestionId.Q1_RECYCLING_SYMBOL, QuestionId.Q2_CLEANED, QuestionId.Q3_SITTING_TIME


# ------------------------------------------------------------------------- likelihoods


@pytest.mark.parametrize(("name", "row"), sorted(likelihoods.all_rows().items()))
def test_likelihood_rows_are_distributions(name: str, row: dict[str, float]) -> None:
    question = {"q1": Q1, "q2": Q2, "q3": Q3}[name.split("/")[0]]
    assert set(row) == set(QUESTION_BANK[question].options)
    assert sum(row.values()) == pytest.approx(1.0)
    assert min(row.values()) >= likelihoods.LIKELIHOOD_FLOOR


def test_every_material_and_condition_has_a_row() -> None:
    assert set(likelihoods.Q1_GIVEN_MATERIAL) == set(Material)
    assert set(likelihoods.Q2_GIVEN_CONDITION) == set(Condition)


def test_q3_is_stronger_evidence_for_perishable_contents() -> None:
    late = "more_than_1_day"
    carton = likelihoods.condition_likelihood(Q3, late, {Material.BEVERAGE_CARTON: 1.0})
    paper = likelihoods.condition_likelihood(Q3, late, {Material.PAPER: 1.0})
    ratio = Condition.FOOD_RESIDUE, Condition.CLEAN
    assert carton[ratio[0]] / carton[ratio[1]] > paper[ratio[0]] / paper[ratio[1]] > 1.0


# ---------------------------------------------------------------------------- belief


def test_update_keeps_distributions_valid() -> None:
    state = make_state({Material.PLASTIC: 0.6})
    for qid, ans in ((Q1, "yes"), (Q2, "no"), (Q3, "same_day")):
        state = decision._update_belief(state, QuestionAnswer(question_id=qid, answer=ans), HDB)
        assert sum(s.probability for s in state.material_scores) == pytest.approx(1.0)
        assert sum(s.probability for s in state.condition_scores) == pytest.approx(1.0)
    assert [a.question_id for a in state.asked] == [Q1, Q2, Q3]


def test_corroborating_answer_raises_confidence() -> None:
    state = make_state({Material.PLASTIC: 0.6, Material.TEXTILE: 0.3})
    after = decision._update_belief(state, QuestionAnswer(question_id=Q1, answer="yes"), HDB)
    assert after.confidence > state.confidence
    materials, _ = decision.beliefs(after)
    assert materials[Material.TEXTILE] < 0.3


def test_q2_moves_the_condition_belief() -> None:
    state = make_state({Material.PLASTIC: 0.9})
    yes = decision._update_belief(state, QuestionAnswer(question_id=Q2, answer="yes"), HDB)
    no = decision._update_belief(state, QuestionAnswer(question_id=Q2, answer="no"), HDB)
    assert decision.beliefs(yes)[1][Condition.CLEAN] > 0.5
    assert decision.beliefs(no)[1][Condition.CLEAN] < 0.05


# --------------------------------------------------------------------------- questions


def test_q1_always_opens() -> None:
    for materials in ({Material.PLASTIC: 0.99}, {Material.PAPER: 0.3}, {Material.E_WASTE: 0.5}):
        question, _ = decision.step(make_state(materials), HDB)
        assert question is not None and question.id is OPENING_QUESTION


def test_confident_recyclable_still_asks_q2() -> None:
    """D-003: the camera cannot see whether a container is clean."""
    asked, verdict = run(make_state({Material.METAL: 0.99}), {Q1: "yes", Q2: "yes", Q3: "same_day"})
    assert asked[:2] == [Q1, Q2]
    assert verdict.bin_key == "blue_bin"


def test_q2_is_asked_even_when_the_bin_is_already_confident() -> None:
    clean = {c: (0.97 if c is Condition.CLEAN else 0.01) for c in Condition}
    state = make_state({Material.PLASTIC: 0.99}, clean)
    assert not decision.is_settled(state, HDB)
    q1_done = decision.answer(state, QuestionAnswer(question_id=Q1, answer="yes"), SECRET, HDB)
    question, _ = decision.step(q1_done, HDB)
    assert question is not None and question.id is Q2


def test_follow_up_is_the_most_informative_question() -> None:
    # Top material is not recyclable, so the entropy rule (not D-003) picks the follow-up.
    state = make_state({Material.TEXTILE: 0.40, Material.PLASTIC: 0.35})
    state = decision.answer(state, QuestionAnswer(question_id=Q1, answer="no"), SECRET, HDB)
    question, _ = decision.step(state, HDB)
    assert question is not None
    gains = {q: decision.information_gain(state, HDB, q) for q in (Q2, Q3)}
    assert question.id is max(gains, key=gains.__getitem__)


def test_q3_follows_an_unsure_q2() -> None:
    answers = {Q1: "yes", Q2: "not_sure", Q3: "same_day"}
    asked, _ = run(make_state({Material.PLASTIC: 0.95}), answers)
    assert asked == [Q1, Q2, Q3]


def test_never_more_than_three_questions() -> None:
    rng = random.Random(0)
    for _ in range(200):
        top = rng.choice(list(Material))
        state = make_state({top: rng.uniform(0.3, 0.99)})
        answers = {q: rng.choice(QUESTION_BANK[q].options) for q in (Q1, Q2, Q3)}
        asked, verdict = run(state, answers)
        assert len(asked) <= 3 and len(set(asked)) == len(asked)
        assert verdict.questions_asked == [] or verdict.questions_asked[0].question_id is Q1


def test_answer_to_the_wrong_question_is_rejected() -> None:
    state = make_state({Material.PLASTIC: 0.6})
    with pytest.raises(decision.InvalidStateError):
        decision.answer(state, QuestionAnswer(question_id=Q2, answer="yes"), SECRET, HDB)


def test_tampered_state_is_rejected() -> None:
    state = make_state({Material.PLASTIC: 0.6}).model_copy(update={"confidence": 0.99})
    with pytest.raises(decision.InvalidStateError):
        decision.answer(state, QuestionAnswer(question_id=Q1, answer="yes"), SECRET, HDB)


# ----------------------------------------------------------------------------- verdicts


def test_clean_plastic_goes_to_the_blue_bin() -> None:
    asked, verdict = run(make_state({Material.PLASTIC: 0.95}), {Q1: "yes", Q2: "yes", Q3: "x"})
    assert asked == [Q1, Q2]
    assert verdict.bin_key == "blue_bin" and verdict.confidence >= 0.8


def test_unrinsed_container_is_general_waste_with_a_tip() -> None:
    _, verdict = run(make_state({Material.PLASTIC: 0.95}), {Q1: "yes", Q2: "no", Q3: "x"})
    assert verdict.bin_key == "general_waste"
    assert verdict.tip is not None
    assert verdict.rule_trace == [
        "Looks like plastic (98% sure)",
        "Probably not clean (99% likely)",
        "You see a recycling symbol",
        "Not rinsed",
        "Container food residue rule, so: General waste",
    ]


def test_styrofoam_needs_no_questions_and_is_general_waste() -> None:
    asked, verdict = run(make_state({Material.STYROFOAM: 0.9}), {})
    assert asked == []
    assert verdict.rule_id == "styrofoam" and verdict.bin_key == "general_waste"


def test_styrofoam_with_a_symbol_is_never_blue_bin() -> None:
    state = make_state({Material.STYROFOAM: 0.6, Material.PLASTIC: 0.3})
    _, verdict = run(state, {Q1: "yes", Q2: "yes", Q3: "under_1_hour"})
    assert verdict.bin_key == "general_waste"


def test_confident_e_waste_is_settled() -> None:
    asked, verdict = run(make_state({Material.E_WASTE: 0.95}), {})
    assert asked == [] and verdict.channel is Channel.E_WASTE


def test_possible_e_waste_blocks_a_lower_gate() -> None:
    state = make_state({Material.OTHER_GENERAL: 0.7, Material.E_WASTE: 0.25})
    low = decision.Gate(threshold=0.6)
    assert not decision.is_settled(state, HDB, low)
    assert decision.is_settled(state, HDB, decision.Gate(threshold=0.6, e_waste_alert=None))


def test_top_e_waste_material_does_not_bypass_the_e_waste_rule() -> None:
    # E-waste is the top material, yet the rest adds up to general waste (0.65).
    state = make_state(
        {
            Material.E_WASTE: 0.35,
            Material.STYROFOAM: 0.30,
            Material.FOOD_WASTE: 0.20,
            Material.OTHER_GENERAL: 0.15,
        }
    )
    low = decision.Gate(threshold=0.6)
    assert decision.bin_confidence(*decision.beliefs(state), HDB) >= 0.6
    assert not decision.is_settled(state, HDB, low)
    assert decision.is_settled(state, HDB, decision.Gate(threshold=0.6, e_waste_alert=None))


def test_still_uncertain_falls_back_to_general_waste_with_nea_link() -> None:
    state = make_state({Material.PLASTIC: 0.5})
    _, verdict = run(state, {Q1: "not_sure", Q2: "not_sure", Q3: "under_1_hour"})
    assert verdict.rule_id == "uncertain_fallback"
    assert verdict.bin_key == HDB.uncertain_fallback
    assert verdict.source_url == HDB.abstain_source_url


def test_fallback_confidence_is_for_the_bin_recommended() -> None:
    # Rinsed but left since earlier today: the blue bin is the best guess, yet not sure enough.
    state = make_state({Material.PLASTIC: 0.6})
    _, verdict = run(state, {Q1: "yes", Q2: "yes", Q3: "same_day"})
    assert verdict.rule_id == "uncertain_fallback"
    assert verdict.rule_trace[-2] == "Best guess Blue recycling bin, only 77% sure"
    assert verdict.confidence == pytest.approx(0.21, abs=0.01)


@pytest.mark.parametrize("site", sorted(POLICIES))
def test_engine_matches_first_match_for_every_pair(site: str) -> None:
    policy = POLICIES[site]
    for material, condition in product(Material, Condition):
        conditions = {c: (0.97 if c is condition else 0.01) for c in Condition}
        state = make_state({material: 0.991}, conditions, site=site)
        verdict = decision.verdict_for(state, policy, uncertain=False)
        assert verdict.rule_id == policy.first_match(material, condition).id


def test_campus_sends_clean_cans_to_the_cans_bin() -> None:
    campus = POLICIES["campus_example"]
    state = make_state({Material.METAL: 0.95}, site="campus_example")
    _, verdict = run(state, {Q1: "yes", Q2: "yes", Q3: "under_1_hour"}, campus)
    assert verdict.bin_key == "cans_glass_bin"


def test_unsure_condition_leans_to_general_waste() -> None:
    """Condition unknown: the most likely bin is general waste, never a tie-break to clean."""
    materials, conditions = decision.beliefs(make_state({Material.PLASTIC: 0.95}))
    best = decision.decide(materials, conditions, HDB)
    assert best.bin_key == "general_waste" and best.condition is not Condition.CLEAN


def test_state_confidence_tracks_the_verdict_bin() -> None:
    """The app's confidence meter reads state.confidence; it must move towards the verdict."""
    state = make_state({Material.PLASTIC: 0.95})
    for qid, ans in ((Q1, "yes"), (Q2, "yes")):
        state = decision.answer(state, QuestionAnswer(question_id=qid, answer=ans), SECRET, HDB)
    _, verdict = decision.step(state, HDB)
    assert verdict is not None and verdict.bin_key == "blue_bin"
    assert state.confidence == pytest.approx(verdict.confidence)
