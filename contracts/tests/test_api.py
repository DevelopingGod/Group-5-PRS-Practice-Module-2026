import base64
import uuid
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from binbuddy_contracts import examples
from binbuddy_contracts.api import (
    AnswerRequest,
    BinQrPayload,
    DecisionState,
    PredictRequest,
    PredictResponse,
    QualityReport,
    Question,
    QuestionAnswer,
    VerdictResult,
    VerifyChecks,
    VerifyResponse,
    VisionAnalysis,
)
from binbuddy_contracts.constants import EMBEDDING_DIM
from binbuddy_contracts.enums import (
    Channel,
    Condition,
    Material,
    QualityIssue,
    QuestionId,
    ResponseStatus,
)

OK = QualityReport(ok=True, blur_score=250.0, brightness=0.5)
BAD = QualityReport(ok=False, blur_score=3.0, brightness=0.5, issues=[QualityIssue.TOO_BLURRY])


def _verdict(channel: Channel = Channel.RECYCLE) -> VerdictResult:
    return VerdictResult(
        channel=channel,
        bin_key="blue_bin",
        bin_label="Blue recycling bin",
        reason="Clean metal goes in the blue bin.",
        rule_id="clean_recyclables",
        rule_trace=["material=metal", "condition=clean", "rule=clean_recyclables"],
        confidence=0.9,
    )


def _state() -> DecisionState:
    return DecisionState(
        scan_id=uuid.uuid4(),
        site_policy="hdb_default",
        material_scores=examples.material_scores(Material.PLASTIC, 0.5),
        condition_scores=examples.condition_scores(Condition.CLEAN, 0.5),
        confidence=0.5,
        signature="0" * 64,
    )


def test_confident_verdict_response_is_valid() -> None:
    PredictResponse(
        scan_id=uuid.uuid4(),
        model_revision="v0.1-baseline",
        status=ResponseStatus.VERDICT,
        quality=OK,
        analysis=examples.analysis(),
        verdict=_verdict(),
    )


def test_retake_must_not_carry_analysis() -> None:
    with pytest.raises(ValidationError, match="retake"):
        PredictResponse(
            scan_id=uuid.uuid4(),
            model_revision="x",
            status=ResponseStatus.RETAKE,
            quality=BAD,
            analysis=examples.analysis(),
        )


def test_question_status_needs_question_and_state() -> None:
    with pytest.raises(ValidationError, match="next_question"):
        PredictResponse(
            scan_id=uuid.uuid4(),
            model_revision="x",
            status=ResponseStatus.QUESTION,
            quality=OK,
            analysis=examples.analysis(confidence=0.4),
        )


def test_probabilities_must_sum_to_one() -> None:
    scores = examples.material_scores(Material.GLASS)
    scores[0] = scores[0].model_copy(update={"probability": 0.99})
    with pytest.raises(ValidationError, match="sum to 1"):
        VisionAnalysis(**{**examples.analysis().model_dump(), "material_scores": scores})


def test_embedding_length_enforced() -> None:
    data = examples.analysis().model_dump()
    data["embedding"] = [1.0] + [0.0] * (EMBEDDING_DIM - 2)
    with pytest.raises(ValidationError, match="embedding"):
        VisionAnalysis(**data)


def test_answer_must_be_a_listed_option() -> None:
    with pytest.raises(ValidationError, match="not a valid answer"):
        QuestionAnswer(question_id=QuestionId.Q3_SITTING_TIME, answer="yes")


def test_question_text_must_match_bank() -> None:
    q = Question.from_bank(QuestionId.Q2_CLEANED)
    with pytest.raises(ValidationError, match="fixed question bank"):
        Question(id=q.id, text="Is it clean?", options=q.options)


def test_question_cannot_be_answered_twice() -> None:
    state = _state()
    state.asked.append(QuestionAnswer(question_id=QuestionId.Q1_RECYCLING_SYMBOL, answer="yes"))
    with pytest.raises(ValidationError, match="already been answered"):
        AnswerRequest(
            state=state,
            answer=QuestionAnswer(question_id=QuestionId.Q1_RECYCLING_SYMBOL, answer="no"),
        )


def test_abstain_needs_source_and_no_bin() -> None:
    with pytest.raises(ValidationError, match="NEA guidance"):
        VerdictResult(
            channel=Channel.ABSTAIN,
            bin_key=None,
            bin_label="No verdict",
            reason="Unfamiliar item",
            rule_id="abstain_ood",
            rule_trace=["ood"],
            confidence=0.2,
        )


def test_verify_passed_must_match_checks() -> None:
    checks = VerifyChecks(
        image_match=True, bin_matches_verdict=False, within_time_window=True, not_duplicate=True
    )
    with pytest.raises(ValidationError, match="every check"):
        VerifyResponse(passed=True, similarity=0.9, checks=checks, reason="x", phash="0" * 16)


def test_bin_qr_payload() -> None:
    BinQrPayload(
        bin_id="demo-hdb-01", site_policy="hdb_default", bin_key="blue_bin", location_label="Blk 1"
    )
    with pytest.raises(ValidationError):
        BinQrPayload(
            bin_id="Demo HDB", site_policy="hdb_default", bin_key="blue_bin", location_label="x"
        )


def test_image_must_be_base64() -> None:
    with pytest.raises(ValidationError, match="base64"):
        PredictRequest(image_base64="not base64!!")
    PredictRequest(image_base64=base64.b64encode(b"\x89PNG....").decode())
    _ = datetime.now(UTC)
