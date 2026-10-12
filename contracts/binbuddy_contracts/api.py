"""Model API contract (FastAPI service on the Hugging Face Space <-> Next.js route handlers).

Endpoints (see openapi.py): GET /health, GET /version, POST /predict, POST /answer, POST /verify.
The API is stateless: the signed DecisionState returned by /predict is sent back to /answer.
"""

from __future__ import annotations

import base64
import binascii
import math
from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from binbuddy_contracts.constants import (
    DEFAULT_SITE_POLICY,
    EMBEDDING_DIM,
    MAX_IMAGE_BYTES,
    PROBABILITY_SUM_TOLERANCE,
)
from binbuddy_contracts.enums import (
    Channel,
    Condition,
    ErrorCode,
    Material,
    QualityIssue,
    QuestionId,
    ResponseStatus,
)
from binbuddy_contracts.questions import QUESTION_BANK

SLUG_PATTERN = r"^[a-z0-9][a-z0-9_-]{1,39}$"
PHASH_PATTERN = r"^[0-9a-f]{16}$"

Probability = Annotated[float, Field(ge=0.0, le=1.0)]
Slug = Annotated[str, Field(pattern=SLUG_PATTERN)]
PHash = Annotated[str, Field(pattern=PHASH_PATTERN, description="64-bit perceptual hash, hex")]


class Strict(BaseModel):
    """Base model: unknown fields are rejected so typos never pass silently."""

    model_config = ConfigDict(extra="forbid")


def _check_image_b64(value: str) -> str:
    try:
        raw = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("image_base64 is not valid base64") from exc
    if len(raw) == 0:
        raise ValueError("image_base64 is empty")
    if len(raw) > MAX_IMAGE_BYTES:
        raise ValueError(f"image exceeds {MAX_IMAGE_BYTES} bytes")
    return value


def _check_embedding(values: list[float]) -> list[float]:
    if len(values) != EMBEDDING_DIM:
        raise ValueError(f"embedding must have {EMBEDDING_DIM} values, got {len(values)}")
    norm = math.sqrt(sum(v * v for v in values))
    if abs(norm - 1.0) > 1e-3:
        raise ValueError(f"embedding must be L2-normalised (norm={norm:.4f})")
    return values


# --------------------------------------------------------------------------- scores


class MaterialScore(Strict):
    label: Material
    probability: Probability


class ConditionScore(Strict):
    label: Condition
    probability: Probability


def _check_distribution(labels: list[str], probs: list[float], expected: set[str]) -> None:
    if len(labels) != len(set(labels)):
        raise ValueError("each label must appear exactly once")
    if set(labels) != expected:
        missing = sorted(expected - set(labels))
        raise ValueError(f"scores must cover every label; missing {missing}")
    total = sum(probs)
    if abs(total - 1.0) > PROBABILITY_SUM_TOLERANCE:
        raise ValueError(f"probabilities must sum to 1 (got {total:.4f})")


class VisionAnalysis(Strict):
    """What the vision ensemble saw (WS1 + WS2)."""

    material_scores: list[MaterialScore]
    condition_scores: list[ConditionScore]
    confidence: Probability = Field(description="Calibrated confidence in the current decision")
    is_ood: bool = Field(description="True if the item is unlike anything in the training data")
    embedding: list[float] = Field(description=f"L2-normalised, length {EMBEDDING_DIM}")
    phash: PHash
    heatmap_png_base64: str | None = Field(
        default=None, description="Grad-CAM overlay as a base64 PNG"
    )

    @field_validator("embedding")
    @classmethod
    def _embedding(cls, v: list[float]) -> list[float]:
        return _check_embedding(v)

    @model_validator(mode="after")
    def _distributions(self) -> VisionAnalysis:
        _check_distribution(
            [s.label.value for s in self.material_scores],
            [s.probability for s in self.material_scores],
            {m.value for m in Material},
        )
        _check_distribution(
            [s.label.value for s in self.condition_scores],
            [s.probability for s in self.condition_scores],
            {c.value for c in Condition},
        )
        return self


class QualityReport(Strict):
    ok: bool
    blur_score: float = Field(ge=0.0, description="Variance of the Laplacian; higher is sharper")
    brightness: Probability = Field(description="Mean brightness, 0 = black, 1 = white")
    issues: list[QualityIssue] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistent(self) -> QualityReport:
        if self.ok and self.issues:
            raise ValueError("quality.ok is true but issues are listed")
        if not self.ok and not self.issues:
            raise ValueError("quality.ok is false but no issues are listed")
        return self


# ------------------------------------------------------------------------ questions


class Question(Strict):
    id: QuestionId
    text: str
    options: list[str]

    @model_validator(mode="after")
    def _matches_bank(self) -> Question:
        spec = QUESTION_BANK[self.id]
        if self.text != spec.text or tuple(self.options) != spec.options:
            raise ValueError(f"question {self.id} must match the fixed question bank")
        return self

    @classmethod
    def from_bank(cls, qid: QuestionId) -> Question:
        spec = QUESTION_BANK[qid]
        return cls(id=spec.id, text=spec.text, options=list(spec.options))


class QuestionAnswer(Strict):
    question_id: QuestionId
    answer: str

    @model_validator(mode="after")
    def _allowed(self) -> QuestionAnswer:
        if self.answer not in QUESTION_BANK[self.question_id].options:
            raise ValueError(f"'{self.answer}' is not a valid answer to {self.question_id}")
        return self


class DecisionState(Strict):
    """Everything needed to continue a decision. Signed by the API; tampering is rejected."""

    scan_id: UUID
    site_policy: Slug
    material_scores: list[MaterialScore]
    condition_scores: list[ConditionScore]
    confidence: Probability
    asked: list[QuestionAnswer] = Field(default_factory=list)
    signature: str = Field(description="HMAC-SHA256 over the other fields, hex")

    @model_validator(mode="after")
    def _no_repeats(self) -> DecisionState:
        ids = [a.question_id for a in self.asked]
        if len(ids) != len(set(ids)):
            raise ValueError("a question cannot be asked twice")
        return self


# -------------------------------------------------------------------------- verdict


class VerdictResult(Strict):
    channel: Channel
    bin_key: Slug | None = Field(description="Bin in the active site policy; null when abstaining")
    bin_label: str
    bin_colour: str | None = Field(default=None, pattern=r"^#[0-9A-Fa-f]{6}$")
    reason: str
    tip: str | None = None
    rule_id: Slug
    rule_trace: list[str] = Field(min_length=1)
    source_url: str | None = None
    confidence: Probability
    questions_asked: list[QuestionAnswer] = Field(default_factory=list)

    @model_validator(mode="after")
    def _abstain_rules(self) -> VerdictResult:
        if self.channel is Channel.ABSTAIN:
            if self.bin_key is not None:
                raise ValueError("an abstain verdict has no bin_key")
            if not self.source_url:
                raise ValueError("an abstain verdict must link to NEA guidance (source_url)")
        elif self.bin_key is None:
            raise ValueError("a non-abstain verdict needs a bin_key")
        return self


# ------------------------------------------------------------------------ endpoints


class HealthResponse(Strict):
    status: Literal["ok"]
    model_loaded: bool


class PolicyVersion(Strict):
    site: Slug
    version: str


class VersionResponse(Strict):
    model_repo: str
    model_revision: str
    contracts_version: str
    embedding_dim: int
    policies: list[PolicyVersion]


class PredictRequest(Strict):
    image_base64: str
    site_policy: Slug = DEFAULT_SITE_POLICY

    @field_validator("image_base64")
    @classmethod
    def _image(cls, v: str) -> str:
        return _check_image_b64(v)


class _StepResponse(Strict):
    status: ResponseStatus
    next_question: Question | None = None
    verdict: VerdictResult | None = None
    state: DecisionState | None = None

    def _check_step(self) -> None:
        if self.status is ResponseStatus.QUESTION:
            if self.next_question is None or self.state is None or self.verdict is not None:
                raise ValueError("status 'question' needs next_question and state, no verdict")
        elif self.status is ResponseStatus.VERDICT and (
            self.verdict is None or self.next_question is not None
        ):
            raise ValueError("status 'verdict' needs a verdict and no next_question")


class PredictResponse(_StepResponse):
    scan_id: UUID
    model_revision: str
    quality: QualityReport
    analysis: VisionAnalysis | None = None

    @model_validator(mode="after")
    def _consistent(self) -> PredictResponse:
        if self.status is ResponseStatus.RETAKE:
            if self.quality.ok:
                raise ValueError("status 'retake' requires a failed quality check")
            if self.analysis or self.next_question or self.verdict or self.state:
                raise ValueError("status 'retake' carries no analysis, question, verdict or state")
            return self
        if not self.quality.ok:
            raise ValueError("a failed quality check must return status 'retake'")
        if self.analysis is None:
            raise ValueError("analysis is required unless status is 'retake'")
        self._check_step()
        return self


class AnswerRequest(Strict):
    state: DecisionState
    answer: QuestionAnswer

    @model_validator(mode="after")
    def _not_repeated(self) -> AnswerRequest:
        if any(a.question_id == self.answer.question_id for a in self.state.asked):
            raise ValueError("this question has already been answered")
        return self


class AnswerResponse(_StepResponse):
    @model_validator(mode="after")
    def _consistent(self) -> AnswerResponse:
        if self.status is ResponseStatus.RETAKE:
            raise ValueError("/answer never returns status 'retake'")
        self._check_step()
        return self


class BinQrPayload(Strict):
    """JSON encoded in the QR code on each demonstration bin."""

    v: Literal[1] = 1
    bin_id: Slug = Field(description="Unique physical bin, e.g. 'demo-hdb-01'")
    site_policy: Slug
    bin_key: Slug = Field(description="Bin key in that site policy, e.g. 'blue_bin'")
    location_label: str = Field(min_length=1, max_length=60)


class VerifyRequest(Strict):
    scan_id: UUID
    image_base64: str
    qr: BinQrPayload
    recommended_site_policy: Slug
    recommended_bin_key: Slug
    original_embedding: list[float]
    original_phash: PHash
    original_created_at: datetime
    submitted_at: datetime
    known_phashes: list[PHash] = Field(
        default_factory=list, description="Recent hashes for this user, for duplicate detection"
    )

    @field_validator("image_base64")
    @classmethod
    def _image(cls, v: str) -> str:
        return _check_image_b64(v)

    @field_validator("original_embedding")
    @classmethod
    def _embedding(cls, v: list[float]) -> list[float]:
        return _check_embedding(v)


class VerifyChecks(Strict):
    image_match: bool
    bin_matches_verdict: bool
    within_time_window: bool
    not_duplicate: bool


class VerifyResponse(Strict):
    passed: bool
    similarity: float = Field(ge=-1.0, le=1.0)
    checks: VerifyChecks
    reason: str
    phash: PHash

    @model_validator(mode="after")
    def _passed_matches_checks(self) -> VerifyResponse:
        if self.passed != all(self.checks.model_dump().values()):
            raise ValueError("passed must be true exactly when every check passes")
        return self


class ErrorResponse(Strict):
    code: ErrorCode
    message: str
