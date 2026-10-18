"""Bin Buddy model API (deployed on Render from render.yaml; runs anywhere Docker runs).

Routes and models come from the contracts package; tests/test_contract.py fails if this
service drifts from contracts/generated/openapi.json.
"""

from __future__ import annotations

import uuid
from functools import lru_cache

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from PIL import Image

from app import decision, vision
from app.quality import InvalidImageError, check_quality, decode_image
from app.settings import Settings, load_settings
from app.verify import verify as run_verify
from binbuddy_contracts.api import (
    AnswerRequest,
    AnswerResponse,
    ErrorResponse,
    HealthResponse,
    PolicyVersion,
    PredictRequest,
    PredictResponse,
    VerifyRequest,
    VerifyResponse,
    VersionResponse,
    VisionAnalysis,
)
from binbuddy_contracts.constants import CONTRACTS_VERSION, EMBEDDING_DIM
from binbuddy_contracts.enums import ErrorCode, ResponseStatus
from binbuddy_contracts.openapi import ERRORS
from binbuddy_contracts.policy import SitePolicy, load_policies

app = FastAPI(title="Bin Buddy model API", version=CONTRACTS_VERSION)


class ApiError(Exception):
    def __init__(self, status: int, code: ErrorCode, message: str) -> None:
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


@lru_cache(maxsize=1)
def settings() -> Settings:
    return load_settings()


@lru_cache(maxsize=1)
def policies() -> dict[str, SitePolicy]:
    return load_policies(settings().policy_dir)


def policy(site: str) -> SitePolicy:
    try:
        return policies()[site]
    except KeyError:
        raise ApiError(400, ErrorCode.UNKNOWN_POLICY, f"unknown site policy '{site}'") from None


@app.exception_handler(ApiError)
async def _api_error(_: Request, exc: ApiError) -> JSONResponse:
    body = ErrorResponse(code=exc.code, message=exc.message)
    return JSONResponse(status_code=exc.status, content=body.model_dump(mode="json"))


@app.exception_handler(RequestValidationError)
async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
    first = exc.errors()[0] if exc.errors() else {"msg": "invalid request"}
    body = ErrorResponse(
        code=ErrorCode.INVALID_ANSWER
        if "answer" in str(first.get("loc", ""))
        else ErrorCode.INVALID_STATE,
        message=str(first["msg"]),
    )
    return JSONResponse(status_code=422, content=body.model_dump(mode="json"))


def _analyse(img: Image.Image) -> VisionAnalysis:
    try:
        return vision.analyse(img, settings())
    except vision.ModelUnavailableError as exc:
        raise ApiError(503, ErrorCode.MODEL_UNAVAILABLE, str(exc)) from exc


@app.get("/health", response_model=HealthResponse, responses=ERRORS)
def health() -> HealthResponse:
    return HealthResponse(status="ok", model_loaded=vision.is_loaded(settings()))


@app.get("/version", response_model=VersionResponse, responses=ERRORS)
def version() -> VersionResponse:
    s = settings()
    return VersionResponse(
        model_repo=s.model_repo,
        model_revision=s.model_revision,
        contracts_version=CONTRACTS_VERSION,
        embedding_dim=EMBEDDING_DIM,
        policies=[PolicyVersion(site=k, version=v.version) for k, v in sorted(policies().items())],
    )


@app.post("/predict", response_model=PredictResponse, responses=ERRORS)
def predict(body: PredictRequest) -> PredictResponse:
    site_policy = policy(body.site_policy)
    try:
        img = decode_image(body.image_base64)
    except InvalidImageError as exc:
        raise ApiError(400, ErrorCode.INVALID_IMAGE, str(exc)) from exc
    scan_id, revision = uuid.uuid4(), settings().model_revision
    quality = check_quality(img)
    if not quality.ok:
        return PredictResponse(
            scan_id=scan_id, model_revision=revision, status=ResponseStatus.RETAKE, quality=quality
        )
    analysis = _analyse(img)
    if analysis.is_ood:
        return PredictResponse(
            scan_id=scan_id,
            model_revision=revision,
            status=ResponseStatus.VERDICT,
            quality=quality,
            analysis=analysis,
            verdict=decision.abstain(site_policy, analysis.confidence),
        )
    state = decision.new_state(scan_id, body.site_policy, analysis, settings().state_secret)
    question, verdict = decision.step(state, site_policy)
    return PredictResponse(
        scan_id=scan_id,
        model_revision=revision,
        status=ResponseStatus.QUESTION if question else ResponseStatus.VERDICT,
        quality=quality,
        analysis=analysis,
        next_question=question,
        verdict=verdict,
        state=state if question else None,
    )


@app.post("/answer", response_model=AnswerResponse, responses=ERRORS)
def answer(body: AnswerRequest) -> AnswerResponse:
    site_policy = policy(body.state.site_policy)
    try:
        state = decision.answer(body.state, body.answer, settings().state_secret, site_policy)
    except decision.InvalidStateError as exc:
        raise ApiError(400, ErrorCode.INVALID_STATE, str(exc)) from exc
    question, verdict = decision.step(state, site_policy)
    return AnswerResponse(
        status=ResponseStatus.QUESTION if question else ResponseStatus.VERDICT,
        next_question=question,
        verdict=verdict,
        state=state if question else None,
    )


@app.post("/verify", response_model=VerifyResponse, responses=ERRORS)
def verify(body: VerifyRequest) -> VerifyResponse:
    try:
        img = decode_image(body.image_base64)
    except InvalidImageError as exc:
        raise ApiError(400, ErrorCode.INVALID_IMAGE, str(exc)) from exc
    analysis = _analyse(img)
    return run_verify(body, analysis.embedding, analysis.phash)
