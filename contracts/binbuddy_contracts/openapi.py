"""Reference route table. The real service must expose exactly these routes and models."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI

from binbuddy_contracts.api import (
    AnswerRequest,
    AnswerResponse,
    ErrorResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
    VerifyRequest,
    VerifyResponse,
    VersionResponse,
)
from binbuddy_contracts.constants import CONTRACTS_VERSION

ERRORS: dict[int | str, dict[str, Any]] = {
    400: {"model": ErrorResponse},
    422: {"model": ErrorResponse},
    503: {"model": ErrorResponse},
}

# (method, path) -> (request model or None, response model)
ROUTES: dict[tuple[str, str], tuple[type | None, type]] = {
    ("get", "/health"): (None, HealthResponse),
    ("get", "/version"): (None, VersionResponse),
    ("post", "/predict"): (PredictRequest, PredictResponse),
    ("post", "/answer"): (AnswerRequest, AnswerResponse),
    ("post", "/verify"): (VerifyRequest, VerifyResponse),
}


def build_contract_app() -> FastAPI:
    """A FastAPI app with the contract routes and no logic; used to export openapi.json."""
    app = FastAPI(title="Bin Buddy model API", version=CONTRACTS_VERSION)

    @app.get("/health", response_model=HealthResponse, responses=ERRORS)
    def health() -> HealthResponse:
        raise NotImplementedError

    @app.get("/version", response_model=VersionResponse, responses=ERRORS)
    def version() -> VersionResponse:
        raise NotImplementedError

    @app.post("/predict", response_model=PredictResponse, responses=ERRORS)
    def predict(body: PredictRequest) -> PredictResponse:
        raise NotImplementedError

    @app.post("/answer", response_model=AnswerResponse, responses=ERRORS)
    def answer(body: AnswerRequest) -> AnswerResponse:
        raise NotImplementedError

    @app.post("/verify", response_model=VerifyResponse, responses=ERRORS)
    def verify(body: VerifyRequest) -> VerifyResponse:
        raise NotImplementedError

    return app


def _ref_name(schema: dict[str, Any] | None) -> str | None:
    if not schema:
        return None
    ref = schema.get("$ref")
    return ref.rsplit("/", 1)[-1] if isinstance(ref, str) else None


def route_signature(openapi: dict[str, Any]) -> dict[str, tuple[str | None, str | None]]:
    """(method path) -> (request schema name, 200 response schema name)."""
    sig: dict[str, tuple[str | None, str | None]] = {}
    for path, ops in openapi.get("paths", {}).items():
        for method, op in ops.items():
            body = op.get("requestBody", {}).get("content", {}).get("application/json", {})
            ok = op.get("responses", {}).get("200", {}).get("content", {})
            sig[f"{method} {path}"] = (
                _ref_name(body.get("schema")),
                _ref_name(ok.get("application/json", {}).get("schema")),
            )
    return sig


def contract_differences(contract: dict[str, Any], service: dict[str, Any]) -> list[str]:
    """Differences between the contract OpenAPI and a service's OpenAPI (empty = compatible)."""
    diffs: list[str] = []
    want, got = route_signature(contract), route_signature(service)
    for key in sorted(set(want) | set(got)):
        if key not in got:
            diffs.append(f"missing route: {key}")
        elif key not in want:
            diffs.append(f"route not in contract: {key}")
        elif want[key] != got[key]:
            diffs.append(f"{key}: expected {want[key]}, service has {got[key]}")
    want_schemas = contract.get("components", {}).get("schemas", {})
    got_schemas = service.get("components", {}).get("schemas", {})
    for name, schema in want_schemas.items():
        if name not in got_schemas:
            diffs.append(f"missing schema: {name}")
        elif got_schemas[name] != schema:
            diffs.append(f"schema differs from contract: {name}")
    return diffs
