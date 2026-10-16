"""Runtime settings from environment variables (set in the Render service settings or locally)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class Settings:
    model_repo: str
    model_revision: str  # "stub" = deterministic fake model (tests, other workstreams)
    state_secret: bytes
    policy_dir: Path
    model_dir: Path | None  # exported model on local disk; skips the Hub download
    hf_token: str | None  # only needed for a private Hub repo


def load_settings() -> Settings:
    model_dir = os.getenv("MODEL_DIR")
    return Settings(
        model_repo=os.getenv("MODEL_REPO", "sankalp-indish/bin-buddy-vision"),
        model_revision=os.getenv("MODEL_REVISION", "stub"),
        state_secret=os.getenv("STATE_SECRET", "dev-only-secret").encode(),
        policy_dir=Path(os.getenv("POLICY_DIR", str(REPO_ROOT / "policies"))),
        model_dir=Path(model_dir) if model_dir else None,
        hf_token=os.getenv("HF_TOKEN") or None,
    )
