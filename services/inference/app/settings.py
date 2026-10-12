"""Runtime settings from environment variables (set as Hugging Face Space secrets/variables)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class Settings:
    model_repo: str
    model_revision: str
    state_secret: bytes
    policy_dir: Path


def load_settings() -> Settings:
    return Settings(
        model_repo=os.getenv("MODEL_REPO", "binbuddy/vision-stub"),
        model_revision=os.getenv("MODEL_REVISION", "stub"),
        state_secret=os.getenv("STATE_SECRET", "dev-only-secret").encode(),
        policy_dir=Path(os.getenv("POLICY_DIR", str(REPO_ROOT / "policies"))),
    )
