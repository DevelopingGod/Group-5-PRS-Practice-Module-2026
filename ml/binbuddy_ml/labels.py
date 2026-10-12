"""Load the validated dataset manifest (data/labels.csv) for training and evaluation."""

from __future__ import annotations

from pathlib import Path

from binbuddy_contracts.constants import DEFAULT_SITE_POLICY
from binbuddy_contracts.dataset import LabelRow, read_labels, validate_labels
from binbuddy_contracts.enums import Split
from binbuddy_contracts.policy import load_policies

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_labels(split: Split | None = None, repo_root: Path = REPO_ROOT) -> list[LabelRow]:
    """Validate the manifest (raises on any error) and return rows, optionally one split."""
    rows = read_labels(repo_root / "data" / "labels.csv")
    hdb = load_policies(repo_root / "policies")[DEFAULT_SITE_POLICY]
    errors = validate_labels(rows, hdb)
    if errors:
        raise ValueError("labels.csv is invalid:\n" + "\n".join(errors))
    parsed = [LabelRow.model_validate(r) for r in rows]
    return [r for r in parsed if split is None or r.split is split]
