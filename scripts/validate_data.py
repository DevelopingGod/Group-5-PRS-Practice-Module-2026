"""Validate every site policy in policies/ and the dataset manifest data/labels.csv."""

from __future__ import annotations

import sys
from pathlib import Path

from binbuddy_contracts.constants import DEFAULT_SITE_POLICY
from binbuddy_contracts.dataset import read_labels, validate_labels
from binbuddy_contracts.policy import load_policies

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    try:
        policies = load_policies(ROOT / "policies")
    except Exception as exc:
        print(f"POLICY ERROR: {exc}")
        return 1
    if DEFAULT_SITE_POLICY not in policies:
        print(f"POLICY ERROR: default policy '{DEFAULT_SITE_POLICY}' is missing")
        return 1
    print(f"policies OK: {', '.join(sorted(policies))}")

    try:
        rows = read_labels(ROOT / "data" / "labels.csv")
    except ValueError as exc:
        print(f"LABELS ERROR: {exc}")
        return 1
    errors = validate_labels(rows, policies[DEFAULT_SITE_POLICY])
    for err in errors:
        print(f"LABELS ERROR: {err}")
    if errors:
        return 1
    print(f"labels OK: {len(rows)} photos")
    return 0


if __name__ == "__main__":
    sys.exit(main())
