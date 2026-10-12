"""On pull requests: if the contracts package changed, the version and changelog must change too.

Usage: python scripts/check_contract_change.py <base-ref>   (e.g. origin/main)
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_DIR = "contracts/binbuddy_contracts/"
CONSTANTS = "contracts/binbuddy_contracts/constants.py"
CHANGELOG = "contracts/CHANGELOG.md"
VERSION_RE = re.compile(r'CONTRACTS_VERSION: Final = "([^"]+)"')


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout


def version_at(ref: str) -> str | None:
    try:
        text = git("show", f"{ref}:{CONSTANTS}")
    except subprocess.CalledProcessError:
        return None
    match = VERSION_RE.search(text)
    return match.group(1) if match else None


def main() -> int:
    base = sys.argv[1] if len(sys.argv) > 1 else "origin/main"
    changed = git("diff", "--name-only", f"{base}...HEAD").splitlines()
    if not any(f.startswith(CONTRACT_DIR) for f in changed):
        print("no contract change")
        return 0
    old, new = version_at(base), version_at("HEAD")
    errors = []
    if old is not None and old == new:
        errors.append(f"contracts changed but CONTRACTS_VERSION is still {new}; bump it")
    if CHANGELOG not in changed:
        errors.append(f"contracts changed but {CHANGELOG} has no new entry")
    for e in errors:
        print(f"CONTRACT CHANGE ERROR: {e}")
    if not errors:
        print(f"contract change OK ({old} -> {new})")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
