"""Fail if the repository contains things that must never be committed.

- photos, model weights or any file over 5 MB (photos live on Google Drive, weights on the HF Hub)
- .env files with secrets
- NUS student IDs (they are in the proposal PDF, not the repository)
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 5_000_000
BLOCKED_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".heic",
    ".webp",
    ".pt",
    ".pth",
    ".onnx",
    ".safetensors",
    ".ckpt",
    ".pkl",
    ".joblib",
}
ALLOWED_PREFIXES = ("apps/web/public/",)
STUDENT_ID = re.compile(r"\bA\d{7}[A-Z]\b")
TEXT_SUFFIXES = {
    ".py",
    ".ts",
    ".tsx",
    ".js",
    ".mjs",
    ".json",
    ".md",
    ".yaml",
    ".yml",
    ".csv",
    ".sql",
    ".toml",
    ".txt",
}


def tracked_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout
    return [line for line in out.splitlines() if line]


def main() -> int:
    problems: list[str] = []
    for rel in tracked_files():
        path = ROOT / rel
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in BLOCKED_SUFFIXES and not rel.startswith(ALLOWED_PREFIXES):
            problems.append(f"{rel}: photos and model weights belong on Google Drive / the HF Hub")
        if path.stat().st_size > MAX_BYTES:
            problems.append(f"{rel}: file is larger than 5 MB")
        if path.name.startswith(".env") and path.name != ".env.example":
            problems.append(f"{rel}: never commit .env files")
        if suffix in TEXT_SUFFIXES and path.stat().st_size < MAX_BYTES:
            text = path.read_text(encoding="utf-8", errors="ignore")
            if STUDENT_ID.search(text):
                problems.append(f"{rel}: contains what looks like an NUS student ID")
    for p in problems:
        print(f"HYGIENE ERROR: {p}")
    if not problems:
        print("repository hygiene OK")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
