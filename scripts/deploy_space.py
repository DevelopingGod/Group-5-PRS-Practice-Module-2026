"""Deploy the model API (services/inference) to a Hugging Face Docker Space (WS1).

The Space gets only what the API needs (contracts, policies, services/inference, the lock file)
plus the Dockerfile at its root. Tokens come from environment variables and are never printed:

    HF_TOKEN        your *write* token: creates the Space and uploads the code (not stored)
    SPACE_HF_TOKEN  a *read* token, stored as the Space secret HF_TOKEN so the Space can download
                    the private model; a read token keeps the running Space from changing anything

    uv run python scripts/deploy_space.py --space sankalp-indish/bin-buddy-api --visibility public
    uv run python scripts/deploy_space.py --dry-run --out space_build   # build the folder only

Space settings it writes: variables MODEL_REVISION and MODEL_REPO; secrets HF_TOKEN and
STATE_SECRET (a new random value on every deploy, which only invalidates question flows that are
in progress at that moment).
"""

from __future__ import annotations

import argparse
import os
import secrets
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = ["pyproject.toml", "uv.lock", "contracts", "policies", "services/inference"]
SKIP = shutil.ignore_patterns("tests", "__pycache__", "*.pyc", ".pytest_cache", ".mypy_cache")
SPACE_README = """---
title: Bin Buddy API
emoji: ♻️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
short_description: Bin Buddy model API (NUS-ISS PRS Group 5)
---

# Bin Buddy model API

FastAPI service for Bin Buddy, an explainable recycling assistant for Singapore (NUS-ISS Pattern
Recognition Systems Practice Module, Group 5). Built from the `ws1/vision-model` branch of
https://github.com/DevelopingGod/Group-5-PRS-Practice-Module-2026 at commit `{commit}`;
model `{model_repo}@{revision}`.

Endpoints: `GET /health`, `GET /version`, `POST /predict`, `POST /answer`, `POST /verify`
(interactive docs at `/docs`). Advisory only; it defers to NEA guidance.
"""


def build(out: Path, commit: str, model_repo: str, revision: str) -> None:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for rel in INCLUDE:
        src = ROOT / rel
        if src.is_dir():
            shutil.copytree(src, out / rel, ignore=SKIP)
        else:
            (out / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, out / rel)
    shutil.copy2(ROOT / "services" / "inference" / "Dockerfile", out / "Dockerfile")
    readme = SPACE_README.format(commit=commit, model_repo=model_repo, revision=revision)
    (out / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--space", default="sankalp-indish/bin-buddy-api")
    ap.add_argument("--visibility", choices=["public", "private"])
    ap.add_argument("--model-repo", default="sankalp-indish/bin-buddy-vision")
    ap.add_argument("--model-revision", default="v0.2-local-data")
    ap.add_argument("--dry-run", action="store_true", help="build the folder, upload nothing")
    ap.add_argument("--out", type=Path, default=None, help="where to build (default: temp folder)")
    args = ap.parse_args()

    commit = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    out = args.out or Path(tempfile.mkdtemp(prefix="binbuddy_space_"))
    build(out, commit, args.model_repo, args.model_revision)
    files = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())
    print(f"built {len(files)} files in {out} (code {commit})")
    if args.dry_run:
        return
    if args.visibility is None:
        raise SystemExit("choose --visibility public or --visibility private")
    write_token = os.environ.get("HF_TOKEN")
    read_token = os.environ.get("SPACE_HF_TOKEN")
    if not write_token or not read_token:
        raise SystemExit("set HF_TOKEN (write) and SPACE_HF_TOKEN (read) in this terminal first")

    from huggingface_hub import HfApi

    api = HfApi(token=write_token)
    print("signed in as:", api.whoami()["name"])
    api.create_repo(
        args.space,
        repo_type="space",
        space_sdk="docker",
        private=args.visibility == "private",
        exist_ok=True,
    )
    api.add_space_variable(args.space, "MODEL_REPO", args.model_repo)
    api.add_space_variable(args.space, "MODEL_REVISION", args.model_revision)
    api.add_space_secret(args.space, "HF_TOKEN", read_token)
    api.add_space_secret(args.space, "STATE_SECRET", secrets.token_urlsafe(32))
    api.upload_folder(
        repo_id=args.space,
        repo_type="space",
        folder_path=str(out),
        commit_message=f"Deploy API from {commit} with model {args.model_revision}",
        delete_patterns=["*"],  # the Space mirrors this build exactly
    )
    owner, name = args.space.split("/")
    print("uploaded. The Space now builds the image (about 5-10 minutes).")
    print("Space page: https://huggingface.co/spaces/" + args.space)
    print(f"API base URL: https://{owner}-{name}.hf.space  (docs: /docs, check: /health)")


if __name__ == "__main__":
    main()
