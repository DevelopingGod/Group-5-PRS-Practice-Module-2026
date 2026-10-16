# Bin Buddy

An adaptive, explainable recycling assistant to reduce blue-bin contamination in Singapore.
NUS-ISS Graduate Certificate in Pattern Recognition Systems, Practice Module, Group 5.

You photograph an item → the app checks the photo quality → a vision ensemble predicts the
**material** and the **contamination state** → if it is unsure, it asks **at most three questions**
(only the ones that help) → a site policy maps the result to a bin and explains why, with a heatmap,
the rule and the NEA source → optionally, you scan the bin's QR code and photograph the item there
to earn leaderboard points.

**Source of truth:** the submitted proposal (27 Sep 2026), restated in Part A of
[`context.md`](context.md). Build decisions are in Part B. Workstream plans are in
[`docs/workstreams/`](docs/workstreams/).

## Monorepo layout

Frontend, backend and AI are separate folders that only talk through the shared contracts.

```
apps/web/              FRONTEND   Next.js + TypeScript on Vercel (camera, questions, verdict, QR, leaderboard)
services/inference/    BACKEND    FastAPI model API on Render, free plan (/predict /answer /verify)
ml/                    AI / ML    Offline training in Colab: CNN, features, ensemble, clustering, OOD
contracts/             SHARED     Pydantic models → JSON Schema / OpenAPI → generated TypeScript types
policies/                         Site policies (HDB default, campus example) in YAML
supabase/                         Postgres + pgvector schema, RLS, leaderboard view
data/labels.csv                   Dataset manifest only (photos live on Google Drive)
scripts/                          Code generation and CI validators
```

| Path | Owner |
|---|---|
| `services/inference/` (service, `quality.py`, `vision.py`) and `ml/` | WS1 See: Indish Sankalp Sandeep |
| `ml/` (features, ensemble, clustering, OOD), embedding store | WS2 Think: Chopra Aryan |
| `services/inference/app/decision.py`, `policies/`, question-flow screens | WS3 Ask and Decide: Baral Diwash |
| `apps/web/`, `services/inference/app/verify.py`, `supabase/` | WS4 Verify and Play: Sarade Shreyash Jitendra |

## Getting started (each teammate)

Install [uv](https://docs.astral.sh/uv/) (`pip install uv` works), Node 22 or newer, and pnpm 9
(`npm i -g pnpm@9`). Then:

```bash
git clone https://github.com/DevelopingGod/Group-5-PRS-Practice-Module-2026.git
cd Group-5-PRS-Practice-Module-2026
git config core.autocrlf false   # Windows: keep LF endings (.gitattributes enforces them)
uv sync                          # Python 3.11 env; uv downloads 3.11 if you do not have it
pnpm install
git switch -c ws2/my-first-task  # use your workstream prefix
```

> **Windows and uv:** if `uv sync` fails with *"Missing expected target directory for Python minor
> version link"* (seen when the Windows user folder has a space in it), point uv at the downloaded
> interpreter: `set UV_PYTHON=%APPDATA%\uv\python\cpython-3.11.<x>-windows-x86_64-none\python.exe`
> (PowerShell: `$env:UV_PYTHON = ...`) and run `uv sync` again.

Run the API and the web app locally:

```bash
cd services/inference && uv run uvicorn app.main:app --reload --port 7860   # http://localhost:7860/docs
pnpm --filter web dev                                                       # http://localhost:3000
```

## Using an AI coding assistant

The repository is set up so every assistant starts with the same brief:

| Assistant | Reads automatically |
|---|---|
| Claude Code | `CLAUDE.md` → imports `context.md` and `AGENTS.md` |
| Codex, Cursor, Gemini CLI, Windsurf | `AGENTS.md` (points to `context.md`) |
| GitHub Copilot (chat and PR review) | `.github/copilot-instructions.md` |

Start a session with your workstream's starter brief from `context.md` Section 11, for example
*"I own WS3 Ask and Decide. Help me ..."*. The assistant should stay inside your workstream's paths
and tell you when a change touches another owner's area or the shared contracts.

## Checks (run before pushing; CI runs the same five)

```bash
python scripts/check_repo_hygiene.py                                          # Repo hygiene
uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest   # Python
uv run python scripts/validate_data.py                                        # Policies and dataset labels
pnpm check:web && pnpm --filter web build                                     # Web
pnpm gen && git diff --exit-code                                              # Contract artefacts up to date
```

CI runs on pull requests and on `main`. A red check **never blocks a push** to your own branch; it
only blocks **merging** that PR into `main`. Push early, open a draft PR, and let CI tell you what
to fix.

## AI code review on pull requests

Every PR gets an automatic review from GitHub Copilot, guided by `.github/copilot-instructions.md`
(contract drift, unsafe advice, secrets, scope). It is **advisory**: Copilot only leaves comments,
it never approves or requests changes, so it cannot block a push or a merge. A human code owner
still approves.

Copilot code review needs a Copilot plan. Students get Copilot Pro free through
[GitHub Education](https://education.github.com/pack) (verify with your NUS email). Once the repo
owner has it, the ruleset below requests Copilot on every PR. Teammates with Copilot can also click
**Reviewers → Copilot** on any PR.

## One-time GitHub setup (repo admin)

1. Add the three teammates as collaborators (Settings → Collaborators). When they accept, replace
   the `@TODO-...` handles in `.github/CODEOWNERS` via a PR.
2. Settings → Rules → Rulesets → **New branch ruleset**, target `main`:
   - Require a pull request before merging: **1 approval**, **Require review from Code Owners**,
     **Automatically request Copilot code review** (after GitHub Education is approved)
   - Require status checks to pass: **Repo hygiene**, **Python (lint, types, tests)**,
     **Policies and dataset labels**, **Web (lint, format, types, tests, build)**,
     **Contract artefacts up to date** (they appear after the first CI run)
   - Block force pushes; restrict deletions
3. Vercel: import the repo, set **Root Directory** to `apps/web`, add the variables from
   `apps/web/.env.example`.
4. Hugging Face Hub: model repo `sankalp-indish/bin-buddy-vision` (private, tag `v0.2-local-data`).
   Render: **New → Blueprint** with this repo's `render.yaml` (see `services/inference/README.md`).
   Live API: https://bin-buddy-api.onrender.com (`/docs`, `/health`).
5. Supabase: create the project, run `supabase/migrations/`, enable anonymous sign-ins.

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the everyday branch → PR → review flow.
