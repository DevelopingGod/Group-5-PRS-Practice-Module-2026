# AGENTS.md

Instructions for every AI coding assistant working in this repository (Claude Code, Codex, Cursor,
GitHub Copilot, Gemini CLI, Windsurf, ...). Claude Code loads this file through `CLAUDE.md`;
Copilot loads `.github/copilot-instructions.md`, which points here.

## Read first

1. [`context.md`](context.md): the project brief. **Part A restates the submitted proposal
   (`Group_5_Bin_Buddy_PRS_Project_Proposal.pdf`, 27 Sep 2026), which is the single source of truth**
   for what we build, why, how it is measured and what is out of scope. Part B records the build
   decisions. The PDF itself is not in the repository because it contains student IDs.
2. The workstream plan for the area you are touching: [`docs/workstreams/`](docs/workstreams/).
3. [`CONTRIBUTING.md`](CONTRIBUTING.md) before opening a pull request.

If the proposal, `context.md` and the code disagree: the proposal wins on scope, objectives,
metrics and ethics; `contracts/binbuddy_contracts/` wins on data shapes.

## What lives where (monorepo)

| Layer | Path | Language | Deployed to | Owner |
|---|---|---|---|---|
| Frontend | `apps/web/` | Next.js + TypeScript | Vercel | WS4 (WS3 builds the question-flow screens) |
| Backend (model API) | `services/inference/` | Python, FastAPI | Hugging Face Docker Space | WS1 (WS3 `app/decision.py`, WS4 `app/verify.py`) |
| AI / ML (offline training) | `ml/` | Python, PyTorch, scikit-learn (Colab GPU) | Hugging Face Hub model repo (weights) | WS1, WS2 |
| Shared contracts | `contracts/` | Python (Pydantic), generates JSON Schema + TS types | n/a | Everyone |
| Site policies | `policies/` | YAML | bundled into the model API | WS3 |
| Database | `supabase/` | SQL migrations | Supabase | WS4 |

The three deployable layers only talk to each other through the contracts: the browser calls
Next.js route handlers, the route handlers call the FastAPI model API, and the model API loads a
pinned weights revision from the Hub. Never import across layers (for example, `apps/web` never
imports Python, `services/inference` never imports `ml/` training code or PyTorch training loops).

## Hard rules

- **Stay in scope.** Do not add features, screens or models the proposal does not list. Future-scope
  items (bin-mounted scanner, auto-segregation, native apps, multi-item detection) are out.
- **Contracts are shared.** Any change under `contracts/binbuddy_contracts/` needs a
  `CONTRACTS_VERSION` bump, a `contracts/CHANGELOG.md` entry, `pnpm gen`, and both the producing
  and consuming owners' approval. Never hand-write an API type in TypeScript; import from
  `@/lib/contracts`.
- **Never commit** photos, model weights, `.env` files, files over 5 MB, or NUS student IDs. Photos
  live on Google Drive (`BinBuddy_Data/`), weights on the Hugging Face Hub. CI rejects them.
- **Secrets** only in `.env.local`, Vercel settings and Hugging Face Space secrets. The browser never
  sees the HF token or the Supabase service-role key.
- **Privacy:** item-only photos, verification photos deleted after `/verify`, pseudonyms only.
- **Safe advice:** abstain on out-of-distribution items; when still uncertain, recommend general waste.
- **Never invent statistics or sources.** Use only the figures in `context.md` Section 3 and 22.
- **Stay in your workstream.** If a task crosses into another owner's path (see
  `.github/CODEOWNERS`), say so and suggest the human check with that owner.
- Singapore context and British spelling in user-facing copy.

## Commands (run from the repository root)

```bash
uv sync                                   # Python 3.11 env (contracts installed as a workspace package)
pnpm install                              # web dependencies (Node 22+, pnpm 9)

# Everything CI runs; all must pass before a PR can merge
python scripts/check_repo_hygiene.py
uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest
uv run python scripts/validate_data.py
pnpm check:web && pnpm --filter web build
pnpm gen && git diff --exit-code          # regenerated contract artefacts must be committed

# Single test / local servers
uv run pytest services/inference/tests/test_endpoints.py -k predict
cd services/inference && uv run uvicorn app.main:app --reload --port 7860   # API docs at /docs
pnpm --filter web dev                                                       # http://localhost:3000
```

Python is strict: ruff (line length 100), mypy `--strict`, every function typed, contract models
instead of raw dicts. TypeScript is strict: ESLint, Prettier, `tsc`, Vitest.

## Git workflow

- Branch from `main` as `ws1/<short-description>` (or `ws2/`, `ws3/`, `ws4/`, `shared/`).
- Small focused commits; open a PR using the template; never push to `main` directly.
- A PR merges only when the five CI checks are green and the code owner approves. The Copilot AI
  review posts advisory comments only and never blocks a merge.
