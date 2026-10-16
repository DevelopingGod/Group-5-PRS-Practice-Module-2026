# Copilot instructions (chat and pull-request review)

Read [`AGENTS.md`](../AGENTS.md) and [`context.md`](../context.md) first. They define the project
(Bin Buddy, NUS-ISS PRS Practice Module, Group 5), its scope, the monorepo layout, and the rules
every change must follow.

## When reviewing a pull request

Comment only on things that matter; skip style nits that ruff, ESLint and Prettier already enforce.
Prioritise, in this order:

1. **Correctness bugs** and broken edge cases.
2. **Contract drift.** Any change under `contracts/binbuddy_contracts/` must bump `CONTRACTS_VERSION`
   in `constants.py`, add a `contracts/CHANGELOG.md` entry, and include regenerated files
   (`contracts/generated/`, `apps/web/src/contracts/`). TypeScript must import API shapes from
   `@/lib/contracts`, never redeclare them. Python must pass contract models, not raw dicts.
3. **Model API rules** (`services/inference/`): scores cover every label once and sum to 1;
   embeddings have `EMBEDDING_DIM` values and unit norm; `/predict` returns `retake`, `question`
   or `verdict`; decision states stay HMAC-signed; errors use the `{code, message}` shape.
4. **Safety of advice:** out-of-distribution items must abstain; low confidence after the question
   bank must fall back to general waste; Q1 always opens and at most three questions are asked.
5. **Privacy and secrets:** no secrets in client-side code (`apps/web` files without `"use server"`
   or outside `src/app/api/`), verification photos deleted after `/verify`, pseudonyms only, no NUS
   student IDs, photos or model weights committed.
6. **Scope:** flag features the proposal does not list (see `context.md` Section 7).
7. **Workstream boundaries:** if a PR edits paths owned by another workstream
   (`.github/CODEOWNERS`), say which owner should also review.

Training code in `ml/` may use PyTorch and scikit-learn; runtime code in `services/inference/`
must stay light enough for a free CPU host (Render free plan; the API uses about 240 MB).
