# Contributing

## Everyday flow

1. Create an issue (template: Task) or pick one assigned to you.
2. Branch from `main`: `ws1/short-description`, `ws2/...`, `ws3/...`, `ws4/...`, or `shared/...`.
3. Commit small, focused changes. Run the checks in the README before pushing.
4. Open a pull request, fill in the template, and request a review from the relevant owner.
5. Merge only when every CI check is green and the owner has approved. Nobody pushes to `main` directly.

## What CI refuses

- Photos, model weights, `.env` files, files over 5 MB, NUS student IDs
- Python that fails ruff, mypy (strict) or pytest
- Web code that fails ESLint, Prettier, TypeScript, Vitest or `next build`
- A policy file that is invalid, leaves a material/condition pair uncovered, or points to an unknown bin
- A `data/labels.csv` row that breaks the manifest rules (see `docs/DATA_COLLECTION.md`)
- Generated files that are out of date with the contracts
- A contract change without a version bump and changelog entry

## Contract changes

Anything in `contracts/binbuddy_contracts/` is shared by several workstreams. To change it:

1. Discuss it first with the workstreams that produce and consume the data (see the table in `contracts/README.md`).
2. Bump `CONTRACTS_VERSION` in `constants.py` and add an entry to `contracts/CHANGELOG.md`.
3. Run `pnpm gen` and commit the regenerated files.
4. Update the Supabase migration if an enum or the embedding size changed (a test checks they match).
5. Get approval from **both** the producing and the consuming owner. GitHub cannot enforce the second approver per path, so this is a team rule; reviewers should check it.

## Python and TypeScript style

- Python 3.11, formatted by ruff, fully typed (mypy strict). Use the contract models; never pass raw dicts across a boundary.
- TypeScript strict. Import API types from `@/lib/contracts`; never hand-write an API shape.
- Secrets live only in `.env.local`, Vercel settings and Hugging Face Space secrets.
