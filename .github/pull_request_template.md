## What and why

<!-- One or two sentences. Link the issue: Closes #__ -->

## Workstream

- [ ] WS1 See
- [ ] WS2 Think
- [ ] WS3 Ask and Decide
- [ ] WS4 Verify and Play

## Checklist

- [ ] I ran the checks locally (`uv run ruff check . && uv run mypy && uv run pytest`, and `pnpm check:web` for web changes)
- [ ] New behaviour has a test
- [ ] No photos, model weights, secrets or student IDs in this PR

## Contract change? (anything in `contracts/binbuddy_contracts/`)

- [ ] Not a contract change
- [ ] Contract change: I bumped `CONTRACTS_VERSION`, added a `contracts/CHANGELOG.md` entry, ran `pnpm gen`, and both the producing and consuming workstream owners have approved
