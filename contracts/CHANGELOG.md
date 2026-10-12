# Contracts changelog

Every change to `contracts/binbuddy_contracts/` needs a version bump in `constants.py` and an entry here (CI enforces both). Say what changed, who produces and who consumes it, and what consumers must update.

## 0.1.0 (2026-09-27)

- Initial contracts, framed on the submitted proposal: label taxonomy (material, condition, channel), question bank Q1–Q3, model API (`/health`, `/version`, `/predict`, `/answer`, `/verify`), site policy schema with first-match semantics and full coverage, dataset manifest (`data/labels.csv`), bin QR payload.
- Provisional values to confirm: `EMBEDDING_DIM` (WS1 + WS2 by 8 Oct), `CONFIDENCE_THRESHOLD` (WS3), verification thresholds and points cap (WS4).
