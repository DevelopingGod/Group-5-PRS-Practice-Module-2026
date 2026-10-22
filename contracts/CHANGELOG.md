# Contracts changelog

Every change to `contracts/binbuddy_contracts/` needs a version bump in `constants.py` and an entry here (CI enforces both). Say what changed, who produces and who consumes it, and what consumers must update.

## 0.2.0 (2026-10-22)

- Q2 text changed to "Is it empty, and did you rinse it? (e.g. a milk carton)" so it covers leftover liquid and items inside as well as rinsing. ID (`q2_cleaned`), answer options and likelihoods are unchanged.
- Q1 no longer always opens. Q2 opens when the top material is recyclable, otherwise Q1 (D-004): there is no contamination training data, so the question stands in for a condition head. The next question is chosen by expected entropy reduction among all questions not yet asked; at most three.
- Producer: WS3 (model API, `app/decision.py`). Consumer: WS4 (web shows the question text from the API; only the short label in `scan-flow.tsx` changes). API shapes, IDs and answer options are unchanged.

## 0.1.0 (2026-09-27)

- Initial contracts, framed on the submitted proposal: label taxonomy (material, condition, channel), question bank Q1–Q3, model API (`/health`, `/version`, `/predict`, `/answer`, `/verify`), site policy schema with first-match semantics and full coverage, dataset manifest (`data/labels.csv`), bin QR payload.
- Provisional values to confirm: `EMBEDDING_DIM` (WS1 + WS2 by 8 Oct), `CONFIDENCE_THRESHOLD` (WS3), verification thresholds and points cap (WS4).
