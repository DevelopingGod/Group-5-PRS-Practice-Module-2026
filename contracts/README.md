# Contracts

The single source of truth for every data shape that crosses a workstream boundary. Python models live here; JSON Schemas, the OpenAPI document and TypeScript types are **generated** from them (`pnpm gen`) and checked for drift in CI.

| Contract | File | Produced by | Consumed by |
|---|---|---|---|
| Label taxonomy (material, condition, channel) | `enums.py` | Team (from the proposal) | Everyone, Supabase enums |
| Question bank Q1–Q3 | `questions.py` | WS3 | WS4 UI, dataset |
| Vision output (`VisionAnalysis`), embedding size | `api.py`, `constants.py` | WS1 + WS2 | WS3, WS4, Supabase |
| Decision flow (`/predict`, `/answer`, `DecisionState`) | `api.py` | WS3 (logic) in WS1's service | WS4 |
| Verification (`/verify`, `BinQrPayload`) | `api.py` | WS4 | WS4 web, bins |
| Site policy schema and first-match semantics | `policy.py` | WS3 | Service, dataset validation |
| Dataset manifest (`data/labels.csv`) | `dataset.py` | Everyone | WS1, WS2, WS3 (likelihoods) |

Taxonomy notes (framed on the proposal):

- **Material**: paper, plastic, glass, metal, beverage_carton, styrofoam, e_waste, textile, food_waste, other_general
- **Condition**: clean, food_residue (includes stained paper), leftover_liquid, item_inside (e.g. banana peel inside a container)
- **Channel**: recycle, general_waste, e_waste, donation, abstain. Site policies map channels to real bins (HDB: one blue bin; campus example: paper/plastic/cans-glass bins).

Values marked PROVISIONAL in `constants.py` must be confirmed by the named workstreams; changing them is a contract change.
