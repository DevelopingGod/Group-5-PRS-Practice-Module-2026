# WS3 report to the team: adaptive questions and policy

**From:** Baral Diwash (WS3 Ask and Decide) · **Date:** 19 Oct 2026 · **Branch:** `ws3/adaptive-decision-flow`
([PR #13](https://github.com/DevelopingGod/Group-5-PRS-Practice-Module-2026/pull/13)) · Full design notes: [`docs/workstreams/WS3_ask_decide.md`](../workstreams/WS3_ask_decide.md)

## Status

| WS3 task (proposal Section 11) | Status |
|---|---|
| Naive-Bayes belief updating | **Done**: `_update_belief` in `services/inference/app/decision.py` |
| Entropy-based next-question selection | **Done**: Q2 opens for recyclable top materials, otherwise Q1 (D-004); then the remaining question with the largest expected entropy reduction; at most 3 |
| YAML policy engine (HDB + multi-stream) | **Done**: `hdb_default.yaml`; `campus_example.yaml` completed (paper, plastic, cans/glass bins) |
| Edge-case rules (Section 5.3) | **Done** in both policies and covered by tests |
| Rule-trace explanations | **Done**: plain English, shown in the app's "How Bin Buddy decided" panel |
| Next.js question-flow screens | **Built by WS4** (PR #1); unchanged, they show WS3's questions, confidence and trace |
| Confidence-threshold tuning | **Analysis done**; **recommend keeping 0.80** (revised 22 Oct after D-004; see below). Full analysis: [`WS3_analysis_report.md`](../workstreams/WS3_analysis_report.md) |
| Answer-likelihood estimation | **Partly done**: tables are hand-set; `scripts/estimate_likelihoods.py` replaces them once Q1–Q3 answers are recorded |

All checks pass on the branch: ruff, mypy, all 146 Python tests, policy validation, repo hygiene, `pnpm gen` (no diff), web checks.

## How a decision is made
1. The photo gives the material probabilities (WS1, calibrated). Cleanliness starts unknown (the model does not judge it).
2. Each answer updates the belief: Q1 (recycling symbol) updates the material; Q2 (rinsed) and Q3 (how long) update the condition. Q3 counts for more on drink and wet-food containers, which spoil within hours.
3. **Confidence is measured on the bin**, not the material: P(bin) adds up every (material, condition) pair the policy sends to that bin. So a confident styrofoam needs no questions, and a plastic of unknown cleanliness leans to general waste.
4. Stop when P(bin) ≥ threshold, except: a recyclable is never sent to the blue bin before Q2 (the camera cannot see if it is clean), and not while e-waste is a likely runner-up (P > 0.2 and the e-waste bin is not the most likely, from the WS1 handoff).
5. Still unsure after 3 questions: **general waste plus the NEA Check My Item link**.

Example trace: *Looks like plastic (95% sure) → Probably not clean (99% likely) → Not empty or not rinsed → Container food residue rule, so: General waste.*

## Threshold results with D-004 (WS1 `predictions.csv`, 247 local val photos × 20 simulated users; rerun 22 Oct)

| Gate | Avg questions | Answered with no questions | Accuracy | Costly errors | Fell back to general waste |
|---|---|---|---|---|---|
| 0.60 | 0.79 | 26% | 87.2% | 6.6% | 1.4% |
| 0.70 | 0.85 | 26% | 87.5% | 4.8% | 3.4% |
| 0.75 | 0.90 | 26% | 87.5% | 3.6% | 5.2% |
| **0.80 (current)** | **1.17** | **25%** | **85.6%** | **2.7%** | **11.3%** |
| 0.90 | 1.72 | 21% | 72.2% | 0.5% | 40.4% |
| Static, all 3 questions (0.80) | 3.00 | 0% | 79.0% | 1.2% | 23.1% |

Costly errors: of the items whose true bin is not the blue bin, the share sent to it (the same definition as `scripts/evaluate_baselines.py`). Caveat: the photos have no recorded answers, so the condition and the answers are simulated from WS3's own tables. This compares gate settings; it does not prove the tables.

## What this means for each of you
- **Indish (WS1):** two lines in `app/main.py` changed (`new_state` and `answer` now take the site policy). Nothing else in the API.
- **Aryan (WS2):** the decision flow uses `material_scores` as they arrive. If the ensemble replaces them, keep them calibrated and summing to 1 and nothing changes. `is_ood = true` still abstains before any question.
- **Shreyash (WS4):** no web changes needed. `state.confidence` now means "how sure of the bin", so the meter moves towards the verdict with each answer. In `scripts/evaluate_baselines.py`: new `--system static` (baseline 2), and a fix: full-system contamination recall was always 0 because the vision condition is always "clean"; it now takes the condition from the rule applied.

## Decisions needed from the team
1. **Threshold: keep 0.80?** (Revised 22 Oct; I had proposed 0.75 on 21 Oct, before D-004.) With D-004, 0.75 is 1.9 points more accurate and asks 0.27 fewer questions than 0.80, but makes 0.9 points more costly errors (3.6% against 2.7%) and reaches the 5% limit at just 10% careless answers (0.80 holds to about 18%). Details: [`WS3_analysis_report.md`](../workstreams/WS3_analysis_report.md). Keeping 0.80 needs no contract change; moving to 0.75 later would (version bump, changelog, WS3 + WS4 approve).
2. **Record Q1–Q3 answers** for the photos (columns already in `data/labels.csv`), about 60 items. Then one command counts the real likelihoods.
3. **Report trade-off:** dirty recyclables settle after 1 question (Q2 opens, D-004); a clean one usually needs 2 unless the vision model is very sure of the material (the 89% coke bottle needs 2, a 95% plastic needs 1). Numbers from the Q1-first design (1.5–1.6 questions) are being rerun, so the "≤ 1 question on average" target is still to be re-measured. We choose safety over fewer questions because the model cannot see cleanliness.

## Code review notes (self-review, 19 Oct)
- **Fixed today:** when still unsure after 3 questions, the general-waste card showed the confidence of the best guess (e.g. "General waste, 77% sure" when the blue bin was the 77% guess). It now shows P(general waste), and the trace says "Best guess Blue recycling bin, only 77% sure". Test added.
- **Fixed after the Copilot review (PR #13):** the e-waste rule now exempts only a winning e-waste bin, not e-waste as the top material; counted likelihoods keep every answer at or above 0.01 (before, 200 identical answers gave 0.0099, which `--write` rejected); the sweep's costly-error rate now uses the evaluator's definition, so the table above was regenerated. Tests added for all three.
- **Known limits:** likelihoods are hand-set until answers are recorded; the questions cannot tell which kind of contamination (food, liquid, item inside), only clean or not; the gate sweep uses simulated answers.

## Commands
```bash
uv run pytest services/inference/tests/test_decision.py                 # decision tests
uv run python scripts/simulate_gate.py --predictions <Drive>/predictions.csv --source local
uv run python scripts/estimate_likelihoods.py --write                    # once answers are recorded
uv run python scripts/evaluate_baselines.py --system static --split test --out results/static
```
