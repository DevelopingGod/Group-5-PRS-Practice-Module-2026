# WS3 Ask and Decide (Baral Diwash): handoff from WS1

**In one line:** the vision model gives you calibrated **material** probabilities; it never judges
cleanliness, so your questions decide that, and one stub behaviour must change first.

## What the model sends you (`VisionAnalysis`)
- `material_scores`: 10 calibrated probabilities (temperature scaling, ECE 0.046).
- `confidence`: the top material probability.
- `condition_scores`: **always 0.25 each**, by design ([uniform_condition_scores](https://github.com/DevelopingGod/Group-5-PRS-Practice-Module-2026/blob/52ba19a08dd9f4fd78bf1cb6a732a6437de43c29/services/inference/app/vision.py#L150)).

## First thing to fix (safety)
The stub's [_top_condition](https://github.com/DevelopingGod/Group-5-PRS-Practice-Module-2026/blob/52ba19a08dd9f4fd78bf1cb6a732a6437de43c29/services/inference/app/decision.py#L68) picks the first of four
equal scores, which is `clean`. With the stub gate, a confident "plastic" then goes straight to the
blue bin without a single question. As agreed (D-003): **always ask Q2 when the top material is
recyclable** (paper, plastic, glass, metal, beverage carton), whatever the confidence.

## Numbers for tuning the confidence gate
The calibrated model on the 395 team test photos:

| Gate | Answered with no questions | Accuracy of those | Costly errors above the gate |
|---|---|---|---|
| 0.60 | 87% | 88.0% | 8 |
| 0.70 | 80% | 90.2% | 7 |
| **0.80 (current)** | **69%** | **91.6%** | **4** |
| 0.90 | 53% | 93.8% | 2 |
| 0.95 | 36% | 97.9% | 0 |

Tune `CONFIDENCE_THRESHOLD` ([constants.py, line 17](https://github.com/DevelopingGod/Group-5-PRS-Practice-Module-2026/blob/52ba19a08dd9f4fd78bf1cb6a732a6437de43c29/contracts/binbuddy_contracts/constants.py#L17))
on the **validation** rows of `predictions.csv` (team Drive, `BinBuddy_Data/results/v0.2-local-data/`),
not on these test numbers. Changing it is a contract change.

## Mistakes the questions should catch
On team test photos the model confuses: styrofoam → plastic (4 of 8), paper → beverage carton (8),
general-waste packaging → plastic (5), plastic → glass (15). Styrofoam with a recycling symbol is the
classic trap: Q1 "yes" must not send it to the blue bin (the policy rule handles it).

**E-waste safety rule (suggestion):** treat an item as possible e-waste when P(e_waste) > 0.2, even if
it is not the top class. On team test photos this catches 15 of 17 e-waste items, with 6 false flags.

## One gap to raise with the team
The photos were collected **without Q1-Q3 answers**, so `data/labels.csv` cannot yet give you answer
likelihoods. Options: record Q1-Q3 for a subset of items now (about 60 items is enough with
smoothing), or start from hand-set likelihoods and say so in the report.

## Where things are
- Decision code: [services/inference/app/decision.py](https://github.com/DevelopingGod/Group-5-PRS-Practice-Module-2026/blob/52ba19a08dd9f4fd78bf1cb6a732a6437de43c29/services/inference/app/decision.py)
- Per-image predictions with confidence: `predictions.csv` on the team Drive (columns on the WS2 page).
- All model results: [ml/EXPERIMENTS.md](https://github.com/DevelopingGod/Group-5-PRS-Practice-Module-2026/blob/52ba19a08dd9f4fd78bf1cb6a732a6437de43c29/ml/EXPERIMENTS.md)
