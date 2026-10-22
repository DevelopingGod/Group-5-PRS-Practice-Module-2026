# WS3 analysis report: how well the adaptive questions and the confidence gate work

**Author:** Baral Diwash (WS3 Ask and Decide) · **Date:** 21 Oct 2026 · **Branch:** `ws3/decision-analysis`
**Every number below** is in [`ws3_analysis/results.json`](ws3_analysis/results.json) and
[`ws3_analysis/tables.md`](ws3_analysis/tables.md), produced by `scripts/analyse_decisions.py`
(see "Reproduce" at the end).

> **Note (22 Oct):** all numbers below were rerun for **D-004** (the rinse question, Q2, opens for
> recyclable items; the earlier Q1-first design is kept as the "old design" variant for comparison).
> Same predictions (`v0.2-local-data`), same seeds. Section 2 shows the two designs side by side.

## Summary

0. **D-004 (Q2 opens for recyclables) halves the questions without losing accuracy, but costs some
   safety margin.** At gate 0.75: 0.90 questions instead of 1.54, accuracy 87.5% against 87.0%,
   costly errors 3.6% against 2.9%. At gate 0.80: 1.17 questions instead of 1.63, 85.6% against
   85.2%, costly errors 2.7% against 2.4%. Likely cause (not isolated in the data): with Q2 first, a "yes, rinsed"
   answer can settle a confident recyclable without a second question, so there is less corroborating
   evidence.
1. **Recommended gate: keep 0.80 for now** (PROVISIONAL, no contract change). Under D-004, 0.75 is
   1.9 points more accurate (95% interval +1.0 to +2.8) and asks 0.27 fewer questions, but makes 0.9
   points more costly errors (+0.4 to +1.5) and has little margin: 3.6% at best, 5.0% with just 10%
   random answers. 0.80 keeps costly errors at 2.7% and stays within 5% up to about 18% random
   answers. 0.75 is the better choice only if real recorded answers show careless answering is rare.
   0.70 is no more accurate than 0.75 (+0.0, interval −0.8 to +0.8) and already breaks the target
   at 10% random answers.
2. **Adaptive beats the static questionnaire** (proposal baseline 2). At gate 0.80 it asks 1.17
   questions instead of 3, with 85.6% accuracy against 79.0%; at 0.75, 0.90 questions and 87.5%
   against 81.8%. The held-out test split confirms this (0.75: 87.1% against 80.7%).
3. **D-003 matters.** D-003 is the rule "a recyclable item always gets the rinse question". Without
   it, at a 0.70 gate, 97% of items get no questions and every recyclable of unknown cleanliness goes
   to general waste, so accuracy falls to 65%. Choosing questions by expected entropy reduction beat
   every fixed order: by 1.0 point at 0.75 and 2.0 points at 0.80.
4. **The flow is robust within limits.** At 0.80 the costly-error target holds:
   - with up to about 18% random answers (at 0.75: about 10%);
   - in 100% of worlds where users' answer tables differ moderately from ours (σ = 0.25), and 85%
     where they differ a lot (σ = 0.5) (at 0.75: 95% and 65%);
   - for any share of clean items from 30% to 70% (at 0.75, 70% clean already gives 4.1%).
5. **The confidence Bin Buddy shows can be trusted.** At gate 0.80 the expected calibration error
   (ECE) is 0.022; when it says 80–90% sure it is right 85.6% of the time. **Over-confident vision
   scores are the real danger.** Sharpening them (temperature 0.5) pushes costly errors above 5% at
   every gate we use; at 0.75 even a mild sharpening (0.75) does. WS2's calibration therefore
   protects WS3's safety target.
6. **Styrofoam is the weak spot:** 11.4% of styrofoam runs are costly errors (7 photos), because it
   is misread as plastic and carries a recycling symbol.


## What WS3 built (work log)

| Date (2026) | Work | Commit |
|---|---|---|
| 18 Oct | Naive-Bayes belief update, entropy-based next question, bin-level confidence gate, D-003 and e-waste safety rules | `051d9dc` |
| 18 Oct | Plain-English rule traces ("How Bin Buddy decided"); verdict follows the most likely bin | `9c65624` |
| 18 Oct | Proposal baseline 2: static questionnaire (`--system static`) | `8c67e3a` |
| 19 Oct | Likelihood counting from recorded answers (`estimate_likelihoods.py`), WS3 docs | `aa74bb9` |
| 19 Oct | Gate sweep on the team's own photos (`simulate_gate.py --source local`) | `b4f1cd9` |
| 19 Oct | Fix: fallback verdict shows the confidence of the bin it recommends; team report | `7393a73` |
| 22 Oct | D-004: Q2 (empty and rinsed) opens for recyclable top materials; Q1 can follow Q2; analysis rerun | `ws3/rinse-question-first` |
| 19 Oct | Copilot review fixes (e-waste rule, likelihood floor, costly-error definition); merged as PR #13 | `deb7826` |
| 21 Oct | **This report:** 9 analyses, bootstrap intervals, held-out check, ablation switches, SVG figures, tests | this branch |

Also WS3: `policies/hdb_default.yaml` checked against NEA guidance and `campus_example.yaml`
completed as the multi-stream site.

## Method

- **Photos:** the team's own local photos from WS1's `predictions.csv` (model `v0.2-local-data`,
  ConvNeXt-T, temperature-calibrated): **247 validation photos (179 physical items)** for tuning
  and **395 test photos (287 items)** checked once at the end.
- **Simulated users:** 20 per photo. Each draws a true condition (50% clean, otherwise one of the
  three contaminated conditions) and Q1–Q3 answers from the likelihood tables. Every setting sees
  the same draws.
- **The real flow:** each run goes through the actual `app/decision.py` code that serves `/predict`
  and `/answer`, not a copy of it.
- **Metrics (proposal Section 9):**
  - average questions per item;
  - share of items answered with no questions;
  - verdict accuracy (against the HDB policy's bin for the true material and condition);
  - **costly-error rate**: of the items whose true bin is not the blue bin, the share sent to it
    (the same definition as `scripts/evaluate_baselines.py`);
  - share of items that fell back to general waste.
- **Uncertainty:** 95% bootstrap intervals with 1,000 resamples, **resampling physical items, not
  photos**, because photos of one item are not independent. Comparisons between settings are
  paired: the same resampled items for both.

## 1. Which confidence gate? (threshold trade-off)

![Threshold trade-off](ws3_analysis/fig_threshold_tradeoff.svg)

| Gate (adaptive) | Avg questions | Accuracy | Costly errors | Fallback |
|---|---|---|---|---|
| 0.60 | 0.79 | 87.2% (84.9–89.2) | 6.6% (4.2–9.4) | 1.4% |
| 0.70 | 0.85 | 87.5% (85.3–89.5) | 4.8% (3.1–6.8) | 3.4% |
| 0.75 | 0.90 | 87.5% (85.3–89.5) | 3.6% (2.4–5.2) | 5.2% |
| **0.80 (current)** | **1.17** | **85.6% (83.2–87.8)** | **2.7% (1.9–3.8)** | **11.3%** |
| 0.90 | 1.72 | 72.2% (69.0–76.0) | 0.5% (0.1–1.0) | 40.4% |
| 0.95 | 1.93 | 64.2% (60.2–69.1) | 0.0% | 52.4% |

| Paired comparison | Accuracy | Costly errors | Questions |
|---|---|---|---|
| 0.75 vs 0.80 | +1.9 (+1.0 to +2.8) | +0.9 (+0.4 to +1.5) | −0.27 |
| 0.70 vs 0.75 | +0.0 (−0.8 to +0.8) | +1.2 (+0.4 to +2.2) | −0.04 |
| adaptive 0.80 vs static 0.80 | +6.5 (+5.0 to +7.9) | +1.5 (+0.9 to +2.4) | −1.83 |
| adaptive 0.75 vs static 0.80 | +8.4 (+6.5 to +10.1) | +2.4 (+1.5 to +3.7) | −2.10 |

**Finding.** Raising the gate trades accuracy for safety. Above 0.80, most of the extra caution
becomes "general waste because unsure": the fallback rate reaches 40% at 0.90. Moving from 0.70 to
0.75 costs no measurable accuracy and cuts costly errors by 1.2 points. Moving from 0.75 to 0.80
costs 1.9 points of accuracy for 0.9 points fewer costly errors, and roughly doubles the fallback
rate (5.2% to 11.3%). Because the costly error is the proposal's headline risk, and 0.80 leaves a
wider margin once answers are imperfect (analyses 3 and 4), we keep 0.80 until real answers are recorded.

## 2. Does entropy-based question selection matter? (ablation)

`Gate` has switches used only in analysis: D-003 on or off, how the follow-up after the opening
question is chosen (expected entropy reduction, or a fixed order), and the opening question itself
(`opening="q1"` is the old design, where Q1 always opens).

| Variant | Gate | Avg questions | No questions | Accuracy | Costly errors |
|---|---|---|---|---|---|
| **As designed (D-004: Q2 opens for recyclables)** | 0.75 | 0.90 | 25.9% | 87.5% | 3.6% |
| Old design (Q1 always opens) | 0.75 | 1.54 | 25.9% | 87.0% | 2.9% |
| Fixed order Q2→Q3 (then Q1), D-003 on | 0.75 | 0.91 | 25.9% | 86.5% | 3.9% |
| Fixed order Q3→Q2 (then Q1), D-003 on | 0.75 | 0.91 | 25.9% | 86.5% | 3.9% |
| **As designed** | 0.80 | 1.17 | 25.1% | 85.6% | 2.7% |
| Old design (Q1 always opens) | 0.80 | 1.63 | 25.1% | 85.2% | 2.4% |
| Fixed order Q2→Q3, D-003 on | 0.80 | 1.20 | 25.1% | 83.6% | 2.6% |
| Entropy, **D-003 off** | 0.70 | 0.05 | 96.8% | **65.4%** | 0.0% |
| Entropy, **D-003 off** | 0.75 | 0.82 | 42.5% | 70.5% | 0.6% |
| Entropy, D-003 off | 0.80 | 1.60 | 26.7% | 85.2% | 2.4% |
| Fixed order **Q3→Q2**, D-003 off | 0.80 | 1.92 | 26.7% | **79.0%** | 1.2% |

**Findings.**
- **D-004 trades safety margin for fewer questions.** It saves 0.64 questions per item at 0.75 and
  0.46 at 0.80, with the same or slightly better accuracy, and adds 0.3 to 0.7 points of costly
  errors. Dirty recyclables are now settled after one question (the answer "not empty or not
  rinsed" alone sends them to general waste), which is the case that matters most.
- **D-003 is what keeps recyclables recyclable at lower gates.** The vision model cannot see
  cleanliness, so a plastic bottle starts at P(not clean) = 0.75 and therefore P(general waste) ≈ 0.75.
  Without D-003, a 0.70 gate settles that at once as general waste, with no questions.
  Costly errors look perfect (0%), but almost every clean recyclable is lost.
- **Entropy selection beats every fixed order.** The fixed orders lose 1.0 point at 0.75 and 2.0
  points at 0.80; the worst (Q3 first, D-003 off) loses 6.2 points at 0.80. With Q1 now allowed after
  Q2, the order varies by item. The observed paths at 0.75 were:

  | Path | Share of runs |
  |---|---|
  | Q2 | 64.6% |
  | No questions | 25.9% |
  | Q2 → Q1 → Q3 | 5.3% |
  | Q2 → Q1 | 3.0% |
  | Q1 → Q2 → Q3 | 0.7% |
  | Q1 → Q2 | 0.5% |

## 3. What if users answer carelessly? (answer noise)

Each answer is replaced by a random one with probability 0–30%. The flow still uses its own tables.

![Answer noise](ws3_analysis/fig_answer_noise.svg)

| Random answers | 0.70 costly | 0.75 costly | **0.80 costly** | 0.80 accuracy | Static 0.80 costly |
|---|---|---|---|---|---|
| 0% | 4.8% | 3.6% | **2.7%** | 85.6% | 1.2% |
| 10% | 6.4% ✗ | 5.0% (at the limit) | **3.8%** | 83.2% | 1.9% |
| 20% | 7.9% ✗ | 6.4% ✗ | **5.2%** ✗ | 80.7% | 2.4% |
| 30% | 9.7% ✗ | 8.2% ✗ | **6.6%** ✗ | 78.3% | 3.3% |

**Finding.** Each extra 10% of random answers costs about 2.4 points of accuracy and adds about 1.3
points of costly errors. 0.70 breaks the target at 10% noise, 0.75 reaches it at about 10% and 0.80
holds to about 18%. **Under D-004 this is the main reason to keep 0.80** (before D-004, 0.75 held to
about 15%). The static questionnaire makes fewer costly errors under noise, but at the price of 6
points of accuracy and twice as many fallbacks.

## 4. What if our answer tables are wrong? (misspecification)

Users follow **different tables** from the ones the flow uses: each cell is multiplied by
exp(N(0, σ)) and renormalised. This was run in 20 random worlds per σ.

| σ | Gate | Accuracy, mean (range) | Costly, mean (worst) | Worlds within 5% |
|---|---|---|---|---|
| 0.25 | 0.70 | 86.9% (84.5–89.3) | 5.3% (7.2%) | 45% |
| 0.25 | 0.75 | 86.8% (84.5–89.2) | 4.1% (5.8%) | 95% |
| 0.25 | **0.80** | 84.9% (82.8–87.4) | 3.0% (4.4%) | **100%** |
| 0.5 | 0.70 | 85.6% (80.4–89.8) | 5.6% (10.5%) | 50% |
| 0.5 | 0.75 | 85.5% (80.2–89.8) | 4.5% (8.8%) | 65% |
| 0.5 | **0.80** | 83.5% (78.8–88.0) | 3.4% (7.0%) | **85%** |

The other unknown is the share of clean items people bring:

| Clean items | 0.80 accuracy | 0.80 costly errors | 0.75 accuracy | 0.75 costly errors |
|---|---|---|---|---|
| 30% | 90.3% | 2.7% | 91.2% | 3.6% |
| 50% | 85.6% | 2.7% | 87.5% | 3.6% |
| 70% | 81.2% | 2.4% | 84.4% | 4.1% |

**Finding.** With moderately wrong tables (σ = 0.25, cells off by about ±25%), 0.80 stays within
the target in every world and 0.75 in 95%. With badly wrong tables (σ = 0.5) it is 85% against 65%.
Costly errors at 0.80 barely move with the share of clean items; at 0.75 they rise to 4.1% when 70%
of items are clean. The hand-set tables are therefore not a single point of failure, but recording
real answers (team decision 2) is still the way to remove this uncertainty.

## 5. Calibration: can users trust the confidence shown?

**Inputs (WS1/WS2):**
- The vision material scores on these photos are close to calibrated: 81.4% top-1 accuracy at 82.6%
  mean confidence, ECE 0.044.

**Outputs (WS3), adaptive at 0.80 (ECE 0.022):**

| Confidence shown on the verdict | Runs | Mean shown | Actually right |
|---|---|---|---|
| 70–80% | 43 | 73.8% | 74.4% |
| 80–90% | 1,973 | 83.3% | 85.6% |
| 90–95% | 410 | 93.1% | 92.4% |
| 95–100% | 1,997 | 97.8% | 97.1% |

**What if the vision scores were over- or under-confident?** Each score p was raised to the power
1/T and renormalised:

![Calibration](ws3_analysis/fig_calibration.svg)

| Temperature | Gate | Accuracy | Costly errors | Questions | Fallback |
|---|---|---|---|---|---|
| 0.5 (over-confident) | 0.80 | 86.9% | **5.6%** ✗ | 0.93 | 5.6% |
| 0.75 | 0.75 | 87.4% | **5.2%** ✗ | 0.84 | 3.2% |
| 1.0 (as delivered) | 0.75 | 87.5% | 3.6% | 0.90 | 5.2% |
| 1.0 (as delivered) | 0.80 | 85.6% | 2.7% | 1.17 | 11.3% |
| 2.0 (under-confident) | 0.80 | 69.0% | 0.3% | 1.81 | 34.4% |

**Findings.**
- The confidence a verdict shows is honest: shown and actual agree within about 4 points in every
  well-populated band at 0.80. Part of this agreement holds by construction, because the simulated
  answers follow the flow's own tables.
- **Over-confidence is the dangerous direction:** the gate passes items it should have asked about,
  and costly errors break the target. Under D-004 the margin is thinner: at 0.75 a mild sharpening
  (T = 0.75) already breaks it, at 0.80 it takes T = 0.5. Under-confidence is only inefficient: more
  questions, more fallbacks.

## 6. Where do the questions and errors go? (per material, 0.75)

| Material | Photos | Avg questions | Accuracy | Costly errors |
|---|---|---|---|---|
| Plastic | 83 | 1.14 | 80.5% | 2.9% |
| Glass | 35 | 1.02 | 89.4% | 3.7% |
| Metal | 21 | 1.25 | 79.5% | 5.9% |
| Paper | 16 | 1.09 | 84.4% | 5.5% |
| Beverage carton | 15 | 1.09 | 89.0% | 3.8% |
| Styrofoam | 7 | 1.17 | 88.6% | **11.4%** |
| Other general | 40 | 0.56 | 95.5% | 4.5% |
| E-waste, textile, food waste | 30 | 0.00 | 100% | 0.0% |

**Findings.**
- The questions go to recyclables, about one each now (about two before D-004). Items that go to the
  same bin whatever their condition get **none**, which keeps the average near 0.9.
- Styrofoam's costly errors come from photos the model reads as plastic: Q1 cannot fix this, because
  styrofoam often carries a recycling symbol (resin code 6). The fix belongs to the vision model (more
  styrofoam photos), not the questions.
- "Other general" items rose from 2.0% to 4.5% costly errors under D-004: probably because items the
  model reads as recyclable now get one question, and a "yes" can settle them as blue bin (not
  isolated in the data). This is worth watching.
- Samples are small (7 styrofoam photos), so treat these figures as warnings, not estimates.

## 7. Held-out check (test split, checked once)

| System | Gate | Avg questions | Accuracy | Costly errors | Fallback |
|---|---|---|---|---|---|
| Adaptive | 0.75 | 0.93 (0.85–1.02) | 87.1% (85.4–88.6) | 3.1% (2.4–4.0) | 5.3% |
| Static | 0.75 | 3.00 | 80.7% (78.9–82.4) | 1.6% (1.2–2.1) | 19.2% |
| **Adaptive** | **0.80** | **1.21 (1.12–1.31)** | **85.2% (83.4–87.0)** | **2.4% (1.8–3.0)** | **12.5%** |

**Finding.** The test split agrees with validation: at 0.80, 85.2% against 85.6% accuracy and 2.4%
against 2.7% costly errors; at 0.75, 87.1% against 87.5% and 3.1% against 3.6%. So the gate was not
overfitted to the validation photos.

## 8. A second site policy (multi-stream campus bins)

| System | Gate | Avg questions | Accuracy | Costly errors | Fallback |
|---|---|---|---|---|---|
| Adaptive | 0.80 | 1.28 | 79.8% | 1.7% | 18.0% |
| Adaptive | 0.75 | 1.03 | 82.1% | 2.7% | 11.1% |
| Static | 0.80 | 3.00 | 75.1% | 0.7% | 27.3% |

**Finding.** The same flow works unchanged on separate paper, plastic and cans/glass bins.
Accuracy is about 4–6 points lower than HDB because material mix-ups now change the bin (paper vs
plastic), whereas HDB's single blue bin absorbs them.

## 9. Should the flow assume more items are clean? (condition prior)

The flow starts every item at P(clean) = 0.25 (uniform over the four conditions). Would 0.50 be
better?

| Prior P(clean) | Gate | Accuracy | Costly errors | Fallback |
|---|---|---|---|---|
| **0.25 (current)** | 0.75 | 87.5% | **3.6%** | **5.2%** |
| 0.50 | 0.75 | 87.3% | 5.4% | 15.9% |
| **0.25 (current)** | 0.80 | 85.6% | **2.7%** | **11.3%** |
| 0.50 | 0.80 | 87.7% | 4.3% | 18.0% |

(50% clean items; the pattern is the same at 30% and 70%.)

**Finding.** Keep the uniform prior. A higher prior makes 1.6 to 1.8 points more costly errors and
up to three times as many fallbacks, and at 0.75 it is not even more accurate. The reason is that "Not
sure" settles as general waste under the current prior, but stays undecided under the higher one.
A data-driven prior would need cleanliness labels, which the dataset does not have.

## Decisions for the team

1. **Keep `CONFIDENCE_THRESHOLD` at 0.80** (no contract change for the gate). This reverses my 21 Oct
   proposal of 0.75, which was made before D-004: under D-004, 0.75 has too little margin (5.0% costly
   errors at 10% random answers). Revisit once real answers are recorded; 0.75 then saves another 0.27
   questions per item.
2. **Record Q1–Q3 answers** when photographing items (unchanged). Analyses 3 and 4 show the flow
   tolerates moderate errors in the tables, but real answers remove the main limitation.
3. **WS1/WS2:** keep the material scores calibrated, never sharpened (analysis 5), and add styrofoam
   photos (analysis 6).

## Limitations

- **Simulated answers and conditions:** see the summary. This tests the decision logic and the gate,
  not the real accuracy of people's answers.
- **Same tables for users and flow:** in analyses 1, 2 and 5–9 the simulated users follow the
  flow's own tables. Analyses 3 and 4 relax this.
- **Small classes:** styrofoam (7 validation photos), food waste (4) and e-waste (11) are too small
  for firm per-material conclusions.
- **Correct out-of-distribution handling assumed:** the analysis uses material scores only, with no
  out-of-distribution abstention.

## Reproduce

```bash
# WS1 predictions: team Drive BinBuddy_Data/results/v0.2-local-data/predictions.csv (~3 minutes)
uv run python scripts/analyse_decisions.py --predictions <path>/predictions.csv \
    --out docs/workstreams/ws3_analysis
uv run pytest scripts/test_analyse_decisions.py services/inference/tests/test_decision.py
```

Defaults:
- `--source local`
- `--thresholds 0.6 0.7 0.75 0.8 0.9 0.95`
- `--repeats 20`
- `--clean-share 0.5`
- `--worlds 20`
- `--resamples 1000`
- `--seed 0`
