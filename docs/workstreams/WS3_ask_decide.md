# WS3 Ask and Decide: adaptive questions and policy

**Owner:** Baral Diwash · **Paths:** `services/inference/app/decision.py`, `services/inference/app/likelihoods.py`, `policies/`, question-flow UI in `apps/web` (with WS4)

**You produce:** the decision flow behind `/predict` and `/answer`, `VerdictResult` with rule trace, site policies, the confidence threshold.
**You consume:** calibrated material scores from WS1/WS2, Q1–Q3 answers recorded in `data/labels.csv`.

## Status

**Until 6 Oct**
- [x] Review `policies/hdb_default.yaml` against NEA's Check My Item (CI checks coverage)
- [x] Walk the stub flow in `app/decision.py`; design the real belief update
- [ ] Make sure everyone records Q1–Q3 answers while collecting photos (not done: the photos have no answers yet, see "Likelihoods")

**7–17 Oct**
- [x] Answer likelihoods P(answer | class): hand-set and provisional; `scripts/estimate_likelihoods.py` replaces them by counting once answers are recorded
- [x] `_update_belief`: naive-Bayes updating
- [x] `_next_question`: Q1 first, then entropy-based selection
- [x] Confidence format: WS1's calibrated material probabilities (`state.confidence` is the verdict bin's probability, see below)

**18–24 Oct**
- [ ] Tune `CONFIDENCE_THRESHOLD` on validation data (contract change): run `scripts/simulate_gate.py` on `predictions.csv`, then decide
- [x] Question-flow screens: built by WS4 (PR #1) against the contract; WS3 supplies the plain-English rule trace and the confidence the meter shows
- [x] `campus_example.yaml` as the multi-stream demonstration

**25–31 Oct**
- [x] Baseline 2, static questionnaire vs adaptive: `--system static` in `scripts/evaluate_baselines.py` (real photos, recorded answers) and static rows in `scripts/simulate_gate.py`

## How the decision works (for the report)

**Belief.** The state holds two distributions: material (calibrated, from the vision model) and
condition (uniform, because the vision model never judges cleanliness; the questions decide it).
Each answer multiplies one distribution by P(answer | class) and renormalises (naive-Bayes evidence
fusion): Q1 updates the material, Q2 and Q3 update the condition.

**Confidence is measured on the bin.** The policy maps every (material, condition) pair to a bin, so
P(bin) = Σ P(material) · P(condition) over the pairs leading to that bin. The verdict names the most
likely bin, with the rule of the most likely pair inside it. Consequences:
- a confident styrofoam or textile needs no questions (every condition gives the same bin);
- an item whose condition is unknown leans to general waste, never to a tie-break of "clean";
- the app's confidence meter (`state.confidence`) moves towards the verdict with each answer.

**Gate.** Settled when P(verdict bin) ≥ `CONFIDENCE_THRESHOLD` (0.80), except:
- **D-003:** a recyclable top material (paper, plastic, glass, metal, carton) is never settled before
  Q2, because the camera cannot see whether it is clean;
- **e-waste rule** (WS1 handoff): while P(e_waste) > 0.2, not settled unless the most likely bin is
  the e-waste one (e-waste can be the top material while the rest adds up to general waste). At a
  0.80 gate this is redundant (no other bin can then reach 80%); it matters if the gate is tuned lower.

**Questions.** Q1 always opens. The follow-up is Q2 when D-003 applies, otherwise whichever of Q2/Q3
has the larger expected entropy reduction of P(bin). At most three questions. Still not settled when
the bank is exhausted: general waste plus the NEA Check My Item link.

**Trade-off to report.** Every recyclable item takes at least two questions (Q1 opens, D-003 adds Q2),
so "≤ 1 question per item on average" is not met for recyclables by design: safety over efficiency,
since the vision model has no condition head.

## Likelihoods (`app/likelihoods.py`)

Hand-set by domain judgement because the photos were collected without Q1–Q3 answers. Each row sums
to 1 and no cell is below 0.01. The questions tell clean from not clean, not which kind of
contamination, so the three contaminated conditions share one row. Q3 is weighted by material:
cartons, cans, bottles and plastic containers mostly hold drinks or wet food that spoil within hours,
so a long wait is strong evidence of residue; for paper and other packaging it says less.

To replace them with counted values once answers are recorded (train and val rows only):

```bash
uv run python scripts/estimate_likelihoods.py            # compare counted vs hand-set
uv run python scripts/estimate_likelihoods.py --write    # writes app/likelihoods_counted.json
```

The API loads `likelihoods_counted.json` at start-up; rows with fewer than 5 answers stay hand-set.

## Tuning and baselines

```bash
# Gate sweep on WS1's predictions (team Drive: BinBuddy_Data/results/v0.2-local-data/)
uv run python scripts/simulate_gate.py --predictions <path>/predictions.csv --out gate.json

# Static questionnaire on real photos (needs recorded answers in data/labels.csv)
uv run python scripts/evaluate_baselines.py --system static --split test --out results/static
```

`simulate_gate.py` simulates the condition and the answers from the likelihood tables, so it compares
gate settings and adaptive vs static; it does not test the tables themselves. Its costly-error rate
uses the same definition as `evaluate_baselines.py`: of the items whose true bin is not a recycling
bin, the share sent to one.

## Definition of done
- [x] Never more than three questions; a tampered state is rejected
- [x] The engine returns the same rule as `SitePolicy.first_match` for every material/condition pair (both policies)
- [x] Uncertain after all questions means general waste plus the NEA link
