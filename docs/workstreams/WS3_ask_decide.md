# WS3 Ask and Decide: adaptive questions and policy

**Owner:** Baral Diwash · **Paths:** `services/inference/app/decision.py`, `policies/`, question-flow UI in `apps/web` (with WS4)

**You produce:** the decision flow behind `/predict` and `/answer`, `VerdictResult` with rule trace, site policies, the confidence threshold.
**You consume:** calibrated scores from WS2, Q1–Q3 answers recorded in `data/labels.csv`.

## Suggested work

**Until 6 Oct**
- [ ] Review `policies/hdb_default.yaml` against NEA's Check My Item; fix any wording or rule order (CI checks coverage)
- [ ] Walk the stub flow in `app/decision.py` (Q1 opens, state is signed); design the real belief update on paper
- [ ] Make sure everyone records Q1–Q3 answers while collecting photos

**7–17 Oct**
- [ ] Estimate answer likelihoods P(answer | material, condition) from `data/labels.csv` (with smoothing)
- [ ] Replace `_update_belief` with naive-Bayes-style updating
- [ ] Replace `_next_question` with entropy-based selection (Q1 always first; proposal 1.2)
- [ ] Agree the confidence format with WS2 by **14 Oct**

**18–24 Oct**
- [ ] Tune `CONFIDENCE_THRESHOLD` on validation data (contract change)
- [ ] Question-flow screens in the web app with WS4, using `QUESTION_BANK` from the generated constants
- [ ] Finish `campus_example.yaml` as the multi-stream demonstration

**25–31 Oct**
- [ ] Baseline: static questionnaire (all three questions every time) vs adaptive; report average questions per item and verdict accuracy

## Definition of done
- Never more than three questions; a tampered state is rejected (tests exist; keep them passing)
- The engine returns the same rule as `SitePolicy.first_match` for every material/condition pair
- Uncertain after all questions means general waste plus the NEA link
