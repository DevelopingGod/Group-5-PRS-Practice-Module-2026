# WS2 Think: classical ML, ensemble and unsupervised learning

**Owner:** Chopra Aryan · **Paths:** `ml/` (shared with WS1), `services/inference/app/vision.py` (co-owner)

**You produce:** the feature-based model, the ensemble and calibrated `confidence`, `is_ood`, embedding-store design, experiment results.
**You consume:** WS1's embeddings and CNN probabilities, `data/labels.csv`.

## Suggested work

**Until 6 Oct**
- [ ] Handcrafted features (colour histograms, HOG, LBP) + SVM and XGBoost baseline on TrashNet/RealWaste
- [ ] Experiment tracking: one results table (model, data version, split, macro-F1, contamination recall) committed as CSV/Markdown
- [ ] Agree the embedding contract with WS1 by **8 Oct**

**7–17 Oct**
- [ ] Soft-voting and stacking ensembles of CNN + feature model; pick one on validation data
- [ ] Calibration (temperature scaling or isotonic); reliability diagram for the report
- [ ] UMAP + k-means/HDBSCAN on embeddings: find confusable groups; produce a "label these next" list for the team
- [ ] OOD detection from embedding distance to training clusters; choose the threshold on validation data
- [ ] Agree the confidence format with WS3 by **14 Oct**

**18–24 Oct**
- [ ] Supabase pgvector: index on `scans.embedding` with WS4; similarity query used by verification
- [ ] Export the ensemble so WS1 can load it in the API

**25–31 Oct**
- [ ] Ensemble vs single-model ablation; abstain rate vs accuracy curve

## Definition of done
- Calibrated confidence values are in [0, 1] and probabilities sum to 1 (the contract rejects anything else)
- Every number in the results table can be traced to a model tag and split
