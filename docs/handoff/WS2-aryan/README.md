# WS2 Think (Chopra Aryan): handoff from WS1

**In one line:** you can start the ensemble, calibration, clustering and OOD work right now from
two files on the team Drive: `BinBuddy_Data/results/v0.2-local-data/predictions.csv` and `embeddings.npy`.

## What you get
- **Calibrated CNN probabilities for all 9,785 images**, with the same train / val / test split and
  item groups the CNN used (`predictions.csv`). Columns: `path, source, split, material, group,
  p_paper ... p_other_general, predicted, confidence`.
- **The 256-value embedding** of every image (`embeddings.npy`; row *i* = row *i* of the CSV): PCA of
  ConvNeXt-T's pooled features, L2-normalised. This is exactly `VisionAnalysis.embedding`.
- The images and manifest (`prepared_v2.tar`, `manifests/vision_manifest_v2.csv`) for your
  handcrafted features (HOG, LBP, colour), so both models see identical splits.
- To recompute any of this (for example on new photos):
  [scripts/predict_with_model.py](https://github.com/DevelopingGod/Group-5-PRS-Practice-Module-2026/blob/52ba19a08dd9f4fd78bf1cb6a732a6437de43c29/scripts/predict_with_model.py) (CPU, about 25 images per second).

## The numbers your ensemble has to beat
| What | Value (395 Singapore team test photos, never used in training) |
|---|---|
| Model | ConvNeXt-T, fine-tuned (Run 3), Hugging Face tag `v0.2-local-data` |
| Material macro-F1 | **0.799** [95% CI 0.735, 0.841] (proposal target 0.80) |
| Accuracy | **81.8%** (top-2: 92.7%) |
| Costly-error rate (vision alone) | 13.9% of non-recyclable items predicted as a blue-bin material |
| Calibration (ECE) | 0.046 after temperature scaling (T = 0.770) |
| CPU time per photo | ~123 ms (model only) |

## Start here
1. **Ensemble.** Fit the soft-voting weights or the stacking model on rows with `split == "val"`, and
   report on `split == "test"` **team photos** (`source == "local"`, 395 rows), the same set as the
   CNN's 0.799. Never tune on test rows. Do not fit on `train` rows either: the CNN has seen them, so
   its probabilities there are over-confident.
2. **Calibration.** The CNN probabilities are already temperature-scaled (T = 0.770, ECE 0.046 on team
   validation photos). If the ensemble changes the scores, recalibrate on the team validation
   photos, because `confidence` drives WS3's question gate.
3. **Clustering / confusable groups.** UMAP + k-means/HDBSCAN on `embeddings.npy`. The CNN's main
   confusions on team test photos, to compare with your clusters: plastic → glass (15), paper →
   beverage carton (8), general waste → plastic (5) and → e-waste (5), styrofoam → plastic (4).
4. **OOD (`is_ood`).** Distance from an embedding to the training-class centres (or k-NN distance),
   threshold chosen on validation data. There is no out-of-distribution photo set yet: about 30
   photos of non-waste objects (keys, a hand, a laptop screen) would make one. The result goes into
   `is_ood` here: [vision.py, line 133](https://github.com/DevelopingGod/Group-5-PRS-Practice-Module-2026/blob/52ba19a08dd9f4fd78bf1cb6a732a6437de43c29/services/inference/app/vision.py#L133) (you co-own this file).

## Good to know
- Same-item photo pairs have embedding cosine similarity median 0.90; different items stay below 0.55
  for 95% of pairs (useful for clustering thresholds and for WS4's verification).
- Weak classes: styrofoam (F1 0.62, only 2 team training photos) and metal (0.57).
- Every model's per-image test predictions from Runs 1-3 are in `results/run1/` ... `run3/`
  (`pred_*.csv`), for an "ensemble vs single models" ablation across architectures.
- Background and every number: [ml/EXPERIMENTS.md](https://github.com/DevelopingGod/Group-5-PRS-Practice-Module-2026/blob/52ba19a08dd9f4fd78bf1cb6a732a6437de43c29/ml/EXPERIMENTS.md).
