# WS1 handoff: what the vision model gives the other workstreams

**From:** Indish Sankalp Sandeep (WS1 See) · **Date:** 16 Oct 2026 · **Branch:** `ws1/vision-model`

The vision model is trained, explained, exported and uploaded. This page and the three pages
below tell each workstream what it can use **today** and what to do first.

| Workstream | Start here |
|---|---|
| WS2 Think (Chopra Aryan) | [WS2 handoff](WS2-aryan/README.md) |
| WS3 Ask and Decide (Baral Diwash) | [WS3 handoff](WS3-diwash/README.md) |
| WS4 Verify and Play (Sarade Shreyash Jitendra) | [WS4 handoff](WS4-shreyash/README.md) |

## The model in one table
| What | Value (395 Singapore team test photos, never used in training) |
|---|---|
| Model | ConvNeXt-T, fine-tuned (Run 3), Hugging Face tag `v0.2-local-data` |
| Material macro-F1 | **0.799** [95% CI 0.735, 0.841] (proposal target 0.80) |
| Accuracy | **81.8%** (top-2: 92.7%) |
| Costly-error rate (vision alone) | 13.9% of non-recyclable items predicted as a blue-bin material |
| Calibration (ECE) | 0.046 after temperature scaling (T = 0.770) |
| CPU time per photo | ~123 ms (model only) |

## What the model does and does not do
- It predicts the **material** of one item per photo (the 10 classes in the contract).
- It does **not** judge cleanliness. `condition_scores` are always uniform (0.25 each) by design;
  the questions (Q2, Q3) decide cleanliness. Reason: no public dataset labels cleanliness, and we
  could not collect enough paired dirty/clean photos to learn it reliably.
- Every answer comes with a **Grad-CAM heatmap** (`heatmap_png_base64`, verified to be exact
  Grad-CAM) and a **256-value embedding** (`embedding`, unit length).
- `is_ood` is still always `false`: out-of-distribution detection is WS2's task (see the WS2 page).

**Shared files on the team Google Drive (`BinBuddy_Data/`)**

| File | What it is |
|---|---|
| `models/v0.2-local-data/` (`config.json`, `model.onnx`, `head.npz`) | the exported model, exactly as the API loads it |
| `results/v0.2-local-data/predictions.csv` | for **every** one of the 9,785 images: split, true material, 10 calibrated probabilities, prediction, confidence |
| `results/v0.2-local-data/embeddings.npy` | the 256-value embedding of every image (same row order as `predictions.csv`) |
| `manifests/vision_manifest_v2.csv` | every image: source, material, item group, split (train / val / test) |
| `prepared_v2.tar` | all 9,785 images, resized (unpack once) |
| `results/run1/`, `results/run2/`, `results/run3/` | the model comparisons (per-image test predictions of every model) |

## How it was built (for the report)
- Experiment log, every run with its settings and results: [ml/EXPERIMENTS.md](https://github.com/DevelopingGod/Group-5-PRS-Practice-Module-2026/blob/52ba19a08dd9f4fd78bf1cb6a732a6437de43c29/ml/EXPERIMENTS.md)
- Notebooks (data, comparison of 9 models, explainability, export): [ml/notebooks/](https://github.com/DevelopingGod/Group-5-PRS-Practice-Module-2026/tree/52ba19a08dd9f4fd78bf1cb6a732a6437de43c29/ml/notebooks)
- Model API code: [services/inference/app/vision.py](https://github.com/DevelopingGod/Group-5-PRS-Practice-Module-2026/blob/52ba19a08dd9f4fd78bf1cb6a732a6437de43c29/services/inference/app/vision.py)
- Run the model over any images: [scripts/predict_with_model.py](https://github.com/DevelopingGod/Group-5-PRS-Practice-Module-2026/blob/52ba19a08dd9f4fd78bf1cb6a732a6437de43c29/scripts/predict_with_model.py)
- Model on the Hugging Face Hub (private; ask WS1 for access): https://huggingface.co/sankalp-indish/bin-buddy-vision/tree/v0.2-local-data

All links point to commit `52ba19a`, so they never change. The newest version of any file is on
the branch, for example [ml/EXPERIMENTS.md (latest)](https://github.com/DevelopingGod/Group-5-PRS-Practice-Module-2026/blob/ws1/vision-model/ml/EXPERIMENTS.md).
