# WS1 experiment log (vision)

Every data build and training run is recorded here: what was run, on which data and code, what
came out, what we learned, and what we decided next. Entries are appended, never rewritten; a
correction gets its own entry. Numbers come from the saved result files named in each entry.

**Conventions**
- **Data versions** (`DATA_VERSION` in the notebooks): each is frozen once built. Files on Google
  Drive: `prepared_<version>.tar`, `manifests/vision_manifest_<version>.csv` (v1 has no suffix).
- **Runs** (`RUN_ID`): each training run writes to its own `results/<run>/` and `models/<run>/`
  folders (Run 1 used the unsuffixed folders), so no run overwrites another.
- **Code**: the commit is printed by every notebook's setup cell and stored with the results.
- **Scores**: macro-F1 with a 95% bootstrap confidence interval (1,000 resamples). "Singapore
  test" = held-out team photos; "public test" = held-out TrashNet, RealWaste and TACO images.

---

## Run 0 · 13 Oct · pipeline test on synthetic data

- **What:** notebooks 01–04 run end to end on small synthetic images before any GPU time was used.
- **Found and fixed:** DINOv2's saved config asked for 518 px input; pandas 3 drops the grouping
  column in `groupby().apply`; union-find grouping chained different team photos into one item
  (replaced by complete linkage); the ONNX exporter restored training mode, making the export
  check fail (wrapper now set to eval).
- **Verified:** for a linear head on globally pooled features (EfficientNet-B0), Grad-CAM equals
  classifier-weight CAM (Pearson r = 1.0000), so the API can draw heatmaps without PyTorch.

## Data v1 · 14 Oct · first real build (notebook 01, code `38a501c`)

- **Attempt 1 failed:** the TrashNet Hugging Face repo holds the 2,527 photos twice and the
  `datasets` loader mis-inferred labels. **Fix:** download only `dataset-resized.zip`.
- **Prevented:** TACO's own `download.py` stops at the first dead Flickr link; in a sample, 20 of
  30 original links were dead. **Fix:** our downloader (`binbuddy_ml/taco.py`) with a 640 px
  fallback (40/40 in the sample).
- **Result:** 8,751 images = TrashNet 2,527 + RealWaste 4,752 + TACO 1,234 crops (1,500 photos:
  591 original, 909 at 640 px, 0 failed) + 207 team photos (181 items) + 31 web images (train only).
  Split: train 6,049 / val 1,361 / test 1,341; Singapore test = 64 photos of 7 materials.
- **Findings:** strong class imbalance (paper 2,198, plastic 2,164 … beverage carton 28,
  e-waste 23); textile and food waste come almost only from RealWaste; measurable domain gap in
  brightness and sharpness between sources; 46 near-duplicate pairs found and kept together.

## Run 1 · 14 Oct · 9 models, E1 and E2 (notebook 02, data v1, code `38a501c`)

- **Design:** E1 trains on public data only; E2 adds the team training photos and selects on team
  validation photos. Selection rule fixed in advance: best Singapore macro-F1 in E2, then lowest
  costly-error rate if confidence intervals overlap.
- **Results file:** `results/comparison_results.json` (34 rows).

| Model | E1 Singapore | E2 Singapore [95% CI] | E2 costly-error | E2 ECE | E2 public |
|---|---|---|---|---|---|
| CLIP ViT-B/16 probe | 0.488 | **0.658** [0.456, 0.770] | **0.333** | **0.083** | 0.705 |
| ConvNeXt-T | 0.381 | 0.565 [0.396, 0.695] | 0.583 | 0.176 | **0.730** |
| EfficientNet-B0 | 0.263 | 0.531 [0.351, 0.665] | 0.417 | 0.240 | 0.665 |
| Swin-T | 0.311 | 0.494 [0.340, 0.607] | 0.500 | 0.200 | 0.696 |
| DINOv2-S probe | **0.490** | 0.473 [0.328, 0.572] | 0.417 | 0.199 | 0.645 |
| DeiT-S | 0.317 | 0.440 [0.299, 0.542] | 0.500 | 0.136 | 0.728 |
| ResNet-50 | 0.353 | 0.438 [0.302, 0.547] | 0.667 | 0.139 | 0.623 |
| MobileNetV3-L | 0.209 | 0.364 [0.247, 0.466] | 0.667 | 0.116 | 0.635 |
| CLIP zero-shot (no training) | 0.481 | n/a | n/a | n/a | 0.449 (E1) |

- **Learned:**
  1. **Large domain gap.** E1 fine-tuned models reach 0.62–0.74 on public test photos but only
     0.21–0.38 on Singapore photos.
  2. **The team photos matter most.** Adding 91 team photos lifted every model on Singapore
     photos (EfficientNet-B0 0.263 → 0.531; CLIP probe 0.488 → 0.658) without hurting public scores.
  3. **Frozen foundation features generalise best without local data** (E1: probes and zero-shot
     ≈ 0.48–0.49 vs fine-tuned 0.21–0.38).
  4. **The test set is too small to separate models:** all E2 confidence intervals overlap.
     McNemar: CLIP probe vs ConvNeXt-T 9 vs 9 (p = 1.0); vs EfficientNet-B0 15 vs 8 (p = 0.21).
  5. **Weak classes:** glass (CLIP probe 3/10 on Singapore test), general-waste packaging (3/9).
     The CLIP probe's 4 costly errors are borderline packaging (tea sachet, noodle cup, laminated
     pouch, sachet).
  6. **Equal-weight averaging of the top models did not help on Singapore photos** (best pair
     0.643 < 0.658 alone), though it helped on public photos (up to 0.762).
- **Rule outcome:** CLIP ViT-B/16 probe (top macro-F1; lowest costly-error among the tied models).
- **Decision:** collect a final, larger batch of team photos focused on the weak classes (glass,
  general-waste packaging, cartons, paper, e-waste, and the missing textile / food waste /
  styrofoam), freeze it as **data v2**, then concentrate on training for robustness.

## Data v2 · planned · final team photo batch

- Public images and their splits reused unchanged from v1 (`REUSE_PUBLIC_FROM = "v1"`), so runs
  on v1 and v2 differ only in the team photos.
- **Added 15 Oct (before building v2):** the 207 v1 team photos also keep their v1 split,
  matched by original file name. New photos of the same item join that item's group and split;
  if a group mixes splits, test wins. Reason: otherwise some of Run 1's test photos could move
  into v2's training set, and Run 1 could not be compared with later runs on the same photos.
  Notebook 01 stops if any v1 test photo leaves the test set.
- The team's final batch is the last data change: after v2, only training settings change.
- **Built 15 Oct (notebook 01, code `42bdad0`):** 9,785 images = the same 8,513 public images +
  **1,241 team photos** (207 Round 1 + 1,034 Round 2; 905 items, 263 with more than one photo) + 31 web
  images (train only). Round 2 = 1,041 photos from 4 collectors, 7 left out (mixed or unclear items);
  labels proposed from contact sheets and accepted by the owner; name/date watermark strips cropped.
- **Checks (passed):** public images and splits identical to v1; all 207 v1 team photos present, 206 in
  the same split and all 64 v1 test photos still in test (one v1 photo moved split by joining a new
  photo's item group); 0 groups across splits; 0 web images outside training.
- **Team photos per split:** train 599 / val 247 / test 395 (v1 test: 64), so Singapore scores are
  far less noisy than in Run 1.

| Material | train | val | test |
|---|---|---|---|
| plastic | 198 | 83 | 134 |
| glass | 105 | 35 | 70 |
| other_general | 90 | 40 | 51 |
| paper | 56 | 16 | 41 |
| beverage_carton | 42 | 15 | 29 |
| e_waste | 37 | 11 | 17 |
| textile | 29 | 15 | 19 |
| metal | 28 | 21 | 13 |
| food_waste | 12 | 4 | 13 |
| styrofoam | 2 | 7 | 8 |

- **Known limits:** most Round 2 photos show new products in shops, not used items at home (possible
  gap to the demo); styrofoam has only 2 team training photos (groups were split by item), so its
  score will rest mainly on TACO crops; plastic dominates the team photos (33%).

## Run 2 · 15 Oct · effect of the new photos alone (notebook 02, data v2, code `197cf5d`)

- **Design:** Run 1's exact settings on data v2, E2 only, 3 models (CLIP ViT-B/16 probe,
  ConvNeXt-T, EfficientNet-B0). Settings in `experiments.RUNS["run2"]`. T4, 13:35–14:01.
- **Results file:** `results/run2/comparison_results.json`; Singapore test = **395** team photos.

| Model | Singapore macro-F1 [95% CI] | accuracy | top-2 | costly-error | ECE | public macro-F1 | CPU ms |
|---|---|---|---|---|---|---|---|
| ConvNeXt-T | **0.791** [0.736, 0.835] | **0.808** | 0.929 | **0.102** | 0.156 | **0.726** | 122 |
| CLIP ViT-B/16 probe | 0.754 [0.698, 0.800] | 0.749 | 0.904 | 0.176 | **0.060** | 0.688 | 389 |
| EfficientNet-B0 | 0.663 [0.596, 0.711] | 0.716 | 0.823 | 0.167 | 0.132 | 0.644 | **35** |

- **Same 64 v1 test photos, Run 1 → Run 2 (the fair before/after):** CLIP probe 0.658 → 0.747;
  ConvNeXt-T 0.565 → 0.674; EfficientNet-B0 0.531 → 0.500 (64 photos: differences of a few photos).
- **Learned:**
  1. **The new photos were the big lever.** On 395 Singapore photos ConvNeXt-T reaches 80.8% accuracy
     (macro-F1 0.791), from 67% in Run 1; confidence intervals are now ~0.10 wide instead of ~0.30.
  2. **With enough local photos, fine-tuning overtakes the frozen probe.** ConvNeXt-T vs CLIP probe:
     McNemar 63 vs 40, p = 0.030 (the first statistically significant difference in this project);
     vs EfficientNet-B0: 67 vs 31, p < 0.001. In Run 1 (91 local training photos) the probe led.
  3. **Overfitting is visible:** ConvNeXt-T training accuracy 0.995 vs best validation score 0.756;
     EfficientNet-B0 0.99 vs 0.673. This is the gap Run 3 targets.
  4. **Weak classes (ConvNeXt-T):** styrofoam F1 0.46 (only 2 team training photos; 5 of 8 test photos
     predicted plastic), metal 0.56 (8 plastic photos predicted metal: shiny / metallised packaging),
     glass → plastic 13 times. **5 of the 11 costly errors are styrofoam → plastic.**
  5. ConvNeXt-T is over-confident (ECE 0.156): temperature scaling in notebook 04 is needed.
- **Rule outcome:** ConvNeXt-T (highest macro-F1; CIs overlap with the CLIP probe, so tie-break on
  costly-error: 0.102 vs 0.176 → ConvNeXt-T). It is a CNN, so Grad-CAM and notebooks 03–04 work
  without changes. Run 3 decides the final model.

## Run 3 · 15 Oct · anti-overfitting settings (notebook 02, data v2, code `b0dd8a6`)

- **Design:** data v2, E2 only, settings `experiments.PROTOCOL_RUN3`, 6 models: CLIP ViT-B/16
  probe, **CLIP ViT-B/16 partial fine-tune (new)**, ConvNeXt-T, Swin-T, EfficientNet-B0,
  DINOv2-S probe. Same pre-registered selection rule as Run 1.
- **Changes and the Run 1 evidence behind each:**

| Change | Value | Reason (from Run 1) |
|---|---|---|
| Team photos ≥ 25% of each epoch (weighted sampling; probes: sample weights) | `local_share=0.25` | Adding only 91 team photos (~1.5% of training images) raised every model's Singapore score the most; they are the only demo-like photos |
| Dropout before the classifier | `drop_rate=0.3` | Fine-tuned models scored 0.62–0.74 on public test but much lower on Singapore photos: they fit dataset-specific cues |
| Stochastic depth | `drop_path_rate=0.1` | Same reason; standard regulariser for ConvNeXt, Swin and ViTs |
| Stronger augmentation (crop 0.4–1, perspective, ±25°, colour 0.4, blur, random erasing) | `strong_augment=True` | Demo photos are hand-held, tilted, sometimes blurry, partly covered by fingers |
| Selection score = mean of local-val and public-val macro-F1 | `selection="mean_local_public_val"` | Selecting on ~50 local val photos alone was noisy from epoch to epoch |
| CLIP partial fine-tune (last 2 blocks, lr 1e-5) | model `clip_b16_ft` | The frozen CLIP probe won Run 1; this tests whether gently adapting its top layers helps |
| Dropped models: MobileNetV3-L, ResNet-50, DeiT-S, CLIP zero-shot | | Lowest E2 Singapore scores in Run 1; saves ~half the GPU time |

- **Ran:** T4, 14:19–15:32. Results file: `results/run3/comparison_results.json`; 395 Singapore test photos.

| Model | Singapore macro-F1 [95% CI] | accuracy | top-2 | costly-error | ECE | public macro-F1 | CPU ms |
|---|---|---|---|---|---|---|---|
| ConvNeXt-T | **0.799** [0.735, 0.841] | 0.818 | 0.927 | **0.139** | 0.076 | 0.725 | 123 |
| Swin-T | 0.797 [0.738, 0.848] | **0.841** | 0.932 | 0.157 | 0.125 | **0.739** | 159 |
| CLIP ViT-B/16 probe | 0.756 [0.705, 0.805] | 0.770 | 0.922 | 0.148 | **0.031** | 0.688 | 387 |
| DINOv2-S probe | 0.746 [0.688, 0.795] | 0.775 | 0.947 | 0.157 | 0.066 | 0.683 | 161 |
| CLIP ViT-B/16 partial fine-tune | 0.743 [0.704, 0.776] | 0.800 | **0.952** | 0.148 | 0.163 | 0.642 | 442 |
| EfficientNet-B0 | 0.690 [0.626, 0.743] | 0.732 | 0.851 | 0.167 | 0.104 | 0.646 | **37** |

- **Learned:**
  1. **The overfitting gap shrank, the score barely moved.** ConvNeXt-T training accuracy (on
     augmented photos) 0.94 vs 0.995 in Run 2; local-val macro-F1 ~0.80 vs 0.76. On the 395 test photos
     ConvNeXt-T 0.791 → 0.799, McNemar 36 vs 32 (p = 0.72): not a significant change. EfficientNet-B0
     0.663 → 0.690 (39 vs 33, p = 0.56); CLIP probe 0.754 → 0.756 (13 vs 5, p = 0.10).
  2. **Calibration improved a lot:** ConvNeXt-T ECE 0.156 → 0.076 (label smoothing was already on;
     dropout, drop path and harder augmentation made it less over-confident). The confidence gate
     depends on this.
  3. **Styrofoam improved** (ConvNeXt-T F1 0.46 → 0.62) but **costly errors rose** 11 → 15 of 108
     non-recyclable test items (0.102 → 0.139), mostly general-waste packaging → plastic (5) and
     styrofoam → plastic (4); in exchange, recyclables classed as a recyclable material rose 268 → 282 of 287.
  4. **Swin-T caught up with ConvNeXt-T** (0.797; highest accuracy 0.841); the two are not
     significantly different (11 vs 20, p = 0.15).
  5. **The CLIP partial fine-tune under-fitted** (training accuracy only 0.64; styrofoam F1 0.00):
     two unfrozen blocks at lr 1e-5 plus strong augmentation and dropout were too restrictive. Its
     probe sibling remained better calibrated and as accurate.
  6. ConvNeXt-T vs CLIP probe: 56 vs 37 (p = 0.061), weaker than in Run 2 (p = 0.030).
- **Rule outcome:** ConvNeXt-T (highest macro-F1; CIs overlap with Swin-T, CLIP probe, DINOv2 probe and
  CLIP fine-tune → lowest costly-error among them, 0.139). Latency 123 ms on CPU; Grad-CAM direct.

- **New logging in every run from now on:** `run_info.json` (run, data, code commit, GPU,
  settings, image counts, start and finish), the full settings in every result row, training
  accuracy per epoch next to the validation scores (to see overfitting), and the fitted probe
  classifiers saved next to the fine-tuned weights.

## Model choice · 16 Oct · Run 3 ConvNeXt-T goes into the app

- **Decision (owner):** ConvNeXt-T from Run 3 (`models/run3/E2_public_plus_local_convnext_t.pt`),
  chosen over Run 2's ConvNeXt-T: same macro-F1 (0.799 vs 0.791, p = 0.72) but calibration error
  halved (ECE 0.076 vs 0.156) and trained on demo-like augmentation; accepted cost: 4 more costly
  errors of 108, mitigated by always asking Q2 for recyclable materials (D-003).
- **Serving fix found while preparing the export:** ConvNeXt's head is pool → LayerNorm → linear,
  so the classifier-weight heatmap the API used is *not* Grad-CAM for it. The API now computes the
  exact Grad-CAM channel weights through the LayerNorm in NumPy (`head_gradient` in
  `services/inference/app/vision.py`; formula in notebook 03, section 5).
- **Verified on the real Run 3 weights (CPU, before any Colab run):** API probabilities = PyTorch
  (max difference 0.0000); same class as the Colab Run 3 predictions on 40/40 team test photos;
  API heatmap = Grad-CAM by definition at 7×7 (Pearson r = 1.000000, max difference 4.4e-7).
  Notebook 04 repeats this check on the exported files and stops if it fails.

## Explainability · 16 Oct · notebook 03 on the Run 3 ConvNeXt-T (code `e369036`, T4)

40 team test photos; deletion AUC lower = better, insertion AUC higher = better; sanity = Spearman
correlation with the map after randomising the classifier (near 0 = the map depends on the model).

| Method | Deletion AUC | Insertion AUC | Sanity (vs randomised) |
|---|---|---|---|
| **Grad-CAM** | **0.482** | **0.721** | 0.032 (pass) |
| LayerCAM | 0.491 | 0.714 | -0.007 (pass) |
| Grad-CAM++ | 0.546 | 0.697 | -0.152 (pass) |
| Score-CAM | 0.548 | 0.661 | 0.260 (weaker) |
| EigenCAM | 0.566 | 0.617 | **1.000 (fails, as expected: ignores the classifier)** |

- Grad-CAM is the most faithful method on both curves and passes the sanity check, so it is the one
  the app shows. Notebook formula vs Grad-CAM library on ConvNeXt-T: r = 1.0000.
- Note: standard deviations are large (~0.15–0.25), so differences between Grad-CAM, LayerCAM and
  Grad-CAM++ are small; EigenCAM's sanity failure is the clear finding.

## Export · 16 Oct · notebook 04 → Hugging Face `v0.2-local-data` (code `e369036`, CPU)

- Temperature T = 0.770 fitted on 247 team validation photos: ECE 0.079 → 0.046 (T < 1: after Run 3's
  regularisation the model was slightly *under*-confident on team photos, so scores are sharpened).
- Embedding: PCA 768 → 256 keeps 91.4% of the variance; same-item photo pairs median cosine 0.90,
  different items 95th percentile 0.55 (WS4's VERIFY_MIN_SIMILARITY 0.85 sits between them).
- ONNX vs PyTorch max relative difference 7.5e-6; the API's own code on the export vs PyTorch:
  probability difference 0.0000, heatmap r = 1.000.
- Uploaded (private): https://huggingface.co/sankalp-indish/bin-buddy-vision/tree/v0.2-local-data
