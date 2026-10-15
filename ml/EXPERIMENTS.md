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

## Run 2 · planned · effect of the new photos alone

- **Design:** Run 1's exact settings on data v2, E2 only, 3 models (CLIP ViT-B/16 probe,
  ConvNeXt-T, EfficientNet-B0: the top two of Run 1 plus the fastest). Settings in
  `experiments.RUNS["run2"]`.
- **Why a separate run:** if I changed data and training together, I could not tell which one
  caused an improvement. Run 2 isolates the photos; Run 3 then isolates the training changes.
- **Compared on:** (a) v2's local test set, and (b) the 64 v1 local test photos, scored for every
  run (notebook 02, section 6.8), which is the fair before/after comparison with Run 1.

## Run 3 · planned · anti-overfitting settings

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

- **New logging in every run from now on:** `run_info.json` (run, data, code commit, GPU,
  settings, image counts, start and finish), the full settings in every result row, training
  accuracy per epoch next to the validation scores (to see overfitting), and the fitted probe
  classifiers saved next to the fine-tuned weights.
