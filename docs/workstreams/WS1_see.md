# WS1 See: vision and sensing

**Owner:** Indish Sankalp Sandeep · **Paths:** `services/inference/`, `ml/` (shared with WS2)

**You produce:** `VisionAnalysis` (material scores, uniform condition scores, embedding, heatmap), `QualityReport`, the running model API.
**You consume:** `data/labels.csv`, calibration and OOD from WS2, decision logic from WS3.

**Status 16 Oct: done.** What changed from the original plan, and why:
- The model predicts **material only**; cleanliness comes from the questions (Q2/Q3). No public
  dataset labels cleanliness, and we had too few paired dirty/clean photos to learn it.
- Nine models were compared instead of one baseline; the chosen model is **ConvNeXt-T** (Run 3),
  macro-F1 0.799 on 395 team test photos ([`ml/EXPERIMENTS.md`](../../ml/EXPERIMENTS.md)).
- The API runs on **Render's free plan**, not a Hugging Face Space (PRO only since 2026).
- Handoff to the other workstreams: [`docs/handoff/`](../handoff/README.md).

## Work done
- [x] Quality checker (`app/quality.py`): blur, dark, over-exposed and motion-blur checks with retake
- [x] Embedding agreed with WS2: 256 values, PCA of pooled features, L2-normalised
- [x] Data: 9,785 images (TrashNet, RealWaste, TACO, 1,241 team photos, 31 web images), leak-free item-level splits
- [x] Comparison of 9 models (fine-tuning, linear probes, zero-shot) in 3 runs, logged in `ml/EXPERIMENTS.md`
- [x] Grad-CAM explanations, compared with 4 alternatives and sanity-checked (notebook 03)
- [x] Calibration, 256-d embedding, ONNX export, Hub tag `v0.2-local-data` (notebook 04)
- [x] `app/vision.py` serves the real model (exact Grad-CAM in NumPy, no PyTorch); stub kept for tests
- [x] Deployment: Render (`render.yaml`) + laptop / Cloudflare tunnel backup
- [x] Ablation with vs without local data (E1 vs E2; Run 1 vs Run 2)

## Still open
- [ ] Integrate WS2's ensemble and OOD (`is_ood`) into `/predict` when ready
- [x] After the PR to `main`: `render.yaml` deploys from `main`

## Definition of done
- `uv run pytest` passes, including `tests/test_contract.py`
- Every prediction validates against `VisionAnalysis` (probabilities sum to 1, embedding length and norm)
- The deployed API answers `/version` with the expected tag (`v0.2-local-data`)
