# WS1 See: vision and sensing

**Owner:** Indish Sankalp Sandeep · **Paths:** `services/inference/`, `ml/` (shared with WS2)

**You produce:** `VisionAnalysis` (material and condition scores, embedding, heatmap), `QualityReport`, the running model API on the Hugging Face Space.
**You consume:** `data/labels.csv`, calibration and OOD from WS2, decision logic from WS3.

## Suggested work

**Until 6 Oct (first presentation)**
- [ ] Create the Hugging Face model repo and a Docker Space; deploy the stub API as it is today, so WS4 can call a real URL
- [ ] Tune the quality thresholds in `app/quality.py` on 20–30 real phone photos (blurry, dark, good)
- [ ] Baseline CNN: EfficientNet-B0 or MobileNetV3 fine-tuned on TrashNet + RealWaste in Colab; report material accuracy
- [ ] With WS2, agree the embedding: dimension (currently `EMBEDDING_DIM = 256`), layer, L2-normalised. Contract change by **8 Oct**

**7–17 Oct**
- [ ] Multi-task model: shared backbone, material head + contamination head; train on public + local data; push each release to the Hub with a tag (`v0.1-baseline`, `v0.2-local-data`)
- [ ] Replace `app/vision.py` with the real model, loading `MODEL_REVISION`; keep the stub for tests
- [ ] Grad-CAM heatmap returned as `heatmap_png_base64`

**18–24 Oct**
- [ ] Integrate WS2's ensemble and calibration into `/predict`
- [ ] GitHub Action that deploys `services/inference` to the Space on merge to `main`
- [ ] Cold-start handling: fast `/health`, model loaded lazily and cached

**25–31 Oct**
- [ ] Ablation: with vs without local data; results for the report

## Definition of done
- `uv run pytest` passes, including `tests/test_contract.py` (your service still matches the contract)
- Every prediction still validates against `VisionAnalysis` (probabilities sum to 1, embedding length and norm)
- The deployed Space answers `/version` with the tag you expect
