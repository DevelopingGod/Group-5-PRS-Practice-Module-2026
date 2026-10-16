# WS4 Verify and Play (Sarade Shreyash Jitendra): handoff from WS1

**In one line:** the API can now run the real model, with a real Grad-CAM heatmap and a 256-value
embedding, and you can test the app against it locally today.

## Run the real model locally
1. Download `BinBuddy_Data/models/v0.2-local-data/` from the team Drive (3 files).
2. From the repository root:
   ```bash
   uv sync
   cd services/inference
   MODEL_DIR=/path/to/v0.2-local-data uv run uvicorn app.main:app --port 7860
   ```
   On Windows PowerShell, set it first: `$env:MODEL_DIR="C:\path\to\v0.2-local-data"`.
3. Open http://localhost:7860/docs and try `POST /predict` with a photo. Without `MODEL_DIR` the API
   keeps the deterministic stub, so nothing breaks for anyone.

**Online API:** Hugging Face no longer offers free Docker Spaces, so the API runs as a free Render
web service (WS1 sends you the URL), with a laptop + Cloudflare tunnel as the demo-day backup. Set
`HF_SPACE_URL` to whichever URL is live; no token is needed to call it. Deploy notes:
[services/inference/README.md](../../../services/inference/README.md).

## What changed for the app
- `analysis.heatmap_png_base64` is now **always** filled: a 224 x 224 PNG of the centre crop with the
  Grad-CAM overlay. Show it on the verdict screen.
- `analysis.embedding` is real: 256 values, unit length ([VisionModel.analyse](https://github.com/DevelopingGod/Group-5-PRS-Practice-Module-2026/blob/52ba19a08dd9f4fd78bf1cb6a732a6437de43c29/services/inference/app/vision.py#L113)).
- The first request after start-up loads the model (a few seconds); after that, about 123 ms of model time per photo.

## For verification (`/verify`)
- Embedding cosine similarity: two photos of the **same** item have a median of **0.90**; photos of
  **different** items stay below **0.55** for 95% of pairs. `VERIFY_MIN_SIMILARITY = 0.85`
  ([constants.py, line 21](https://github.com/DevelopingGod/Group-5-PRS-Practice-Module-2026/blob/52ba19a08dd9f4fd78bf1cb6a732a6437de43c29/contracts/binbuddy_contracts/constants.py#L21)) sits between them.
- Caveat: those same-item pairs were taken seconds apart. A photo at the bin (different light and
  angle) may score lower; check it in the field test before relying on 0.85.

## For the evaluation and the field test
| What | Value (395 Singapore team test photos, never used in training) |
|---|---|
| Model | ConvNeXt-T, fine-tuned (Run 3), Hugging Face tag `v0.2-local-data` |
| Material macro-F1 | **0.799** [95% CI 0.735, 0.841] (proposal target 0.80) |
| Accuracy | **81.8%** (top-2: 92.7%) |
| Costly-error rate (vision alone) | 13.9% of non-recyclable items predicted as a blue-bin material |
| Calibration (ECE) | 0.046 after temperature scaling (T = 0.770) |
| CPU time per photo | ~123 ms (model only) |

- The costly-error rate above is the vision model **alone**; the proposal's 5% target is for the full
  system (questions + policy), which the field test should measure.
- The model handles **one item per photo**; a shelf with many items is out of scope.
- Include the weak classes in the field test: styrofoam, metal, general-waste packaging.

## Where things are
- API code: [services/inference/app/](https://github.com/DevelopingGod/Group-5-PRS-Practice-Module-2026/tree/52ba19a08dd9f4fd78bf1cb6a732a6437de43c29/services/inference/app)
- Contracts (the shapes the app uses): [contracts/binbuddy_contracts/api.py](https://github.com/DevelopingGod/Group-5-PRS-Practice-Module-2026/blob/52ba19a08dd9f4fd78bf1cb6a732a6437de43c29/contracts/binbuddy_contracts/api.py)
- All model results: [ml/EXPERIMENTS.md](https://github.com/DevelopingGod/Group-5-PRS-Practice-Module-2026/blob/52ba19a08dd9f4fd78bf1cb6a732a6437de43c29/ml/EXPERIMENTS.md)
