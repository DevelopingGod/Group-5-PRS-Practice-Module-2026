# Model API (FastAPI on a Hugging Face Docker Space)

Owner: WS1 (service and vision), with WS3 (`app/decision.py`) and WS4 (`app/verify.py`).

The routes and every request/response model come from `contracts/`. `tests/test_contract.py` fails if the service drifts from `contracts/generated/openapi.json`, so the web app can rely on the generated TypeScript types.

`app/vision.py` runs the exported vision model (ONNX Runtime on CPU, no PyTorch): material
probabilities, a 256-d embedding and a Grad-CAM heatmap computed in NumPy. The export comes from
`ml/notebooks/04_final_model_export.ipynb`. With `MODEL_REVISION=stub` (the default) it returns
deterministic, contract-valid fake predictions, so tests and the other workstreams never need weights.
The model predicts **material only**; condition scores are a uniform prior, because cleanliness
is established by the questions (Q2/Q3).

## Run locally

```bash
uv sync
cd services/inference && uv run uvicorn app.main:app --reload --port 7860
# http://localhost:7860/docs
```

## Environment variables (Space settings)

| Name | Meaning |
|---|---|
| `MODEL_REVISION` | `stub` (default, fake model) or a Hub tag; the current model is `v0.2-local-data` |
| `MODEL_REPO` | Hub repo with the export (default `sankalp-indish/bin-buddy-vision`, private) |
| `HF_TOKEN` | Read token, only needed because the Hub repo is private |
| `MODEL_DIR` | Folder with `config.json`, `model.onnx`, `head.npz`; overrides the Hub (offline, free local setup) |
| `STATE_SECRET` | Secret for signing decision states (set as a Space **secret**) |
| `POLICY_DIR` | Defaults to `/app/policies` in the image |

## Run the real model locally (Docker, no internet needed at runtime)

```bash
docker build -f services/inference/Dockerfile -t binbuddy-api .
docker run -p 7860:7860 -v /path/to/export:/model -e MODEL_DIR=/model -e MODEL_REVISION=local binbuddy-api
```

The model loads on the first photo (`/health` reports `model_loaded`), then stays in memory.

## Deploy

Create a Docker Space (port 7860), then upload the files the image needs (`Dockerfile` at the Space root, plus `pyproject.toml`, `uv.lock`, `contracts/`, `policies/`, `services/inference/`). Automating this with a GitHub Action on merges to `main` is a WS1 task. Free CPU Spaces sleep after about 48 hours without traffic: call `/api/model-health` on the web app before any demo.
