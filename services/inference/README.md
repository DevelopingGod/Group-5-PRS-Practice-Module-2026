# Model API (FastAPI, Docker)

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

## Deploy (free)

Live: **https://bin-buddy-api.onrender.com** (`/docs`, `/health`). The web app reads this URL from
`HF_SPACE_URL` (the name is historical; any model API URL works) and needs no token.

Hugging Face stopped offering free CPU Docker Spaces in 2026 (PRO only; creating one returns
HTTP 402), so the API runs on:

1. **Render free web service** (online URL for the app). The repository's
   [`render.yaml`](../../render.yaml) defines it: Render dashboard → **New → Blueprint** → this
   repository → enter `HF_TOKEN` (a Hugging Face **read** token). It builds this Dockerfile, listens
   on Render's `PORT`, and checks `/health`. Free services sleep after 15 minutes without traffic
   and take about a minute to wake: open `/health` before a demo.
2. **Backup for demos: your laptop + a Cloudflare Quick Tunnel** (no account needed; the URL
   changes on every start). From the repository root, with the model folder downloaded:

   ```bash
   docker build -f services/inference/Dockerfile -t binbuddy-api .
   docker run -d --name binbuddy-api -p 7860:7860 -e MODEL_DIR=/model -e MODEL_REVISION=v0.2-local-data -v <model folder>:/model:ro binbuddy-api
   cloudflared tunnel --url http://localhost:7860
   ```

   `cloudflared` prints a `https://<random>.trycloudflare.com` URL; use it as `HF_SPACE_URL`.

Measured on 16 Oct: on Render's free plan the first photo after a restart takes about 25 s (model
download), then about 4-5 s per photo, plus about 50 s to wake after 15 minutes idle. Laptop +
tunnel: 0.2-1 s per photo. The API needs about 240 MB of memory with the model loaded. Render
redeploys on every push to the branch in `render.yaml` (`main`).
