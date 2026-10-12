# Model API (FastAPI on a Hugging Face Docker Space)

Owner: WS1 (service and vision), with WS3 (`app/decision.py`) and WS4 (`app/verify.py`).

The routes and every request/response model come from `contracts/`. `tests/test_contract.py` fails if the service drifts from `contracts/generated/openapi.json`, so the web app can rely on the generated TypeScript types.

`app/vision.py` is a **stub** that returns deterministic, contract-valid predictions so WS3 and WS4 can build from day one. WS1 replaces it with the real model, loaded from a pinned Hugging Face Hub revision (`MODEL_REPO`, `MODEL_REVISION`).

## Run locally

```bash
uv sync
cd services/inference && uv run uvicorn app.main:app --reload --port 7860
# http://localhost:7860/docs
```

## Environment variables (Space settings)

| Name | Meaning |
|---|---|
| `MODEL_REPO` | Hub repo with the fine-tuned weights |
| `MODEL_REVISION` | Tag to load, e.g. `v0.2-local-data` (roll forward/back without code changes) |
| `STATE_SECRET` | Secret for signing decision states (set as a Space **secret**) |
| `POLICY_DIR` | Defaults to `/app/policies` in the image |

## Deploy

Create a Docker Space (port 7860), then upload the files the image needs (`Dockerfile` at the Space root, plus `pyproject.toml`, `uv.lock`, `contracts/`, `policies/`, `services/inference/`). Automating this with a GitHub Action on merges to `main` is a WS1 task. Free CPU Spaces sleep after about 48 hours without traffic: call `/api/model-health` on the web app before any demo.
