# Model API image for the Hugging Face Docker Space. Build context: the repository root.
#   docker build -f services/inference/Dockerfile -t binbuddy-api .
FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY contracts ./contracts
RUN uv sync --frozen --no-dev

COPY policies ./policies
COPY services/inference ./services/inference

ENV PATH="/app/.venv/bin:$PATH" \
    POLICY_DIR=/app/policies \
    HF_HOME=/data/.huggingface

WORKDIR /app/services/inference
EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
