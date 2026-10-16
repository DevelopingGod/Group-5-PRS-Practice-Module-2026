# Model API image for the Hugging Face Docker Space. Build context: the repository root.
# HF_HOME is under /tmp: free Spaces have no persistent /data disk, and the Space runs as a
# non-root user who can always write to /tmp. The model is downloaded on the first photo.
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
    HF_HOME=/tmp/huggingface

WORKDIR /app/services/inference
EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
