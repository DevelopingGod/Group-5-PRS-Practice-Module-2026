# Model API image (Render free web service; also runs locally). Build context: the repository root.
# HF_HOME is under /tmp: free hosts have no persistent disk, and the container may run as a
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

# Run as an unprivileged user: a compromised process cannot modify the image or the host.
RUN useradd --create-home --uid 1000 app
USER app

WORKDIR /app/services/inference
EXPOSE 7860
# Render sets PORT (default 10000); everywhere else (local, tunnel) the API uses 7860.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
