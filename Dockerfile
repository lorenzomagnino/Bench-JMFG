# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY README.md .

# Install CPU-only JAX and all project dependencies
RUN pip install --upgrade pip \
    && pip install --no-cache-dir "jax[cpu]" \
    && pip install --no-cache-dir -e ".[dev]"

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy source code
COPY conf/ conf/
COPY envs/ envs/
COPY learner/ learner/
COPY utility/ utility/
COPY main.py .

# Default: run the main training entry point
ENTRYPOINT ["python", "main.py"]
