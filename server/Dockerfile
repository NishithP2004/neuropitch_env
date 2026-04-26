# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Multi-stage build using openenv-base
# This Dockerfile is flexible and works for both:
# - In-repo environments (with local OpenEnv sources)
# - Standalone environments (with openenv from PyPI/Git)
# The build script (openenv build) handles context detection and sets appropriate build args.

ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE} AS builder

WORKDIR /app

# git: VCS deps; gcc/build-essential: Triton JIT-compiles CUDA driver.c at runtime
RUN apt-get update && \
    apt-get install -y --no-install-recommends git gcc build-essential && \
    rm -rf /var/lib/apt/lists/*

# Build argument to control whether we're building standalone or in-repo
ARG BUILD_MODE=in-repo
ARG ENV_NAME=neuropitch_env

WORKDIR /app/env
ENV UV_LINK_MODE=copy

# Ensure uv is available (for local builds where base image lacks it)
RUN if ! command -v uv >/dev/null 2>&1; then \
        curl -LsSf https://astral.sh/uv/install.sh | sh && \
        mv /root/.local/bin/uv /usr/local/bin/uv && \
        mv /root/.local/bin/uvx /usr/local/bin/uvx; \
    fi

# Copy dependency manifests first so dependency layers stay cached when app code changes
COPY pyproject.toml /app/env/pyproject.toml
COPY uv.lock /app/env/uv.lock

# Install third-party dependencies only (fast rebuilds when source changes)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-install-project --no-editable

# Copy environment code after deps to maximize Docker cache hit rate
COPY . /app/env

# Install the local project into the virtualenv
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-editable

# Final runtime stage
FROM ${BASE_IMAGE}

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive

# Runtime packages:
# - curl/zstd/ca-certificates: Ollama bootstrap + health checks
# - gcc/build-essential: Triton JIT-compiles a small C file (driver.c) on first
#   CUDA kernel invocation; without a C compiler Triton raises
#   "Failed to find C compiler" even when only running inference.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl zstd ca-certificates gcc build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Copy the virtual environment from builder
COPY --from=builder /app/env/.venv /app/.venv

# Copy the environment code
COPY --from=builder /app/env /app/env

# Set PATH to use the virtual environment
ENV PATH="/app/.venv/bin:$PATH"

# Set PYTHONPATH so imports work correctly
ENV PYTHONPATH="/app/env:$PYTHONPATH"

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run Ollama and FastAPI together
CMD ["sh", "-c", "mkdir -p /data/ollama/models /data/logs /data/tribe-cache; export OLLAMA_MODELS=/data/ollama/models; ollama serve > /data/logs/ollama-runtime.log 2>&1 & for i in $(seq 1 30); do ollama list >/dev/null 2>&1 && break; sleep 1; done; cd /app/env && exec uvicorn server.app:app --host 0.0.0.0 --port 8000"]
