FROM nvcr.io/nvidia/tritonserver:24.08-py3 AS base

USER root
RUN mkdir -p /tmp && chmod 1777 /tmp

FROM ghcr.io/astral-sh/uv:latest AS uvbin
FROM base AS runtime
COPY --from=uvbin /uv /uvx /usr/local/bin/


RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/tritonserver


COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --no-cache-dir -r pyproject.toml
# 


COPY model_repository /opt/tritonserver/model_repository


ENV PYTHONUNBUFFERED=1 \
    TRANSFORMERS_CACHE=/opt/tritonserver/model_repository/hf_cache \
    HF_HOME=/opt/tritonserver/model_repository/hf_cache

EXPOSE 8000 8001 8002