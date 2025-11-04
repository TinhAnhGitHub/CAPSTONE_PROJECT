# ---- Base Image: Prebuilt PyTorch with CUDA ----
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-venv ffmpeg libgl1-mesa-glx libglib2.0-0 git wget curl \
    && rm -rf /var/lib/apt/lists/*

FROM ghcr.io/astral-sh/uv:latest AS uvbin

FROM base AS runtime
COPY --from=uvbin /uv /uvx /bin/

WORKDIR /app

COPY ./pyproject.toml ./
COPY ./uv.lock ./

RUN uv venv --system-site-packages .venv
RUN uv sync --frozen --no-dev 

# CMD ["tail", "-f", "/dev/null"]
CMD ["uv", "run", "service_llm/main.py"]
