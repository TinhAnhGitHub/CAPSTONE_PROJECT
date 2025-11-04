FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_SYSTEM_PYTHON=1


RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg build-essential curl \
    && rm -rf /var/lib/apt/lists/*

FROM ghcr.io/astral-sh/uv:latest AS uvbin


FROM base AS runtime
COPY --from=uvbin /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN uv venv --system-site-packages .venv
RUN uv sync --frozen --no-dev 



# COPY ./api /app/api
# COPY ./core /app/core
# COPY ./flow /app/flow
# COPY ./task /app/task
# COPY ./main.py /app
# COPY ./prefect_agent/shared /app/prefect_agent/shared

# COPY ./prefect_agent/service_asr /app/prefect_agent/service_asr
# COPY ./prefect_agent/service_autoshot /app/prefect_agent/service_autoshot
# COPY ./prefect_agent/service_llm /app/prefect_agent/service_llm
# COPY ./prefect_agent/service_image_embedding /app/prefect_agent/service_image_embedding
# COPY ./prefect_agent/service_text_embedding /app/prefect_agent/service_text_embedding





EXPOSE 8000

CMD ["uv", "run", "main.py"]
# CMD ["tail", "-f", "/dev/null"]