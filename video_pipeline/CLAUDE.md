# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Development Setup
```bash
# Install with uv (recommended)
uv sync --extra worker

# Or with pip
pip install -e ".[worker]"
```

### Running Services
```bash
# Start all infrastructure (Prefect, Postgres, Redis, MinIO, Qdrant)
cd docker && docker-compose up -d

# Run the API server
video-pipeline-api
# Or directly:
uvicorn video_pipeline.api.app:app --host 0.0.0.0 --port 8050

# Run Prefect worker (after infrastructure is up)
prefect worker start --pool local-pool
```

### Linting and Testing
```bash
# Lint with ruff
ruff check src/
ruff format src/

# Run tests
pytest tests/
pytest tests/test_specific.py -v  # Single file with verbose output
```

## Architecture Overview

This is a Prefect-based video processing pipeline that extracts, analyzes, and indexes video content.

### Core Components

- **Prefect Orchestration**: Flows in `src/video_pipeline/flow/` coordinate task execution using DaskTaskRunner for parallelism
- **Task System**: Each task in `src/video_pipeline/task/` follows the BaseTask template pattern
- **Artifacts**: Pydantic models in `core/artifact/artifact.py` represent pipeline outputs with lineage tracking
- **Storage**: MinIO for objects, PostgreSQL for artifact metadata, Qdrant for vector search

### Task Pattern

All tasks extend `BaseTask[InputT, OutputT]` and implement three methods:
1. `preprocess()` - Validate/transform input, load data, prepare batches
2. `execute()` - Core processing logic (calls external services, models)
3. `postprocess()` - Create artifacts, persist to storage

Task configuration is loaded from `src/video_pipeline/config/tasks.yaml` and converted to Prefect `@task` decorator kwargs via `TaskConfig.from_yaml("task_name")`.

### Configuration

Settings use Pydantic-settings with environment variable overrides. See `src/video_pipeline/config/settings.py`. Environment-specific YAML can be placed in `config/environments/{env}.yaml`.

### Key Infrastructure (docker-compose)

- **Prefect Server**: Workflow orchestration UI at port 4200
- **MinIO**: Object storage (port 9000 API, 9001 console)
- **Qdrant**: Vector database (port 6333)
- **PostgreSQL**: Artifact metadata storage

### Data Flow

```
Video → VideoRegistry → Autoshot → [ASR, ImageExtraction] → [Caption, Embedding, OCR] → Qdrant Indexing
```

Each stage produces typed artifacts that become inputs to downstream tasks, with lineage tracked via `artifact_id` references.