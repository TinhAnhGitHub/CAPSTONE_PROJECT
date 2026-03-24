# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Video Pipeline is a video processing system that extracts metadata, generates embeddings, and builds knowledge graphs from video content. It uses Prefect for orchestration, FastAPI for the submission API, and multiple storage backends (MinIO, PostgreSQL, Qdrant, Elasticsearch, ArangoDB).

## Development Commands

```bash
# Install dependencies (base package)
uv sync

# Install dependencies (with worker extras for ML/inference tasks)
uv sync --extra worker

# Run the API server locally
video-pipeline-api
# Or: uv run video-pipeline-api

# Run linting and formatting
uv run ruff check .
uv run ruff format .

# Run pre-commit on all files
pre-commit run --all-files

# Run tests
uv run pytest
```

## Docker Development

```bash
# Start all services (from docker directory)
cd docker && docker-compose up -d

# Services started:
# - Prefect server (port 4200) - orchestration UI
# - Prefect worker (port 8787) - runs pipeline tasks
# - API (port 8050) - video submission endpoint
# - MinIO (ports 9000/9001) - object storage
# - Qdrant (ports 6333/6334) - vector DB
# - Elasticsearch (port 9200) - text search
# - ArangoDB (port 8529) - graph DB
# - PostgreSQL/Redis - Prefect backend
```

## Architecture

### Three Main Components

1. **API (`src/video_pipeline/api/`)** - FastAPI application that accepts video upload requests and triggers Prefect flows
2. **Flow (`src/video_pipeline/flow/main.py`)** - Main Prefect flow orchestration with DaskTaskRunner for parallel execution
3. **Tasks (`src/video_pipeline/task/`)** - Individual processing tasks following a common pattern

### Task Pattern

All tasks inherit from `BaseTask` and follow a three-phase lifecycle:
- `preprocess()` - Transform raw input
- `execute()` - Core processing logic
- `postprocess()` - Create final artifacts

Task configuration is loaded from YAML (`src/video_pipeline/config/tasks.yaml`), including retries, timeouts, caching, and model-specific kwargs.

### Artifact System

Artifacts (`src/video_pipeline/core/artifact/artifact.py`) are Pydantic models that track data lineage through `lineage_parents`. Each processing stage produces typed artifacts (VideoArtifact, ImageArtifact, etc.) persisted via `ArtifactPersistentVisitor`.

### Storage Clients (`src/video_pipeline/core/client/storage/`)

- **MinIO** - Raw video/frame storage
- **PostgreSQL** - Artifact metadata tracking
- **Qdrant** - Vector embeddings (images, segments, captions)
- **Elasticsearch** - OCR text indexing
- **ArangoDB** - Knowledge graph storage

### Inference Clients (`src/video_pipeline/core/client/inference/`)

External ML services accessed via HTTP/gRPC:
- ASR, OCR, Autoshot, QwenVL embeddings, SPLADE sparse embeddings

## Pipeline Stages

The main flow (`single_video_processing_flow`) processes videos through:

1. **Video Registration** - Extract metadata (fps, duration, format)
2. **Shot Detection (Autoshot)** - Scene boundary detection
3. **Preprocessing** - Split into ASR and image batches
4. **ASR** - Speech-to-text transcription
5. **Audio Segmentation** - Semantic chunking with LLM
6. **Image Extraction** - Extract representative frames
7. **Image Caption/OCR** - VLM captioning and text extraction
8. **Embedding Generation** - QwenVL (images), mmBERT (text)
9. **Qdrant Indexing** - Vector storage for retrieval
10. **Knowledge Graph** - Entity extraction, event linking, community detection
11. **ArangoDB Indexing** - Graph storage

## Configuration

- Environment variables loaded from `.env` at project root
- Settings in `src/video_pipeline/config/settings.py` using pydantic-settings
- Task-specific config in `tasks.yaml`
- Deployment config in `prefect.yaml`

## Key Files

- `src/video_pipeline/flow/main.py` - Main pipeline orchestration
- `src/video_pipeline/task/base/base_task.py` - Task base class and config loading
- `src/video_pipeline/core/artifact/artifact.py` - All artifact type definitions
- `src/video_pipeline/config/tasks.yaml` - Task configurations (timeouts, caching, model URLs)