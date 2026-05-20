# Video Pipeline

Prefect-based ingestion service that turns source videos into searchable multimodal artifacts for VideoDeepSearch. The pipeline extracts video metadata, detects shots, transcribes audio, captions frames and segments, generates embeddings, indexes OCR text, and builds a knowledge graph for graph-based retrieval.

The service is implemented as a Python package under `src/video_pipeline` and exposes a FastAPI submission API plus a Prefect deployment named `local-deployment`.

## Contents

- [Architecture](#architecture)
- [Pipeline Flow](#pipeline-flow)
- [Storage and Indexes](#storage-and-indexes)
- [API](#api)
- [Configuration](#configuration)
- [Local Setup](#local-setup)
- [Docker Setup](#docker-setup)
- [Development](#development)
- [Project Layout](#project-layout)

## Architecture

```text
Client
  |
  | POST /uploads/
  v
FastAPI API :8050
  |
  | run_deployment("local-deployment")
  v
Prefect Server :4200  <---- Prefect Worker
  |                         |
  |                         v
  |                  single_video_processing_flow
  |
  v
Storage and retrieval backends
  MinIO + PostgreSQL + Qdrant + Elasticsearch + ArangoDB
```

Core runtime components:

| Component | Purpose |
|-----------|---------|
| FastAPI API | Accepts upload, retrieval, deletion, and health requests. |
| Prefect 3 | Registers and schedules the `single_video_processing_flow` deployment. |
| Dask task runner | Runs pipeline branches and chunked tasks in parallel inside the worker. |
| External inference services | ASR, shot detection, OCR, QwenVL embeddings, mmBERT embeddings, SPLADE sparse embeddings, and LLM calls. |
| Storage clients | Persist files, lineage, vectors, OCR documents, and graph data. |

## Pipeline Flow

The deployment entrypoint is:

```text
src/video_pipeline/flow/main.py:single_video_processing_flow
```

High-level stages:

1. Register the video and persist a `VideoArtifact`.
2. Run Autoshot scene boundary detection.
3. Preprocess the video into ASR batches and image batches.
4. Run the audio branch: ASR transcription, LLM-based audio segmentation, segment captioning, segment multimodal embeddings, audio transcript text embeddings, segment caption text embeddings, and Qdrant indexing.
5. Run the image branch: representative frame extraction, combined image caption and OCR extraction, image multimodal embeddings, image caption text embeddings, Qdrant indexing, and Elasticsearch OCR indexing.
6. Build the knowledge graph: KG extraction from segment captions, entity resolution, event and micro-event linking, KG finalization, and ArangoDB indexing.
7. Emit Prefect markdown summaries and optional HTTP progress updates.

Task retry, timeout, cache, batch, model, and concurrency settings are defined in `src/video_pipeline/config/tasks.yaml`.

## Storage and Indexes

| Backend | Default port | Used for |
|---------|--------------|----------|
| PostgreSQL | 5432 | Artifact metadata, lineage, and Prefect server database. |
| MinIO | 9000 / 9001 | Raw videos, extracted frames, and Prefect result storage. |
| Qdrant | 6333 / 6334 | Dense and sparse vector indexes for images, captions, segments, and audio transcripts. |
| Elasticsearch | 9200 | OCR text search. |
| ArangoDB | 8529 | Video knowledge graph. |
| Redis | 6379 | Prefect messaging broker and cache. |

Main artifact models live in `src/video_pipeline/core/artifact/artifact.py`:

| Artifact | Description |
|----------|-------------|
| `VideoArtifact` | Registered source video metadata and MinIO path. |
| `AutoshotArtifact` | Shot or scene boundary output. |
| `ASRArtifact` | Audio transcript chunks. |
| `AudioSegmentArtifact` | Semantically grouped transcript segments with timestamps and frame ranges. |
| `ImageArtifact` | Extracted frame metadata and storage path. |
| `ImageCaptionArtifact` / `ImageOCRArtifact` | VLM caption and OCR results for frames. |
| `ImageEmbeddingArtifact` / `SegmentEmbeddingArtifact` | QwenVL multimodal embeddings. |
| `TextCaptionEmbeddingArtifact` / `TextCapSegmentEmbedArtifact` / `AudioTranscriptEmbedArtifact` | mmBERT text embeddings. |
| `KGExtractionArtifact` / `KGGraphArtifact` | Intermediate and final knowledge graph outputs. |
| `ArangoIndexingArtifact` | ArangoDB indexing result counts. |

## API

The API listens on port `8050` when started through Docker or the `video-pipeline-api` console script. Swagger docs are available at:

```text
http://localhost:8050/docs
```

### Submit Videos

```http
POST /uploads/
```

```json
{
  "user_id": "user-abc",
  "videos": [
    {
      "video_id": "vid-001",
      "video_url": "s3://bucket/path/video.mp4"
    }
  ]
}
```

Returns `202 Accepted` after scheduling one Prefect flow run per video.

### Retrieve Video Data

```http
GET /videos/{video_id}/full
```

Query parameters:

| Parameter | Description |
|-----------|-------------|
| `sources` | Optional comma-separated subset: `postgres,arango,qdrant,elasticsearch`. |
| `include_vectors` | Include raw Qdrant vectors when `true`. Defaults to `false`. |

### Delete Video Data

```http
DELETE /videos/{video_id}
```

Deletes related data from PostgreSQL, MinIO, Qdrant, ArangoDB, and Elasticsearch.

### Health Checks

```http
GET /api/health
GET /api/health/prefect
```

## Configuration

Application settings are loaded with Pydantic settings from environment variables and, for local Python runs, from `video_pipeline/.env` when that file exists. Docker services also load `video_pipeline/docker/.env` through Compose.

Important environment prefixes:

| Prefix | Examples | Purpose |
|--------|----------|---------|
| `MINIO_` | `MINIO_ENDPOINT`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`, `MINIO_BUCKET_NAME` | Object storage. |
| `POSTGRES_` | `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_DATABASE`, `POSTGRES_USER`, `POSTGRES_PASSWORD` | Artifact and lineage database. |
| `QDRANT_` | `QDRANT_HOST`, `QDRANT_PORT`, `QDRANT_API_KEY`, `QDRANT_COLLECTION_BASE` | Vector database. |
| `ELASTICSEARCH_` | `ELASTICSEARCH_HOST`, `ELASTICSEARCH_PORT`, `ELASTICSEARCH_INDEX_NAME` | OCR indexing. |
| `ARANGO_` | `ARANGO_HOST`, `ARANGO_DATABASE`, `ARANGO_GRAPH_NAME`, `ARANGO_USERNAME`, `ARANGO_PASSWORD` | Knowledge graph storage. |
| `DASK_` | `DASK_N_WORKERS`, `DASK_THREADS_PER_WORKER`, `DASK_PROCESSES` | Worker parallelism. |
| `EMBEDDING_` | `EMBEDDING_MMBERT_URL`, `EMBEDDING_QWEN_VL_URL`, `EMBEDDING_SPLADE_URL` | Shared embedding service URLs. |
| `TRITON_` | `TRITON_URL`, `TRITON_TIMEOUT` | Triton inference server defaults. |
| `TRACKER_` | `TRACKER_URL`, `TRACKER_TIMEOUT` | Optional progress callback endpoint. |
| `TASK_` | `TASK_DEFAULT_RETRIES`, `TASK_DEFAULT_TIMEOUT` | Global task defaults. |

Other commonly required variables include `APP_ENV`, `PREFECT_API_URL`, `OPENROUTER_API_KEY`, and provider-specific LLM keys used by the configured LLM clients.

Environment-specific YAML is loaded from:

```text
src/video_pipeline/config/environments/{APP_ENV}.yaml
```

The default `APP_ENV` is `dev`.

## Local Setup

Install dependencies:

```bash
uv sync
```

Install worker extras for the full pipeline:

```bash
uv sync --extra worker
```

Start the API locally:

```bash
uv run video-pipeline-api
```

Register deployments and start a local Prefect worker after the Prefect server is available:

```bash
bash scripts/prefect-init.sh
prefect worker start --pool local-pool
```

The API schedules `local-deployment`, defined in `prefect.yaml`.

## Docker Setup

The Compose stack is in `docker/docker-compose.yaml`. It expects an external Docker network named `video_shared_net` and a `docker/.env` file.

Create the shared network once:

```bash
docker network create video_shared_net
```

Create or edit `docker/.env` with at least:

```env
DOCKER_HOST=localhost
DOCKER_LOCAL_VOLUME=/absolute/path/for/video-pipeline-data
```

Start the stack:

```bash
cd docker
docker compose up -d
```

Main services:

| Service | Port | Notes |
|---------|------|-------|
| `prefect-server` | 4200 | Prefect API and UI. |
| `prefect-services` | none | Prefect background services. |
| `prefect-worker` | 8787 | Registers deployments and starts `local-pool`. |
| `video-pipeline-api` | 8050 | FastAPI submission API. |
| `postgres` | 5432 | PostgreSQL. |
| `redis` | 6379 | Prefect messaging. |
| `minio` | 9000 / 9001 | S3 API and console. |
| `qdrant` | 6333 / 6334 | Vector database. |
| `arangodb` | 8529 | Graph database with experimental vector index enabled. |
| `elasticsearch` | 9200 | OCR search index. |

Check API docs:

```text
http://localhost:8050/docs
```

Check Prefect UI:

```text
http://localhost:4200
```

## Development

Run tests:

```bash
uv run pytest
```

Run tests with coverage:

```bash
uv run pytest --cov=video_pipeline --cov-report=html
```

Run Ruff:

```bash
uv run ruff check .
uv run ruff format .
```

Ruff is configured in `pyproject.toml` with Python 3.12 targeting and a 100-character line length.

## Project Layout

```text
video_pipeline/
|-- docker/
|   |-- docker-compose.yaml
|   |-- Dockerfile.api
|   `-- Dockerfile.worker
|-- scripts/
|   `-- prefect-init.sh
|-- src/video_pipeline/
|   |-- api/
|   |   |-- app.py
|   |   |-- lifespan.py
|   |   |-- routers/
|   |   `-- services/
|   |-- config/
|   |   |-- settings.py
|   |   |-- tasks.yaml
|   |   `-- environments/
|   |-- core/
|   |   |-- artifact/
|   |   |-- client/
|   |   |-- state.py
|   |   `-- storage/
|   |-- flow/
|   |   |-- main.py
|   |   |-- subtask.py
|   |   `-- batch_helper.py
|   `-- task/
|       |-- arango_indexing/
|       |-- asr/
|       |-- audio_segment/
|       |-- audio_transcript_embedding/
|       |-- autoshot/
|       |-- image_caption_embedding/
|       |-- image_caption_ocr/
|       |-- image_embedding/
|       |-- image_extraction/
|       |-- kg_graph/
|       |-- ocr_indexing/
|       |-- qdrant_indexing/
|       |-- segment_caption/
|       |-- segment_caption_embedding/
|       |-- segment_embedding/
|       `-- video/
|-- prefect.yaml
|-- pyproject.toml
`-- uv.lock
```
