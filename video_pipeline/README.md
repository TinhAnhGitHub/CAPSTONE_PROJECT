# Video Pipeline

Video Pipeline is the ingestion and indexing service for the Capstone video search system. It turns raw videos into structured artifacts that can be searched by VideoDeepSearch through vector search, OCR search, metadata lookup, and graph retrieval.

The service is implemented as a Python package under `src/video_pipeline`. It exposes a FastAPI submission API, schedules asynchronous Prefect flow runs, executes the processing graph with Dask, and persists outputs into MinIO, PostgreSQL, Qdrant, Elasticsearch, and ArangoDB.

## Table Of Contents

- [What This Service Does](#what-this-service-does)
- [Runtime Architecture](#runtime-architecture)
- [Pipeline Stages](#pipeline-stages)
- [Task Execution And Dask](#task-execution-and-dask)
- [Artifacts And Lineage](#artifacts-and-lineage)
- [Storage Backends](#storage-backends)
- [Knowledge Graph](#knowledge-graph)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Local Development](#local-development)
- [Docker Development](#docker-development)
- [Operations And Debugging](#operations-and-debugging)
- [Project Layout](#project-layout)
- [Task Documentation](#task-documentation)

## What This Service Does

The pipeline ingests one or more videos and produces retrieval-ready data:

| Output | Purpose |
|--------|---------|
| Video metadata | FPS, duration, extension, object path, and user/video identity. |
| Shot boundaries | Scene/shot frame intervals used for downstream batching. |
| ASR transcripts | Spoken content aligned to frame and timestamp ranges. |
| Audio segments | Conservative semantic chunks built from raw ASR segments. |
| Extracted images | Representative frames from video segments. |
| Image captions and OCR | VLM-generated frame captions plus visible text. |
| Multimodal embeddings | QwenVL embeddings for images and temporal segments. |
| Text embeddings | mmBERT embeddings for captions and transcripts. |
| Sparse vectors | SPLADE sparse vectors for hybrid text retrieval in Qdrant. |
| OCR index | Elasticsearch documents for visible text search. |
| Knowledge graph | Canonical entities, events, micro-events, and graph edges in ArangoDB. |

## Runtime Architecture

```text
Client or frontend
  |
  | POST /uploads/
  v
FastAPI API on :8050
  |
  | prefect.deployments.run_deployment(...)
  v
Prefect Server on :4200
  |
  | schedules flow run
  v
Prefect Worker
  |
  | DaskTaskRunner
  v
single_video_processing_flow
  |
  +-- MinIO: raw files, extracted images, JSON payloads, .npy vectors
  +-- PostgreSQL: artifact metadata and lineage
  +-- Qdrant: vector indexes
  +-- Elasticsearch: OCR text index
  +-- ArangoDB: knowledge graph
  +-- External inference: Triton, Qwen ASR, QwenVL, mmBERT, SPLADE, OpenRouter LLMs
```

Important components:

| Component | Code location | Responsibility |
|-----------|---------------|----------------|
| FastAPI app | `src/video_pipeline/api/app.py` | HTTP API, CORS, router registration, global exception handling. |
| API lifespan | `src/video_pipeline/api/lifespan.py` | Checks whether the configured Prefect deployment exists. |
| Main flow | `src/video_pipeline/flow/main.py` | Orchestrates the full single-video pipeline. |
| Video preprocess subtask | `src/video_pipeline/flow/subtask.py` | Creates ASR and image extraction batches from shot boundaries. |
| Task base class | `src/video_pipeline/task/base/base_task.py` | Shared `preprocess -> execute -> postprocess -> summary` lifecycle. |
| Task config | `src/video_pipeline/config/tasks.yaml` | Names, retries, timeouts, cache keys, model endpoints, batch/concurrency settings. |
| Settings | `src/video_pipeline/config/settings.py` | Environment and YAML-backed service configuration. |
| Artifact models | `src/video_pipeline/core/artifact/artifact.py` | Typed Pydantic artifact definitions. |

## Pipeline Stages

The Prefect deployment entrypoint is `src/video_pipeline/flow/main.py:single_video_processing_flow`.

| Stage | Main task | Input | Output | Notes |
|-------|-----------|-------|--------|-------|
| 1. Video registration | `video_reg_task` | `VideoInput` | `VideoArtifact` | Downloads source video temporarily, probes FPS and duration, persists metadata. |
| 2. Shot detection | `autoshot_task` | `VideoArtifact` | `AutoshotArtifact` | Extracts frames, batches model windows, predicts shot boundaries. |
| 3. Video preprocess | `preprocess_video_task` | `AutoshotArtifact` | ASR batches and image batches | Extracts audio chunks and representative frame indices. |
| 4. ASR | `asr_chunk_task.map(...)` | ASR batches | `ASRArtifact` list | Batched Qwen ASR transcription. |
| 5. Audio segmentation | `audio_segment_task` | all ASR artifacts | `AudioSegmentArtifact` list | Preserves small inputs, uses LLM merge rules for large inputs. |
| 6. Segment caption | `segment_caption_chunk_task.map(...)` | audio segment batches | `SegmentCaptionArtifact` list | VLM sees sampled frames plus transcript. |
| 7. Segment embedding | `segment_embedding_chunk_task.map(...)` | segment captions | `SegmentEmbeddingArtifact` list | QwenVL multimodal segment vectors. |
| 8. Audio transcript embedding | `audio_transcript_embedding_chunk_task.map(...)` | audio segments | `AudioTranscriptEmbedArtifact` list | mmBERT vectors over raw transcript text. |
| 9. Segment caption embedding | `segment_caption_embedding_chunk_task.map(...)` | segment captions | `TextCapSegmentEmbedArtifact` list | mmBERT vectors over summary captions. |
| 10. Image extraction | `image_chunk_task.map(...)` | image batches | `ImageArtifact` list | Extracts WebP representative frames. |
| 11. Image caption and OCR | `image_caption_ocr_chunk_task.map(...)` | image batches | `ImageCaptionArtifact`, `ImageOCRArtifact` | One structured VLM call per frame. |
| 12. Image embedding | `image_embedding_chunk_task.map(...)` | image captions | `ImageEmbeddingArtifact` list | QwenVL image+caption vectors. |
| 13. Image caption embedding | `image_caption_embedding_chunk_task.map(...)` | image captions | `TextCaptionEmbeddingArtifact` list | mmBERT caption vectors. |
| 14. Qdrant indexing | multiple `*_qdrant_indexing_chunk_task.map(...)` | embedding artifacts | point IDs | Dense and hybrid vector collections. |
| 15. OCR indexing | `ocr_indexing_chunk_task.map(...)` | OCR artifacts | document IDs | Elasticsearch OCR documents with optional embeddings. |
| 16. KG extraction | `kg_extraction_chunk_task.map(...)` | segment captions | `KGExtractionArtifact` list | LLM extracts entities and relations per segment batch. |
| 17. KG entity resolution | `kg_entity_resolution_task` | all KG extractions | `KGEntityResolutionResult` | Dense+sparse clustering and LLM duplicate resolution. |
| 18. KG finalization | `kg_finalization_task` | resolved KG | `KGGraphArtifact` | Builds event and micro-event layers. |
| 19. Arango indexing | `arango_indexing_task` | KG graph artifact | `ArangoIndexingArtifact` | Inserts graph vertices and edges. |

## Task Execution And Dask

The flow uses `DaskTaskRunner(cluster_kwargs=get_settings().dask.to_cluster_kwargs())`.

Execution patterns:

| Pattern | Used for | Why |
|---------|----------|-----|
| `.submit()` | Single video-level stages such as registration, autoshot, audio segmentation, entity resolution, KG finalization, and Arango indexing. | These stages require full-video context or produce one global artifact. |
| `.map(batches)` | ASR, image extraction, image caption/OCR, embeddings, Qdrant indexing, OCR indexing, KG extraction. | These stages can run independently over batches. |
| Internal `asyncio.gather()` | Autoshot model batches, VLM image calls, VLM segment calls, KG extraction calls. | Reduces latency for network-bound model calls inside one Dask task. |

Batch sizes and concurrency are configured in `tasks.yaml`, for example:

| Config key | Meaning |
|------------|---------|
| `video_preprocess.asr_batch_size` | Number of audio chunks per ASR mapped task. |
| `video_preprocess.image_batch_size` | Number of frame extraction requests per mapped task. |
| `video_preprocess.frames_per_segment` | Representative frames selected per detected shot. |
| `image_caption_ocr.max_concurrent` | Concurrent VLM calls inside one image caption/OCR task. |
| `segment_caption.max_concurrent` | Concurrent VLM calls inside one segment caption task. |
| `kg_extraction.max_concurrent` | Concurrent LLM KG extraction calls per KG extraction batch. |
| `kg_entity_resolution.max_entities_per_cluster` | Maximum candidate entities sent to one LLM verification call. |

## Artifacts And Lineage

All pipeline outputs are represented as typed artifacts. Artifacts are persisted to PostgreSQL, and binary payloads are stored in MinIO when needed. The artifact visitor is `ArtifactPersistentVisitor`.

Main artifact families:

| Artifact | Produced by | Contents |
|----------|-------------|----------|
| `VideoArtifact` | video registration | Source video metadata and MinIO URL. |
| `AutoshotArtifact` | autoshot | Shot frame intervals in `metadata.segments`. |
| `ASRArtifact` | ASR | Transcript text, frame range, timestamp range, duration. |
| `AudioSegmentArtifact` | audio segmentation | Merged/passthrough transcript segments. |
| `ImageArtifact` | image extraction | Extracted frame metadata and image object path. |
| `ImageCaptionArtifact` | image caption/OCR | Caption JSON metadata and source image link. |
| `ImageOCRArtifact` | image caption/OCR | OCR JSON metadata and source image link. |
| `ImageEmbeddingArtifact` | image embedding | QwenVL vector stored as `.npy`. |
| `TextCaptionEmbeddingArtifact` | image caption embedding | mmBERT caption vector stored as `.npy`. |
| `SegmentCaptionArtifact` | segment caption | Summary caption and atomic event captions. |
| `SegmentEmbeddingArtifact` | segment embedding | Multimodal segment vector stored as `.npy`. |
| `TextCapSegmentEmbedArtifact` | segment caption embedding | mmBERT segment-caption vector stored as `.npy`. |
| `AudioTranscriptEmbedArtifact` | audio transcript embedding | mmBERT transcript vector stored as `.npy`. |
| `KGExtractionArtifact` | KG extraction | Raw per-segment entities, relations, micro-events, and LLM cost. |
| `KGGraphArtifact` | KG finalization | Canonical graph data ready for ArangoDB. |
| `ArangoIndexingArtifact` | Arango indexing | Insert counts by graph collection. |

## Storage Backends

| Backend | Default port | Data stored |
|---------|--------------|-------------|
| MinIO | `9000`, console `9001` | Raw videos, extracted frames, caption/OCR JSON, embedding `.npy`, Prefect result storage. |
| PostgreSQL | `5432` | Artifact metadata, lineage records, Prefect database. |
| Redis | `6379` | Prefect messaging broker and cache. |
| Qdrant | `6333`, gRPC `6334` | Dense and sparse vector indexes. |
| Elasticsearch | `9200` | OCR text documents and searchable visible text. |
| ArangoDB | `8529` | Knowledge graph vertices and edges. |

Qdrant collections are built from `QDRANT_COLLECTION_BASE` and task constants:

| Collection suffix | Vector type |
|-------------------|-------------|
| `_image` | QwenVL image+caption dense vectors. |
| `_segment` | QwenVL segment dense vectors. |
| `_audio_transcript` | mmBERT dense plus SPLADE sparse vectors. |
| `_image_caption` | mmBERT dense plus SPLADE sparse vectors. |
| `_segment_caption` | mmBERT dense plus SPLADE sparse vectors. |

## Knowledge Graph

The KG path starts after `SegmentCaptionArtifact` objects exist.

Graph construction:

1. Each segment caption is converted into a `CaptionSegment` with timing, summary, and event captions.
2. KG extraction asks an LLM for entities and relationships using structured output.
3. Local per-segment IDs are replaced with UUID-backed IDs to avoid collisions.
4. Entity resolution flattens all raw entities and embeds `entity_name: desc` with mmBERT and SPLADE.
5. Hybrid dense+sparse similarity drives agglomerative clustering.
6. The LLM verifies each candidate cluster, splits false positives, creates canonical names, and merges descriptions.
7. Relationships are remapped from raw entity IDs to canonical global entity IDs and duplicate triples are collapsed with weights.
8. Event linking creates one event node per segment and one micro-event node per atomic event caption.
9. Event and micro-event edges are added by temporal order, semantic similarity, shared entity context, and LLM confirmation for borderline pairs.

ArangoDB vertex collections:

| Collection | Contents |
|------------|----------|
| `videos` | One summary document per processed video. |
| `entities` | Canonical entities with semantic embeddings. |
| `events` | Segment-level event nodes with captions and embeddings. |
| `micro_events` | Fine-grained event nodes with text and embeddings. |

ArangoDB edge collections:

| Collection | Edge meaning |
|------------|--------------|
| `entity_relations` | Canonical entity-to-entity triples. |
| `event_sequences` | Event-to-event temporal, semantic, context, or causal links. |
| `event_entities` | Event participation links to entities. |
| `micro_event_sequences` | Micro-event-to-micro-event links. |
| `micro_event_parents` | Micro-event to parent event hierarchy. |
| `micro_event_entities` | Micro-event participation links to entities. |

Detailed KG implementation notes are in `src/video_pipeline/task/kg_graph/README.md`.

## API Reference

The API runs on port `8050` by default.

Interactive docs:

```text
http://localhost:8050/docs
```

Root endpoint:

```http
GET /
```

Returns service metadata and endpoint hints.

### Submit Videos

```http
POST /uploads/
```

Request body:

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

Response body:

```json
{
  "run_id": "uuid",
  "user_id": "user-abc",
  "video_count": 1,
  "results": [
    {
      "video_id": "vid-001",
      "flow_run_id": "prefect-flow-run-id",
      "state": "SCHEDULED"
    }
  ],
  "status": "SCHEDULED",
  "message": "1 video(s) submitted for processing",
  "deployment_name": "Single Video Processing Flow/local-deployment"
}
```

The API returns `202 Accepted`. Processing continues asynchronously in Prefect.

### Retrieve Video Data

```http
GET /videos/{video_id}/full
```

Query parameters:

| Parameter | Type | Meaning |
|-----------|------|---------|
| `sources` | string | Optional comma-separated subset of `postgres,arango,qdrant,elasticsearch`. If omitted, all sources are queried. |
| `include_vectors` | bool | Include raw Qdrant vectors when `true`. Defaults to `false` because vector payloads are large. |

Example:

```bash
curl "http://localhost:8050/videos/vid-001/full?sources=postgres,arango&include_vectors=false"
```

### Delete Video Data

```http
DELETE /videos/{video_id}
```

Deletes video-related data from PostgreSQL, MinIO, Qdrant, ArangoDB, and Elasticsearch. The response includes per-backend deletion summaries.

### Health

```http
GET /health/
GET /health/prefect
```

`/health/prefect` checks Prefect API connectivity and whether the configured deployment can be read.

## Configuration

Settings are loaded from:

1. `video_pipeline/.env` if present.
2. `src/video_pipeline/config/environments/{APP_ENV}.yaml`, defaulting to `APP_ENV=dev`.
3. Environment variables, which override YAML where Pydantic settings are used.

Important environment variables:

| Variable or prefix | Purpose |
|--------------------|---------|
| `APP_ENV` | Selects environment YAML from `config/environments`. Defaults to `dev`. |
| `PREFECT_API_URL` | Prefect API endpoint used by API and worker. |
| `PREFECT_DEPLOYMENT_NAME` | Deployment suffix used by API startup and `/uploads/`. Set this to `local-deployment` when using the included `prefect.yaml`. |
| `OPENROUTER_API_KEY` | Required for LLM and VLM calls through OpenRouter. |
| `MINIO_*` | Object storage endpoint, credentials, and bucket defaults. |
| `POSTGRES_*` | Artifact and Prefect PostgreSQL connection. |
| `QDRANT_*` | Qdrant host, port, API key, and collection base. |
| `ELASTICSEARCH_*` | OCR search index configuration. |
| `ARANGO_*` | Graph database connection and graph name. |
| `DASK_*` | Local Dask worker count, threads, and process mode. |
| `TASK_*` | Global task retry and timeout defaults. |
| `TRITON_*` | Triton inference server URL and timeout. |
| `TRACKER_*` | Optional external HTTP progress tracker endpoint. |
| `EMBEDDING_*` | Central mmBERT, QwenVL, and SPLADE endpoint defaults. |

Deployment note:

`prefect.yaml` defines `local-deployment`, but `src/video_pipeline/api/lifespan.py` defaults `PREFECT_DEPLOYMENT_NAME` to `poc-deployment`. For the checked-in deployment file, set:

```env
PREFECT_DEPLOYMENT_NAME=local-deployment
```

Without this, `/uploads/` can fail with a missing Prefect deployment even when `local-deployment` is registered.

## Local Development

Install base dependencies:

```bash
uv sync
```

Install worker dependencies for the full ML pipeline:

```bash
uv sync --extra worker
```

Start the API:

```bash
uv run video-pipeline-api
```

Start the API directly with Uvicorn if needed:

```bash
uv run uvicorn video_pipeline.api.app:app --host 0.0.0.0 --port 8050
```

Register Prefect deployments after Prefect is available:

```bash
bash scripts/prefect-init.sh
```

Start a worker:

```bash
prefect worker start --pool local-pool
```

Run tests:

```bash
uv run pytest
```

Run tests with coverage:

```bash
uv run pytest --cov=video_pipeline --cov-report=html
```

Run linting and formatting:

```bash
uv run ruff check .
uv run ruff format .
```

## Docker Development

Compose file:

```text
video_pipeline/docker/docker-compose.yaml
```

Create the shared Docker network once:

```bash
docker network create video_shared_net
```

Create `video_pipeline/docker/.env`. At minimum, set:

```env
DOCKER_HOST=localhost
DOCKER_LOCAL_VOLUME=/absolute/path/for/video-pipeline-data
PREFECT_DEPLOYMENT_NAME=local-deployment
OPENROUTER_API_KEY=your-openrouter-key
```

Start the stack:

```bash
cd video_pipeline/docker
docker compose up -d
```

Services:

| Compose service | Port | Role |
|-----------------|------|------|
| `video-pipeline-api` | `8050` | FastAPI API. |
| `prefect-server` | `4200` | Prefect API and UI. |
| `prefect-services` | none | Prefect background services. |
| `prefect-worker` | `8787` | Worker process that registers deployments and starts `local-pool`. |
| `postgres` | `5432` | PostgreSQL. |
| `redis` | `6379` | Prefect Redis messaging. |
| `minio` | `9000`, `9001` | Object storage and console. |
| `qdrant` | `6333`, `6334` | Vector database. |
| `arangodb` | `8529` | Graph database with experimental vector index enabled. |
| `elasticsearch` | `9200` | OCR text search. |

Useful URLs:

```text
Video Pipeline API: http://localhost:8050/docs
Prefect UI:         http://localhost:4200
MinIO Console:     http://localhost:9001
Qdrant HTTP:       http://localhost:6333
ArangoDB UI:       http://localhost:8529
Elasticsearch:     http://localhost:9200
```

## Operations And Debugging

Check API health:

```bash
curl http://localhost:8050/health/
curl http://localhost:8050/health/prefect
```

Check whether the deployment exists:

```bash
prefect deployment ls
```

Expected deployment name for the included config:

```text
Single Video Processing Flow/local-deployment
```

Common issues:

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `/uploads/` returns deployment not found | `PREFECT_DEPLOYMENT_NAME` does not match `prefect.yaml`. | Set `PREFECT_DEPLOYMENT_NAME=local-deployment` and redeploy. |
| Worker starts but tasks cannot reach model services | Docker DNS names do not resolve or model containers are not on `video_shared_net`. | Put inference containers on the shared network or update endpoints in `tasks.yaml` and env vars. |
| MinIO object fetch fails | `video_url` bucket/object is wrong or credentials mismatch. | Verify the object exists and the bucket matches the submitted `s3://...` URL. |
| Qdrant vector size errors | Collection was created with a different vector dimension. | Delete/recreate the affected collection or use a different `QDRANT_COLLECTION_BASE`. |
| Arango edge insert errors | KG edge endpoints are malformed or graph collections are missing. | Check `arango_indexing` logs and Arango schema setup. |
| OpenRouter failures | Missing or invalid `OPENROUTER_API_KEY`. | Set the key in `.env` or environment. |

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
|       |-- base/
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

## Task Documentation

Detailed per-task documentation lives under `src/video_pipeline/task`:

| README | Covers |
|--------|--------|
| `src/video_pipeline/task/README.md` | Shared task lifecycle, Dask model, and stage summary. |
| `src/video_pipeline/task/video/README.md` | Video registration. |
| `src/video_pipeline/task/autoshot/README.md` | Shot detection. |
| `src/video_pipeline/task/asr/README.md` | ASR transcription. |
| `src/video_pipeline/task/audio_segment/README.md` | Audio segmentation and LLM merge rules. |
| `src/video_pipeline/task/image_extraction/README.md` | Representative frame extraction. |
| `src/video_pipeline/task/image_caption_ocr/README.md` | Combined image caption and OCR. |
| `src/video_pipeline/task/image_embedding/README.md` | QwenVL image embeddings. |
| `src/video_pipeline/task/image_caption_embedding/README.md` | mmBERT image-caption embeddings. |
| `src/video_pipeline/task/segment_caption/README.md` | Segment VLM captions and event captions. |
| `src/video_pipeline/task/segment_embedding/README.md` | Multimodal segment embeddings. |
| `src/video_pipeline/task/segment_caption_embedding/README.md` | Segment-caption text embeddings. |
| `src/video_pipeline/task/audio_transcript_embedding/README.md` | Transcript text embeddings. |
| `src/video_pipeline/task/qdrant_indexing/README.md` | Dense and hybrid vector indexing. |
| `src/video_pipeline/task/ocr_indexing/README.md` | OCR Elasticsearch indexing. |
| `src/video_pipeline/task/kg_graph/README.md` | KG extraction, entity resolution, event linking. |
| `src/video_pipeline/task/arango_indexing/README.md` | ArangoDB graph storage. |
| `src/video_pipeline/task/base/README.md` | Shared task infrastructure and cache keys. |
