# Video Pipeline

A scalable, multi-modal video processing and ingestion system. Automatically extracts rich metadata, generates embeddings, indexes text, and constructs knowledge graphs from video content — preparing it for intelligent semantic search and retrieval by the [VideoDeepSearch](../videodeepsearch/README.md) agent system.

Built on [Prefect](https://www.prefect.io/) for orchestration, [FastAPI](https://fastapi.tiangolo.com/) for the ingestion API, and a suite of specialized ML inference services.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Pipeline Stages](#pipeline-stages)
  - [Audio Branch](#audio-branch)
  - [Image Branch](#image-branch)
  - [Knowledge Graph Branch](#knowledge-graph-branch)
- [Storage Backends](#storage-backends)
- [Inference Services](#inference-services)
- [Artifact System & Data Lineage](#artifact-system--data-lineage)
- [API Reference](#api-reference)
- [Knowledge Graph Details](#knowledge-graph-details)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Docker Setup](#docker-setup)
  - [Local Development](#local-development)
- [Configuration](#configuration)
- [Code Quality](#code-quality)

---

## Overview

Video Pipeline ingests raw video files and transforms them into a rich, multi-modal index ready for intelligent search. For each video, the pipeline:

1. Detects shot boundaries and extracts key frames.
2. Runs Automatic Speech Recognition (ASR) on the audio track.
3. Groups transcript segments semantically using an LLM.
4. Captions frames and segments using Vision Language Models.
5. Generates dense and sparse embeddings for all modalities (image, caption, audio text).
6. Indexes all vectors into Qdrant, OCR text into Elasticsearch, and builds a full Knowledge Graph stored in ArangoDB.

Processing is **fully asynchronous and parallel** using Prefect's `DaskTaskRunner`, with every stage producing typed **Artifacts** that track data lineage through the entire pipeline.

---

## Architecture

The system consists of three main components:

```
┌─────────────────────────────────────────────────────────────┐
│                   FastAPI Submission API                     │
│              POST /uploads  →  Prefect Deployment            │
│              DELETE /videos/{id}  ·  GET /videos/{id}/full  │
└────────────────────────┬────────────────────────────────────┘
                         │ triggers
                         ▼
┌─────────────────────────────────────────────────────────────┐
│           Prefect Flow  (DaskTaskRunner)                     │
│                                                              │
│  Video Registration                                          │
│       │                                                      │
│       ▼                                                      │
│  Shot Detection (Autoshot)                                   │
│       │                                                      │
│       ▼                                                      │
│  Preprocessing (split into batches)                          │
│       │                    │                                 │
│  ┌────▼────┐         ┌─────▼────┐                           │
│  │  AUDIO  │         │  IMAGE   │                           │
│  │ BRANCH  │         │  BRANCH  │                           │
│  └────┬────┘         └─────┬────┘                           │
│       │                    │                                 │
│  ASR → Segmentation   Extraction → Caption+OCR              │
│  Seg Caption          Image Embedding → Qdrant               │
│  Seg Embedding → Qdrant  Caption Embed → Qdrant             │
│  Audio Embed → Qdrant     OCR → Elasticsearch               │
│       │                                                      │
│  Knowledge Graph Pipeline                                    │
│       └──────────→ ArangoDB Indexing                        │
└─────────────────────────────────────────────────────────────┘
                         │ stores into
                         ▼
┌───────────┬──────────────┬──────────┬───────────┬──────────┐
│  MinIO    │  PostgreSQL  │  Qdrant  │  Arango   │  Elastic │
│  (files)  │  (lineage)   │ (vectors)│  (graph)  │  (OCR)   │
└───────────┴──────────────┴──────────┴───────────┴──────────┘
```

---

## Pipeline Stages

The main flow (`single_video_processing_flow`) processes a video through up to **20 stages** in two parallel branches. Each stage is a Prefect task with configurable retries, timeouts, and caching.

### Audio Branch

| Stage | Task | Description |
|-------|------|-------------|
| 1 | **Video Registration** | Extract and store video metadata: fps, duration, format. Creates `VideoArtifact`. |
| 2 | **Shot Detection** | Call the Autoshot service to detect scene boundaries. Creates `AutoshotArtifact`. |
| 3 | **Preprocessing** | Split video into ASR and image batches with configurable batch sizes. |
| 4 | **ASR** | Send audio batches to the ASR inference service (speech-to-text). Creates `ASRArtifact` per batch. |
| 5 | **Audio Segmentation** | Use an LLM to semantically merge ASR chunks into coherent `AudioSegmentArtifact` objects with `start_sec`/`end_sec` timestamps. |
| 6 | **Segment Captioning** | For each audio segment, call a VLM to produce a `summary_caption` and a list of `event_captions`. Tracks LLM cost. Creates `SegmentCaptionArtifact`. |
| 7 | **Segment Embedding** | Generate multimodal embeddings (frames + caption text) via QwenVL for each segment. Creates `SegmentEmbeddingArtifact` (dim=1536). |
| 8 | **Audio Transcript Embedding** | Generate dense text embeddings from raw ASR `audio_text` via mmBERT. Creates `AudioTranscriptEmbedArtifact` (dim=768). |
| 9 | **Segment Qdrant Indexing** | Insert segment multimodal embeddings into Qdrant for retrieval. |
| 10 | **Audio Transcript Qdrant Indexing** | Insert audio text embeddings into Qdrant for spoken-content search. |
| 11 | **Segment Caption Embedding** | Generate text-only embeddings for segment captions via mmBERT. Creates `TextCapSegmentEmbedArtifact`. |
| 12 | **Segment Caption Qdrant Indexing** | Insert segment caption text embeddings into Qdrant. |

### Image Branch

| Stage | Task | Description |
|-------|------|-------------|
| 3 | **Image Extraction** | Extract representative frames from the video using OpenCV/FFmpeg. Creates `ImageArtifact` per frame. Frames stored in MinIO. |
| 4 | **Image Caption + OCR** | Single combined call: generate caption and extract text (OCR) from each frame via VLM. Creates `ImageCaptionArtifact` and `ImageOCRArtifact`. Tracks LLM cost. |
| 5 | **Image Embedding** | Generate multimodal embeddings (image pixel data + caption text) via QwenVL. Creates `ImageEmbeddingArtifact`. |
| 6 | **Image Qdrant Indexing** | Index image multimodal embeddings into Qdrant for frame-level visual search. |
| 7 | **OCR Elasticsearch Indexing** | Index all OCR text from frames into Elasticsearch for full-text BM25 search. |
| 8 | **Image Caption Embedding** | Generate dense text embeddings for image captions via mmBERT. Creates `TextCaptionEmbeddingArtifact`. |
| 9 | **Image Caption Qdrant Indexing** | Index image caption text embeddings into Qdrant. |

### Knowledge Graph Branch

The KG pipeline runs after segment captioning is complete, consuming `SegmentCaptionArtifact` objects.

| Stage | Description |
|-------|-------------|
| **KG Extraction** | For each segment, an LLM extracts structured `Entity` (people, objects, places) and `Relationship` objects from the event captions and summary. IDs are normalized to globally unique identifiers. |
| **Entity Resolution** | Flatten entities across all segments, compute hybrid similarity (dense + sparse), cluster with `AgglomerativeClustering`, and use an LLM to confirm canonical real-world entities. Reduces duplicates. |
| **Event Linking** | Build a two-level event hierarchy: segment-level `EventNode`s and frame-level `MicroEventNode`s. A 4-pass algorithm determines event linkages based on semantic similarity, temporal proximity, and shared entities. |
| **Community Detection** | Build an entity-relationship graph, apply the **Leiden algorithm** (via `python-igraph`) to detect communities, and use an LLM to generate a community summary. |
| **Node2Vec Embeddings** | Train Node2Vec structural embeddings on three graph views: (1) entities only, (2) entities + micro events, (3) full graph. Stored per node. |
| **ArangoDB Indexing** | Persist the entire knowledge graph into ArangoDB with 5 vertex collections and 8 edge collections. |

---

## Storage Backends

| Backend | Port | Purpose |
|---------|------|---------|
| **MinIO** | 9000 / 9001 | Object storage for raw videos, extracted frames. S3-compatible API. |
| **PostgreSQL** | 5432 | Artifact registry and data lineage tracking. Every artifact is persisted with its `lineage_parents`. |
| **Qdrant** | 6333 / 6334 | Vector database. Stores dense embeddings for images, image captions, segment multimodal, audio transcripts, and segment captions. |
| **Elasticsearch** | 9200 | Full-text BM25 search over OCR text extracted from video frames. |
| **ArangoDB** | 8529 | Graph database for the knowledge graph. Supports experimental vector index. |

### Qdrant Collections

| Collection | Embedding Model | Artifact Type |
|------------|----------------|---------------|
| Images (multimodal) | QwenVL | `ImageEmbeddingArtifact` |
| Image Captions (text) | mmBERT | `TextCaptionEmbeddingArtifact` |
| Segments (multimodal) | QwenVL | `SegmentEmbeddingArtifact` |
| Segment Captions (text) | mmBERT | `TextCapSegmentEmbedArtifact` |
| Audio Transcripts (text) | mmBERT | `AudioTranscriptEmbedArtifact` |

### ArangoDB Collections

**Vertex Collections:**
- `videos` — One document per ingested video
- `entities` — Canonical entities (people, objects, locations)
- `events` — Segment-level event nodes
- `micro_events` — Frame-level micro-event nodes
- `communities` — Detected thematic clusters with LLM summaries

**Edge Collections:**
- `entity_relations` — entity ↔ entity relationships
- `event_sequences` — event ↔ event temporal/semantic links
- `event_entities` — event ↔ entity associations
- `micro_event_sequences` — micro_event ↔ micro_event links
- `micro_event_parents` — micro_event → event (parent)
- `micro_event_entities` — micro_event ↔ entity
- `community_members` — entity → community
- `event_communities` — event → community

---

## Inference Services

All ML inference is offloaded to external services accessed via HTTP/gRPC clients:

| Client | Service | Usage |
|--------|---------|-------|
| `asr_client.py` | ASR service | Speech-to-text transcription |
| `autoshot_client.py` | Autoshot service | Shot/scene boundary detection |
| `ocr_client.py` | OCR service | Text extraction from frames |
| `qwenvl_embed.py` | QwenVL Triton server | Multimodal (image + text) embeddings (dim=1536) |
| `te_client.py` | Text Encoder (mmBERT) | Dense text embeddings (dim=768) |
| `splade_client.py` | SPLADE service | Sparse text embeddings for hybrid search |
| LLM (LlamaIndex) | OpenRouter / Google GenAI | Segment captioning, entity extraction, community summaries |

---

## Artifact System & Data Lineage

Every processing stage produces a typed **Artifact** (a Pydantic model inheriting from `BaseArtifact`). Each artifact:

- Gets a unique `artifact_id` (UUID4).
- Declares `lineage_parents` — the IDs of upstream artifacts it was derived from.
- Is persisted to PostgreSQL via the `ArtifactPersistentVisitor` pattern.
- May also store binary data in MinIO (via `minio_url_path`).

This creates a full **data lineage graph** allowing you to trace any artifact back to its source video.

```
VideoArtifact
  └── AutoshotArtifact
        ├── ASRArtifact (×N)
        │     └── AudioSegmentArtifact (×M)
        │           ├── SegmentCaptionArtifact
        │           │     ├── SegmentEmbeddingArtifact
        │           │     ├── TextCapSegmentEmbedArtifact
        │           │     └── KGGraphArtifact
        │           │           └── ArangoIndexingArtifact
        │           └── AudioTranscriptEmbedArtifact
        └── ImageArtifact (×K)
              ├── ImageCaptionArtifact
              │     ├── ImageEmbeddingArtifact
              │     └── TextCaptionEmbeddingArtifact
              └── ImageOCRArtifact
```

### Full Artifact Catalogue

| Artifact | Key Fields |
|----------|-----------|
| `VideoArtifact` | `video_id`, `fps`, `video_minio_url`, `video_extension` |
| `AutoshotArtifact` | `related_video_id`, `related_video_fps` |
| `ASRArtifact` | `related_autoshot_artifact_id`, `related_video_id` |
| `AudioSegmentArtifact` | `segment_index`, `start_sec`, `end_sec`, `audio_text`, `start_frame`, `end_frame` |
| `AudioTranscriptEmbedArtifact` | `embedding_dim=768`, `audio_text`, timestamp fields |
| `SegmentCaptionArtifact` | `summary_caption`, `event_captions: list[str]` |
| `SegmentEmbeddingArtifact` | `embedding_dim=1536`, `frame_indices`, `caption_text` |
| `TextCapSegmentEmbedArtifact` | `related_segment_caption_url`, `segment_cap_id` |
| `SegmentCaptionMultimodalEmbedArtifact` | QwenVL multimodal segment caption embedding |
| `ImageArtifact` | `frame_index`, `timestamp_sec`, `content_type`, `autoshot_artifact_id` |
| `ImageCaptionArtifact` | `frame_index`, `timestamp`, `image_minio_url` |
| `ImageOCRArtifact` | `frame_index`, `timestamp`, `image_minio_url` |
| `ImageEmbeddingArtifact` | `caption_text`, `image_minio_url`, multimodal (QwenVL) |
| `TextCaptionEmbeddingArtifact` | `caption_id`, `image_id` (mmBERT text-only) |
| `ImageCaptionMultimodalEmbeddingArtifact` | `caption_id`, `image_id` (QwenVL multimodal) |
| `KGGraphArtifact` | entities, events, micro_events, communities, node2vec_meta, LLM cost stats |
| `ArangoIndexingArtifact` | counts for all vertex/edge collections inserted |

---

## API Reference

The API runs on port **8050**. Interactive docs are available at `http://localhost:8050/docs`.

### `POST /uploads/`

Submit one or more videos for asynchronous processing.

**Request body:**
```json
{
  "user_id": "user-abc",
  "videos": [
    {
      "video_id": "vid-001",
      "video_url": "s3://my-bucket/videos/lecture.mp4"
    }
  ]
}
```

**Response** (`202 Accepted`):
```json
{
  "run_id": "<uuid>",
  "user_id": "user-abc",
  "video_count": 1,
  "results": [
    {
      "video_id": "vid-001",
      "flow_run_id": "<prefect-flow-run-id>",
      "state": "SCHEDULED"
    }
  ],
  "status": "SCHEDULED",
  "message": "1 video(s) submitted for processing",
  "deployment_name": "<deployment-name>"
}
```

Each video triggers an independent Prefect flow run. The API uses idempotency keys — resubmitting the same `run_id + video_id` combination will not create duplicate runs.

### `DELETE /videos/{video_id}`

Cascade-delete all artifacts for a video across **all** storage backends:

1. Query PostgreSQL for all related artifacts
2. Delete objects from MinIO
3. Delete lineage and artifact records from PostgreSQL
4. Delete vectors from Qdrant
5. Delete knowledge graph nodes/edges from ArangoDB
6. Delete OCR documents from Elasticsearch

**Response:**
```json
{
  "video_id": "vid-001",
  "postgres": { "deleted_artifacts": 42, "deleted_lineage": 38 },
  "minio": { "deleted_objects": 120 },
  "qdrant": { "deleted_vectors": 350 },
  "arango": { "deleted_nodes": 58, "deleted_edges": 94 },
  "elasticsearch": { "deleted_documents": 80 },
  "summary": { ... }
}
```

### `GET /videos/{video_id}/full`

Retrieve all indexed data for a video from all backends.

**Query parameters:**
- `sources` (optional): comma-separated subset, e.g. `postgres,arango,qdrant,elasticsearch`
- `include_vectors` (bool, default `false`): include raw embedding vectors in Qdrant results

### `GET /api/health`

Returns per-service readiness status.

### `GET /api/health/prefect`

Returns Prefect server and deployment health.

---

## Knowledge Graph Details

The KG pipeline runs 5 stages on `SegmentCaptionArtifact` data:

### Stage 1: KG Extraction (`extract_kg.py`)

For each segment, an LLM extracts a structured `VideoGraphExtraction`:

```python
class Entity(BaseModel):
    entity_id: str        # global UUID after normalization
    name: str
    entity_type: str      # person | object | location | concept

class Relationship(BaseModel):
    source_id: str
    target_id: str
    relation: str

class Event(BaseModel):
    event_id: str
    event_des: str        # concise description of the action

class VideoGraphExtraction(BaseModel):
    entities: list[Entity]
    relations: list[Relationship]
```

### Stage 2: Entity Resolution (`entity_resolution.py`)

- Flatten all entities across segments
- Compute hybrid similarity: dense (QwenVL/mmBERT) + sparse (SPLADE)
- Cluster with `AgglomerativeClustering`
- For each cluster, ask an LLM: "Do these refer to the same real-world entity?"
- Produce canonical entities, reducing duplicates across the video

### Stage 3: Event Linking (`event_linking.py`)

Builds a two-level hierarchy:
- **EventNode** (segment-level): dense embedding from the segment caption, linked to entities
- **MicroEventNode** (frame-level): extracted from individual event captions

A 4-pass algorithm determines event links based on:
1. Semantic embedding similarity
2. Temporal proximity of timestamps
3. Shared entity overlap
4. LLM confirmation for borderline cases

### Stage 4: Community Detection (`community_detection.py`)

- Builds an entity-relationship graph using `python-igraph`
- Runs the **Leiden algorithm** (`leidenalg`) for community detection
- For each detected community, generates an LLM summary of the thematic cluster
- Computes and stores `graph_modularity`

### Stage 5: Node2Vec Embeddings (`node2vec.py`)

Trains structural graph embeddings on 3 graph views:
1. Entities only
2. Entities + micro events
3. Full graph (entities + events + micro-events + communities)

Final `KGGraphArtifact` includes aggregated statistics:

| Field | Description |
|-------|-------------|
| `total_raw_entities` | Entities before resolution |
| `total_canonical_entities` | Entities after deduplication |
| `total_relationships` | Unique entity-entity edges |
| `total_events` | Segment-level event nodes |
| `total_micro_events` | Frame-level micro-event nodes |
| `total_communities` | Detected communities |
| `graph_modularity` | Leiden modularity score (0–1) |
| `total_nodes_with_embeddings` | Nodes with Node2Vec embeddings |
| `total_llm_cost` | Total USD cost for all LLM calls in this video |
| `llm_calls` | Number of LLM API calls made |

---

## Project Structure

```
video_pipeline/
├── docker/
│   ├── docker-compose.yaml          # All services: Prefect, API, MinIO, Qdrant, Arango, ES, PG
│   ├── Dockerfile.api               # FastAPI submission API image
│   └── Dockerfile.worker            # Prefect worker image (includes ML extras)
├── scripts/
│   └── prefect-init.sh              # Worker bootstrap: register deployment
├── src/video_pipeline/
│   ├── api/
│   │   ├── app.py                   # FastAPI app, routers, global exception handler
│   │   ├── lifespan.py              # Startup: load settings, resolve deployment name
│   │   ├── routers/
│   │   │   ├── upload.py            # POST /uploads/ — trigger Prefect flow
│   │   │   ├── videos.py            # DELETE + GET /videos/{id}
│   │   │   └── health.py            # GET /api/health, /api/health/prefect
│   │   └── services/
│   │       ├── deletion.py          # VideoDeletionService (cascade across all backends)
│   │       └── retrieval.py         # VideoRetrievalService (multi-source fetch)
│   ├── flow/
│   │   ├── main.py                  # single_video_processing_flow (1038 lines)
│   │   │                            # TimingRegistry: records wall-clock per stage
│   │   │                            # Audio branch + Image branch (parallel)
│   │   └── subtask.py               # preprocess_video_task: splits into batches
│   ├── task/
│   │   ├── base/base_task.py        # BaseTask: preprocess/execute/postprocess lifecycle
│   │   ├── video/                   # VideoRegistryTask
│   │   ├── autoshot/                # AutoshotTask — Autoshot HTTP client
│   │   ├── asr/                     # ASRTask — ASR HTTP client
│   │   ├── audio_segment/           # AudioSegmentTask — LLM-based chunking
│   │   ├── audio_transcript_embedding/ # mmBERT audio text embeddings
│   │   ├── image_extraction/        # ImageExtractionTask — OpenCV frame extraction
│   │   ├── image_caption_ocr/       # Combined VLM caption + OCR task
│   │   ├── image_embedding/         # QwenVL multimodal image embedding
│   │   ├── image_caption_embedding/ # mmBERT image caption text embedding
│   │   ├── segment_caption/         # VLM segment summary + event captions
│   │   ├── segment_embedding/       # QwenVL multimodal segment embedding
│   │   ├── segment_caption_embedding/ # mmBERT segment caption text embedding
│   │   ├── ocr_indexing/            # Elasticsearch OCR indexing
│   │   ├── qdrant_indexing/         # 5 Qdrant indexing tasks (image, caption, segment, audio)
│   │   ├── kg_graph/                # Full KG pipeline (5 stages)
│   │   │   ├── extract_kg.py        # LLM entity/relation extraction
│   │   │   ├── entity_resolution.py # Hybrid clustering + LLM deduplication
│   │   │   ├── event_linking.py     # 4-pass event linkage (EventNode + MicroEventNode)
│   │   │   ├── community_detection.py # Leiden algorithm + LLM summaries
│   │   │   ├── node2vec.py          # Node2Vec structural embeddings
│   │   │   ├── node2vec_embeddings.py # Embedding storage helpers
│   │   │   ├── models.py            # KG data models + CostTracker
│   │   │   └── prompt.py            # LLM prompt templates
│   │   └── arango_indexing/         # ArangoIndexingTask — write graph to ArangoDB
│   ├── core/
│   │   ├── artifact/artifact.py     # All 16 typed Artifact Pydantic models
│   │   ├── state.py                 # Shared mutable state (singleton clients)
│   │   ├── storage/                 # ArtifactPersistentVisitor (PostgreSQL lineage)
│   │   └── client/
│   │       ├── inference/           # asr, autoshot, ocr, qwenvl, te, splade clients
│   │       ├── storage/             # pg, minio, qdrant, elasticsearch, arango clients
│   │       ├── llm_provider/        # LlamaIndex LLM wrappers
│   │       └── progress/            # HTTPProgressTracker, StageRegistry
│   └── config/
│       ├── settings.py              # Pydantic-settings with all env vars
│       └── tasks.yaml               # Per-task: retries, timeouts, batch sizes, model URLs
```

---

## Getting Started

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- Docker and Docker Compose
- Access to external ML inference services (ASR, Autoshot, QwenVL, mmBERT, SPLADE)
- OpenRouter or Google GenAI API key (for LLM captioning and KG extraction)

### Installation

```bash
# Base dependencies (API + Prefect orchestration)
uv sync

# Full dependencies including ML worker extras (OpenCV, PyTorch, etc.)
uv sync --extra worker
```

### Docker Setup

All infrastructure services are configured in `docker/docker-compose.yaml`.

**Step 1: Create the shared Docker network:**
```bash
docker network create video_shared_net
```

**Step 2: Configure environment:**
```bash
cp docker/.env.example docker/.env
# Edit docker/.env with your service credentials and host settings
```

**Step 3: Start all services:**
```bash
cd docker && docker-compose up -d
```

This starts the following services:

| Service | Port | Purpose |
|---------|------|---------|
| `prefect-server` | 4200 | Prefect orchestration UI and API |
| `prefect-services` | — | Prefect background services (scheduler, etc.) |
| `prefect-worker` | 8787 | Dask task runner (executes pipeline tasks) |
| `video-pipeline-api` | 8050 | FastAPI submission API |
| `minio` | 9000 / 9001 | Object storage (API / Console UI) |
| `qdrant` | 6333 / 6334 | Vector database |
| `arangodb` | 8529 | Graph database |
| `elasticsearch` | 9200 | Full-text search |
| `postgres` | 5432 | Prefect backend + artifact metadata |
| `redis` | — | Prefect messaging broker |

### Local Development

To run the API server locally (without Docker):

```bash
# Start the API on port 8050
video-pipeline-api

# Or via uv
uv run video-pipeline-api
```

The Prefect worker must still be running (either via Docker or locally):

```bash
# Register the deployment first
bash scripts/prefect-init.sh

# Start the worker
prefect worker start --pool local-pool
```

Monitor pipeline runs on the Prefect UI at `http://localhost:4200`.

---

## Configuration

All settings are managed via Pydantic-settings in `src/video_pipeline/config/settings.py`, loaded from environment variables (`.env` file in the `docker/` directory).

Key configuration sections:

| Section | Variables | Description |
|---------|-----------|-------------|
| Database | `POSTGRES_*` | PostgreSQL connection for artifact lineage |
| MinIO | `MINIO_*` | Object storage credentials and endpoint |
| Qdrant | `QDRANT_HOST`, `QDRANT_PORT` | Vector DB connection |
| Elasticsearch | `ES_HOST`, `ES_PORT` | Full-text search connection |
| ArangoDB | `ARANGO_HOST`, `ARANGO_PORT`, `ARANGO_DB` | Graph DB connection |
| Inference | `ASR_URL`, `AUTOSHOT_URL`, `QWENVL_URL`, etc. | External ML service endpoints |
| LLM | `OPENROUTER_API_KEY` / `GOOGLE_API_KEY` | LLM provider credentials |
| Tracker | `TRACKER_URL` | Optional progress callback endpoint |
| Prefect | `PREFECT_API_URL` | Prefect server URL |

Task-specific settings (batch sizes, timeouts, model URLs, retry counts) are configured in `src/video_pipeline/config/tasks.yaml` and loaded by `BaseTask` at runtime.

---

## Code Quality

```bash
# Lint and format with Ruff
uv run ruff check .
uv run ruff format .

# Run all pre-commit hooks
pre-commit run --all-files

# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=video_pipeline --cov-report=html
```

Ruff is configured to enforce: `E`/`F`/`W` (correctness), `I` (imports), `UP` (pyupgrade), `B` (bugbear), `SIM` (simplifications), `C4` (comprehensions), `ANN` (type annotations).
