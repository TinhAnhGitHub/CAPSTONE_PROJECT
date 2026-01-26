# Ingestion Service Architecture

This document provides a comprehensive technical overview of the ingestion service architecture, covering the overall system design, data flow patterns, and the integration between the ingestion pipeline and the microservice layer.

## Table of Contents

1. [System Overview](#system-overview)
2. [High-Level Architecture](#high-level-architecture)
3. [Ingestion Pipeline Flow](#ingestion-pipeline-flow)
4. [API Layer](#api-layer)
5. [Orchestration Layer (Prefect)](#orchestration-layer-prefect)
6. [Storage Layer](#storage-layer)
7. [Metadata & Lineage Tracking](#metadata--lineage-tracking)
8. [Microservices Architecture](#microservices-architecture)
9. [Service Discovery & Communication](#service-discovery--communication)
10. [Data Flow Through Microservices](#data-flow-through-microservices)
11. [Artifact Schema & Storage Layout](#artifact-schema--storage-layout)
12. [Vector Database Integration](#vector-database-integration)
13. [Configuration Management](#configuration-management)
14. [Error Handling & Resilience](#error-handling--resilience)
15. [Monitoring & Observability](#monitoring--observability)
16. [Deployment Architecture](#deployment-architecture)
17. [Extension Points](#extension-points)

---

## System Overview

The ingestion service is a **production-ready, modular video processing pipeline** that orchestrates end-to-end multimedia analysis. It combines multiple AI/ML models to transform raw video content into rich, searchable multimedia assets with semantic embeddings.

### Core Capabilities

- **Video Upload & Ingestion** - Accepts video files via REST API, stores originals in object storage
- **Shot Boundary Detection** - Identifies scene changes using TransNetV2
- **Speech Recognition** - Transcribes audio to text with word-level timestamps using Chunkformer
- **Visual Understanding** - Extracts keyframes and generates captions via LLM (Gemini/OpenRouter)
- **Embedding Generation** - Creates semantic vectors for images and text
- **Vector Persistence** - Stores embeddings in Milvus for similarity search
- **Lineage Tracking** - Maintains complete audit trail of all processing stages

### Key Design Principles

1. **Modularity** - Each processing stage is isolated, replaceable, and independently scalable
2. **Idempotency** - Tasks verify artifact existence before reprocessing
3. **Resilience** - Automatic retries, health checks, and fallback mechanisms
4. **Observability** - Comprehensive logging, metrics, and progress tracking
5. **Service Discovery** - Dynamic microservice location via Consul
6. **Artifact Lineage** - Complete traceability from source to derived outputs

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                    CLIENTS                                           │
│                     (Mobile Apps, Web UIs, External APIs)                           │
└───────────────────────────────────┬─────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              API GATEWAY LAYER                                       │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────────────┐  │
│  │  Upload Router  │  │  Health Router  │  │         Management Router           │  │
│  │  POST /uploads  │  │ GET /pipeline_* │  │  GET /status, DELETE /videos/*      │  │
│  └────────┬────────┘  └────────┬────────┘  └──────────────┬──────────────────────┘  │
│           │                    │                           │                         │
│           └────────────────────┼───────────────────────────┘                         │
│                                ▼                                                     │
│                    ┌─────────────────────────────────┐                              │
│                    │      FastAPI Application         │                              │
│                    │      (main.py + lifespan.py)     │                              │
│                    └─────────────┬───────────────────┘                              │
│                                  │                                                   │
│                                  ▼                                                   │
│                    ┌─────────────────────────────────┐                              │
│                    │    Prefect Deployment Trigger    │                              │
│                    │    run_deployment()             │                              │
│                    └─────────────┬───────────────────┘                              │
└──────────────────────────────────┼──────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           ORCHESTRATION LAYER                                        │
│  ┌──────────────────────────────────────────────────────────────────────────────┐   │
│  │                     video_processing_flow                                      │   │
│  │                  (Prefect 3 Async Flow)                                        │   │
│  │                                                                              │   │
│  │  Tasks:                                                                      │   │
│  │  • entry_video_ingestion → Videos                                              │   │
│  │  • autoshot_task → AutoshotArtifacts                                          │   │
│  │  • asr_task → ASRArtifacts                                                    │   │
│  │  • image_processing_task → ImageArtifacts                                     │   │
│  │  • segment_caption_task → SegmentCaptionArtifacts                             │   │
│  │  • image_caption_task → ImageCaptionArtifacts                                 │   │
│  │  • image_embedding_task → ImageEmbeddingArtifacts                             │   │
│  │  • text_image_caption_embedding_task → TextCaptionEmbeddingArtifacts          │   │
│  │  • segment_text_caption_embedding_task → TextCapSegmentEmbedArtifacts         │   │
│  │  • image_embedding_milvus_persist_task                                        │   │
│  │  • text_segment_caption_milvus_persist_task                                   │   │
│  └────────────────────────────────────────────┬──────────────────────────────────┘   │
│                                               │                                       │
└───────────────────────────────────────────────┼───────────────────────────────────────┘
                                                │
         ┌──────────────────────────────────────┼──────────────────────────────────────┐
         │                                      │                                      │
         ▼                                      ▼                                      ▼
┌─────────────────┐            ┌─────────────────┐            ┌─────────────────┐
│     MinIO       │            │   PostgreSQL    │            │     Consul      │
│   (S3 Storage)  │            │  (Metadata DB)  │            │ (Service Disc.) │
│                 │            │                 │            │                 │
│ • videos/       │            │ • artifacts_*   │            │ • Service reg.  │
│ • autoshot/     │            │ • lineage_*     │            │ • Health checks │
│ • asr/          │            │ • Progress      │            │ • Discovery     │
│ • images/       │            │                 │            │                 │
│ • caption/      │            └─────────────────┘            └─────────────────┘
│ • embedding/    │                          │
└─────────────────┘                          │
         │                                   │
         └───────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           MICROSERVICES LAYER                                        │
│                                                                                      │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐   │
│  │  Autoshot   │ │     ASR     │ │     LLM     │ │Image Embed  │ │Text Embed   │   │
│  │  (8001)     │ │  (8002)     │ │  (8004)     │ │  (8000)     │ │  (8003)     │   │
│  │             │ │             │ │             │ │             │ │             │   │
│  │ TransNetV2  │ │ Chunkformer │ │  Gemini/    │ │  OpenCLIP/  │ │ Sentence-   │   │
│  │  (PyTorch)  │ │   (RNN-T)   │ │  OpenRouter │ │   BEiT3     │ │  BERT       │   │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘   │
│                                                                                      │
│  Each service:                                                                      │
│  • FastAPI app with lifespan management                                             │
│  • BaseService abstract class inheritance                                          │
│  • Consul service registration                                                     │
│  • Model registry with @register_model decorator                                   │
│  • Prometheus metrics (/metrics)                                                   │
│  • Health check endpoint (/health)                                                 │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                               │
                                               ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                          VECTOR DATABASE LAYER                                       │
│  ┌─────────────────────────────────────────────────────────────────────────────┐    │
│  │                              Milvus                                          │    │
│  │                                                                             │    │
│  │  Collections:                                                               │    │
│  │  • image_milvus (512-dim visual + 384-dim caption embeddings)              │    │
│  │  • segment_milvus (512-dim segment caption embeddings)                     │    │
│  │                                                                             │    │
│  │  Index Types: HNSW (dense), BM25 (sparse)                                   │    │
│  │  Metrics: COSINE                                                            │    │
│  └─────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Ingestion Pipeline Flow

The ingestion pipeline executes in distinct stages, with parallel execution where dependencies allow:

### Stage 1: Video Ingestion

```
Upload Request → FastAPI → Prefect Deployment → VideoIngestionTask

Input:
  • video_files: list of (video_id, s3_url) tuples
  • user_id: namespace for artifact organization
  • run_id: unique identifier for this processing run

Processing:
  1. Validate video URLs are accessible in MinIO
  2. Create VideoArtifact records in PostgreSQL
  3. Register artifact lineage (source → video)
  4. Report progress: VIDEO_INGEST → 100%

Output:
  • VideoArtifact list containing:
    - artifact_id (UUID)
    - video_name, video_url, video_extension
    - minio_url_path
    - related_video_id (self-reference for root)
    - task_name: "video_ingestion"
```

### Stage 2: Parallel Processing (Autoshot + ASR)

```
                    ┌─────────────────────┐
                    │   VideoArtifacts    │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                                 ▼
┌─────────────────────────┐     ┌─────────────────────────┐
│   Autoshot Processing   │     │     ASR Processing      │
│   (Shot Boundaries)     │     │   (Speech-to-Text)      │
├─────────────────────────┤     ├─────────────────────────┤
│ Model: TransNetV2       │     │ Model: Chunkformer      │
│ Port: 8001              │     │ Port: 8002              │
│ Endpoint: /autoshot/infer│    │ Endpoint: /asr/infer    │
├─────────────────────────┤     ├─────────────────────────┤
│ Input: Video S3 URL     │     │ Input: Video S3 URL     │
│ Output: [start, end]    │     │ Output: text +          │
│         frame pairs     │     │         timestamps      │
├─────────────────────────┤     ├─────────────────────────┤
│ Storage:                │     │ Storage:                │
│ autoshot/{name}.json    │     │ asr/{name}.json         │
├─────────────────────────┤     ├─────────────────────────┤
│ Lineage:                │     │ Lineage:                │
│ Video → Autoshot        │     │ Video → ASR             │
└───────────┬─────────────┘     └───────────┬─────────────┘
            │                             │
            └──────────────┬──────────────┘
                           │
                           ▼
              ┌─────────────────────────────┐
              │  AutoshotArtifact List      │
              │  ASRArtifact List           │
              └──────────────┬──────────────┘
                             │
```

### Stage 3A: Segment Captioning & Embedding

```
                    ┌─────────────────────────────┐
                    │  AutoshotArtifacts +        │
                    │  ASRArtifacts               │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
              ┌─────────────────────────────┐
              │  Segment Caption LLM Task   │
              │  (Parallel: Image Caption)  │
├─────────────────────────────────────────┤
│ Model: Gemini 1.5 Flash / OpenRouter    │
│ Port: 8004                              │
│ Endpoint: /llm/infer                    │
├─────────────────────────────────────────┤
│ Input:                                  │
│   • Segment metadata (start, end)       │
│   • ASR transcript for segment          │
│   • Optional sample frames              │
├─────────────────────────────────────────┤
│ Prompt Template:                        │
│ "Describe what happens in this video    │
│  segment from {start}s to {end}s.       │
│  Audio transcript: {transcript}"        │
├─────────────────────────────────────────┤
│ Output: SegmentCaptionArtifact          │
│   • caption_text                        │
│   • segment_start, segment_end          │
│   • related_video_id                    │
├─────────────────────────────────────────┤
│ Storage: caption/segment/{vid}/{s}_{e}.json
└───────────┬─────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────┐
│  Text Embedding Service (Segment)       │
│  Port: 8003                             │
│  Endpoint: /text-embedding/infer        │
├─────────────────────────────────────────┤
│ Model: Sentence-BERT / mmBERT           │
│ Output Dimension: 512                   │
├─────────────────────────────────────────┤
│ Storage: embedding/caption_segment/     │
│           {vid}/{s}_{e}.npy             │
├─────────────────────────────────────────┤
│ Lineage:                                │
│ Video → ASR → SegmentCaption →          │
│ TextCapSegmentEmbedding                 │
└─────────────────────────────────────────┘
```

### Stage 3B: Image Processing & Embedding

```
                    ┌─────────────────────────────┐
                    │  AutoshotArtifacts          │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
              ┌─────────────────────────────┐
              │  Image Extraction Task      │
├─────────────────────────────────────────┤
│ Extract N frames per segment            │
│ Configurable: num_img_per_segment       │
│ Format: WebP (lossless compression)     │
├─────────────────────────────────────────┤
│ Storage: images/{vid}/{frame_idx}.webp  │
├─────────────────────────────────────────┤
│ Output: ImageArtifact                   │
│   • frame_index                         │
│   • segment_start, segment_end          │
│   • minio_url_path                      │
└───────────┬─────────────────────────────┘
            │
            ▼
    ┌───────┴───────┐
    │               │
    ▼               ▼
┌─────────┐    ┌─────────────────────────┐
│ Image   │    │  Image Embedding        │
│Caption  │    │  Service                │
│ LLM     │    │  Port: 8000             │
│ Task    │    │  Endpoint: /image-      │
│         │    │  embedding/infer        │
├─────────┤    ├─────────────────────────┤
│ Model:  │    │ Model: OpenCLIP/BEiT3   │
│ Gemini  │    │ Output: 512-dim vector  │
├─────────┤    ├─────────────────────────┤
│ Storage:│    │ Storage: embedding/     │
│ caption/│    │ image/{vid}/{idx}.npy   │
│ image/  │    └───────────┬─────────────┘
│ {v}/{i}.│                │
│ json    │                │
└────┬────┘                │
     │                     │
     ▼                     ▼
┌─────────────────────────┐
│  Text Embedding Service │
│  (Image Captions)       │
│  Port: 8003             │
├─────────────────────────┤
│ Model: Sentence-BERT    │
│ Output: 384-dim vector  │
├─────────────────────────┤
│ Storage: embedding/     │
│ image_caption/{v}/{i}.npy
└─────────────────────────┘
```

### Stage 4: Vector Persistence (Optional)

```
                    ┌─────────────────────────────────────┐
                    │  Image Embeddings +                 │
                    │  Text Caption Embeddings +          │
                    │  Segment Caption Embeddings         │
                    └──────────────┬──────────────────────┘
                                   │
           ┌───────────────────────┴───────────────────────┐
           ▼                                               ▼
┌─────────────────────────┐                 ┌─────────────────────────┐
│  Image Milvus Client    │                 │  Segment Milvus Client  │
│  Collection: image_milvus│                │  Collection: segment_   │
│                         │                │  milvus                  │
├─────────────────────────┤                ├─────────────────────────┤
│ Fields:                 │                │ Fields:                 │
│ • id (auto)             │                │ • id (auto)             │
│ • artifact_id           │                │ • artifact_id           │
│ • video_id              │                │ • video_id              │
│ • visual_embedding      │                │ • caption_embedding     │
│   (512-dim, HNSW)       │                │   (512-dim, HNSW)       │
│ • caption_embedding     │                │ • caption_text          │
│   (384-dim, HNSW)       │                │ • segment_start/end     │
│ • caption_text          │                │                         │
│ • frame_index           │                └─────────────────────────┘
│ • segment_start/end     │
└─────────────────────────┘
```

### Complete Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                   CLIENT                                             │
│                          POST /uploads/                                             │
└─────────────────────────────────────┬───────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              FASTAPI (main.py)                                       │
│  1. Validate upload request                                                          │
│  2. Generate run_id (UUID)                                                          │
│  3. Trigger Prefect deployment: run_deployment()                                    │
│  4. Return 202 Accepted with tracking URL                                            │
└─────────────────────────────────────┬───────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                         PREFECT ORCHESTRATION                                        │
│  video_processing_flow(run_id, video_files, user_id)                                │
└─────────────────────────────────────┬───────────────────────────────────────────────┘
                                      │
    ┌─────────────────────────────────┼─────────────────────────────────────────┐
    │                                 │                                         │
    ▼                                 ▼                                         ▼
┌──────────────┐              ┌──────────────┐                          ┌──────────────┐
│   MinIO      │              │  PostgreSQL  │                          │   Consul     │
│   Storage    │              │  Tracker     │                          │  Registry    │
│              │              │              │                          │              │
│ • Upload     │              │ • Create     │                          │ • Discover   │
│   originals  │              │   VideoArtifact                            │   services   │
│ • Store all  │              │ • Track      │                          │ • Health     │
│   artifacts  │              │   lineage    │                          │   checks     │
│ • Return S3  │              │              │                          │              │
│   URLs       │              └──────────────┘                          └──────────────┘
└──────┬───────┘                                                                      │
       │                                                                              │
       │    ┌─────────────────────────────────────────────────────────────────────┐   │
       │    │                      MICROSERVICES INVOCATION                       │   │
       │    │                                                                      │   │
       │    │  ┌───────────────┐   ┌───────────────┐   ┌───────────────┐          │   │
       │    │  │   Autoshot    │   │     ASR       │   │     LLM       │          │   │
       │    │  │   Service     │   │   Service     │   │   Service     │          │   │
       │    │  │   (8001)      │   │   (8002)      │   │   (8004)      │          │   │
       │    │  └───────┬───────┘   └───────┬───────┘   └───────┬───────┘          │   │
       │    │          │                   │                   │                  │   │
       │    │          ▼                   ▼                   ▼                  │   │
       │    │  ┌──────────────────────────────────────────────────────────────┐   │   │
       │    │  │                  BaseServiceClient                           │   │   │
       │    │  │  • Consul discovery → get_healthy_service()                  │   │   │
       │    │  │  • HTTP request with retry → AsyncRetrying()                 │   │   │
       │    │  │  • Response validation → Pydantic model                      │   │   │
       │    │  └──────────────────────────────────────────────────────────────┘   │   │
       │    │                                                                      │   │
       │    └─────────────────────────────────────────────────────────────────────┘   │
       │                                                                              │
       │    ┌─────────────────────────────────────────────────────────────────────┐   │
       │    │                      PROCESSING STAGES                              │   │
       │    │                                                                      │   │
       │    │  Stage 1: Video Ingestion                                           │   │
       │    │  • entry_video_ingestion()                                          │   │
       │    │  • Upload videos to MinIO                                           │   │
       │    │  • Create VideoArtifact records                                     │   │
       │    │                                                                      │   │
       │    │  Stage 2: Parallel Autoshot + ASR                                   │   │
       │    │  • autoshot_task() → AutoshotArtifact[]                             │   │
       │    │  • asr_task() → ASRArtifact[]                                       │   │
       │    │                                                                      │   │
       │    │  Stage 3A: Segment Captions + Embeddings                            │   │
       │    │  • segment_caption_task() → SegmentCaptionArtifact[]                 │   │
       │    │  • segment_text_caption_embedding_task() → TextCapSegmentEmbed[]     │   │
       │    │                                                                      │   │
       │    │  Stage 3B: Image Processing                                         │   │
       │    │  • image_processing_task() → ImageArtifact[]                        │   │
       │    │  • image_caption_task() → ImageCaptionArtifact[]                    │   │
       │    │  • image_embedding_task() → ImageEmbeddingArtifact[]                │   │
       │    │  • text_image_caption_embedding_task() → TextCaptionEmbedding[]      │   │
       │    │                                                                      │   │
       │    │  Stage 4: Milvus Persistence (Optional)                             │   │
       │    │  • image_embedding_milvus_persist_task()                            │   │
       │    │  • text_segment_caption_milvus_persist_task()                       │   │
       │    │                                                                      │   │
       │    └─────────────────────────────────────────────────────────────────────┘   │
       │                                                                              │
       ▼                                                                              │
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                    Milvus                                            │
│  • image_milvus collection (visual + caption embeddings)                            │
│  • segment_milvus collection (segment caption embeddings)                           │
│  • Enable semantic similarity search                                                 │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## API Layer

### Upload Router (`api/upload.py`)

**Endpoint: `POST /uploads/`**

```python
class UploadRequest(BaseModel):
    videos: list[VideoObject]  # (video_id, video_s3_url)
    user_id: str

class VideoObject(BaseModel):
    video_id: str
    video_url: str  # MinIO S3 URL

class UploadResponse(BaseModel):
    run_id: str
    flow_run_id: str
    video_count: int
    video_names: list[str]
    status: str
    message: str
    deployment_name: str
    tracking_url: Optional[str]
```

**Request Example:**
```bash
curl -X POST "http://localhost:8000/uploads/" \
  -H "Content-Type: application/json" \
  -d '{
    "videos": [
      {"video_id": "video_001", "video_url": "s3://videos/test.mp4"},
      {"video_id": "video_002", "video_url": "s3://videos/demo.mp4"}
    ],
    "user_id": "user_123"
  }'
```

**Response Example:**
```json
{
  "run_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "flow_run_id": "xYz-abc-123",
  "video_count": 2,
  "video_names": ["video_001", "video_002"],
  "status": "SCHEDULED",
  "message": "Video processing deployment submitted successfully",
  "deployment_name": "video-processing/primary-gpu",
  "tracking_url": "/api/management/videos/a1b2c3d4-e5f6-7890-abcd-ef1234567890/status"
}
```

### Health Router (`api/health.py`)

**Endpoints:**
- `GET /pipeline_check` - Overall system health
- `GET /pipeline_check/database` - PostgreSQL connectivity
- `GET /pipeline_check/storage` - MinIO bucket listing
- `GET /pipeline_check/consul` - Consul service catalog
- `GET /pipeline_check/services/{service_name}` - Individual microservice health
- `GET /pipeline_check/milvus/{collection}` - Milvus collection status

**Response Structure:**
```json
{
  "status": "healthy",
  "components": {
    "database": {"status": "healthy", "latency_ms": 5},
    "storage": {"status": "healthy", "buckets": ["videos", "video-artifacts"]},
    "consul": {"status": "healthy", "services": 5},
    "milvus": {"status": "healthy", "collections": ["image_milvus", "segment_milvus"]},
    "services": {
      "autoshot": {"status": "healthy", "url": "http://autoshot:8001"},
      "asr": {"status": "healthy", "url": "http://asr:8002"},
      "llm": {"status": "healthy", "url": "http://llm:8004"},
      "image-embedding": {"status": "healthy", "url": "http://image-embedding:8000"},
      "text-embedding": {"status": "healthy", "url": "http://text-embedding:8003"}
    }
  }
}
```

### Management Router (`api/management.py`)

**Endpoints:**
- `GET /management/videos/{video_id}/status` - Processing status
- `DELETE /management/videos/{video_id}` - Cascade delete video + descendants
- `DELETE /management/videos/{video_id}/stages/{artifact_type}` - Delete stage subtree
- `POST /management/videos/batch-delete` - Batch delete by IDs

**Status Response:**
```json
{
  "video_id": "video_001",
  "status": "completed",
  "stages": {
    "VIDEO_INGEST": {"status": "completed", "progress": 100},
    "AUTOSHOT_SEGMENTATION": {"status": "completed", "progress": 100},
    "ASR_TRANSCRIPTION": {"status": "completed", "progress": 100},
    "SEGMENT_CAPTIONING": {"status": "completed", "progress": 100},
    "IMAGE_EXTRACTION": {"status": "completed", "progress": 100},
    "IMAGE_CAPTIONING": {"status": "completed", "progress": 100},
    "IMAGE_EMBEDDING": {"status": "completed", "progress": 100},
    "TEXT_CAP_IMAGE_EMBEDDING": {"status": "completed", "progress": 100},
    "TEXT_CAP_SEGMENT_EMBEDDING": {"status": "completed", "progress": 100},
    "IMAGE_MILVUS": {"status": "completed", "progress": 100},
    "TEXT_CAP_SEGMENT_MILVUS": {"status": "completed", "progress": 100}
  }
}
```

---

## Orchestration Layer (Prefect)

### Flow Definition (`flow/video_processing.py`)

```python
@flow(
    name="complete-video-processing-pipeline",
    description="End-to-end video processing with parallel task execution",
    persist_result=True,
    log_prints=True
)
async def video_processing_flow(
    video_files: list[tuple[str, str]],  # (video_id, s3_url)
    user_id: str,
    run_id: str,
) -> dict[str, Any]:
    """Main Prefect flow for video processing pipeline."""
```

### Task Definitions

| Task Name | Input Type | Output Type | Service | Description |
|-----------|------------|-------------|---------|-------------|
| `entry_video_ingestion` | VideoInput | list[VideoArtifact] | MinIO | Upload videos, create records |
| `autoshot_task` | list[VideoArtifact] | list[AutoshotArtifact] | Autoshot | Shot boundary detection |
| `asr_task` | list[VideoArtifact] | list[ASRArtifact] | ASR | Speech-to-text transcription |
| `image_processing_task` | list[AutoshotArtifact] | list[ImageArtifact] | Local | Extract frames from segments |
| `segment_caption_task` | list[AutoshotArtifact], list[ASRArtifact] | list[SegmentCaptionArtifact] | LLM | Generate segment captions |
| `image_caption_task` | list[ImageArtifact] | list[ImageCaptionArtifact] | LLM | Generate image captions |
| `image_embedding_task` | list[ImageArtifact] | list[ImageEmbeddingArtifact] | Image Embed | Generate image vectors |
| `text_image_caption_embedding_task` | list[ImageCaptionArtifact] | list[TextCaptionEmbeddingArtifact] | Text Embed | Generate caption vectors |
| `segment_text_caption_embedding_task` | list[SegmentCaptionArtifact] | list[TextCapSegmentEmbedArtifact] | Text Embed | Generate segment caption vectors |
| `image_embedding_milvus_persist_task` | tuple | list | Milvus | Persist image embeddings |
| `text_segment_caption_milvus_persist_task` | list | list | Milvus | Persist segment caption embeddings |

### Task Execution Pattern

All tasks follow a consistent pattern:

```python
async def example_task(input_data):
    # Get task instance from AppState
    task_instance = AppState().task_name
    
    # Get client configuration
    client_config = AppState().base_client_config
    
    # Create microservice client
    async with SomeClient(config=client_config) as client:
        # Load model onto device
        await client.load_model(model_name="model_name", device="cuda")
        
        # Preprocess input
        preprocessed = await task_instance.preprocess(input_data)
        
        # Execute with client
        results = []
        async for result in task_instance.execute(preprocessed, client):
            # Postprocess result
            processed = await task_instance.postprocess(result)
            results.append(processed)
        
        # Unload model
        await client.unload_model()
    
    return results
```

### Concurrency Control

```python
# From core/lifespan.py
worker_pool_cmds = [
    ["uv", "run", "prefect", "work-pool", "create", "--type", "process", "local-pool"],
    ["uv", "run", "prefect", "concurrency-limit", "create", "llm-service", "3"],
    ["uv", "run", "prefect", "concurrency-limit", "create", "embedding-service", "3"],
    ["uv", "run", "prefect", "concurrency-limit", "create", "autoshot-task", "1"],
    ["uv", "run", "prefect", "concurrency-limit", "create", "asr-task", "1"],
    ["uv", "run", "prefect", "concurrency-limit", "create", "video-registry", "1"],
]
```

---

## Storage Layer

### MinIO Storage Client (`core/storage.py`)

```python
class StorageClient:
    """MinIO (S3-compatible) client for artifact storage."""
    
    def __init__(self, settings: MinioSettings):
        self.client = Minio(
            endpoint=settings.endpoint,
            access_key=settings.access_key,
            secret_key=settings.secret_key,
            secure=settings.secure,
        )
        self.bucket_videos = "videos"
        self.bucket_artifacts = "video-artifacts"
    
    async def upload_fileobj(
        self, 
        file_obj: IO[bytes], 
        bucket: str, 
        object_key: str
    ) -> str:
        """Upload file-like object to MinIO."""
    
    async def put_json(
        self, 
        bucket: str, 
        object_key: str, 
        data: dict
    ) -> str:
        """Upload JSON data to MinIO."""
    
    async def get_object(
        self, 
        bucket: str, 
        object_key: str
    ) -> bytes:
        """Download object from MinIO."""
    
    async def list_objects(
        self, 
        bucket: str, 
        prefix: str = ""
    ) -> list[str]:
        """List objects with given prefix."""
    
    async def object_exists(
        self, 
        bucket: str, 
        object_key: str
    ) -> bool:
        """Check if object exists."""
```

### Storage Layout

All artifacts follow consistent naming conventions:

| Artifact Type | MinIO Path Pattern | Example |
|--------------|-------------------|---------|
| Video | `videos/{video_name}.{ext}` | `videos/test_video.mp4` |
| Autoshot Segments | `autoshot/{video_name}.json` | `autoshot/test_video.json` |
| ASR Transcripts | `asr/{video_name}.json` | `asr/test_video.json` |
| Images | `images/{video_name}/{frame_index}.webp` | `images/test_video/00012545.webp` |
| Image Captions | `caption/image/{video_name}/{frame_index}.json` | `caption/image/test_video/00012545.json` |
| Segment Captions | `caption/segment/{video_name}/{start}_{end}.json` | `caption/segment/test_video/0_30.json` |
| Image Embeddings | `embedding/image/{video_name}/{frame_index}.npy` | `embedding/image/test_video/00012545.npy` |
| Image Caption Embeddings | `embedding/image_caption/{video_name}/{frame_index}.npy` | `embedding/image_caption/test_video/00012545.npy` |
| Segment Caption Embeddings | `embedding/caption_segment/{video_name}/{start}_{end}.npy` | `embedding/caption_segment/test_video/0_30.npy` |

---

## Metadata & Lineage Tracking

### Database Schema

**Table: `artifacts_application`**

```sql
CREATE TABLE artifacts_application (
    id UUID PRIMARY KEY,
    artifact_id VARCHAR(255) NOT NULL,
    artifact_type VARCHAR(100) NOT NULL,
    minio_url TEXT NOT NULL,
    minio_url_path TEXT NOT NULL,
    parent_artifact_id UUID REFERENCES artifacts_application(id),
    task_name VARCHAR(255) NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for efficient queries
CREATE INDEX idx_artifacts_video_id ON artifacts_application(artifact_id);
CREATE INDEX idx_artifacts_type ON artifacts_application(artifact_type);
CREATE INDEX idx_artifacts_parent ON artifacts_application(parent_artifact_id);
```

**Table: `artifact_lineage_application`**

```sql
CREATE TABLE artifact_lineage_application (
    id UUID PRIMARY KEY,
    parent_artifact_id UUID NOT NULL REFERENCES artifacts_application(id),
    child_artifact_id UUID NOT NULL REFERENCES artifacts_application(id),
    transformation_type VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    
    UNIQUE(parent_artifact_id, child_artifact_id)
);

-- Index for ancestor queries
CREATE INDEX idx_lineage_parent ON artifact_lineage_application(parent_artifact_id);
CREATE INDEX idx_lineage_child ON artifact_lineage_application(child_artifact_id);
```

### Artifact Tracker (`core/pipeline/tracker.py`)

```python
class ArtifactTracker:
    """Manages artifact persistence and lineage tracking."""
    
    def __init__(self, database_url: str):
        self.engine = create_async_engine(database_url)
        self.session_factory = async_sessionmaker(self.engine)
    
    async def initialize(self):
        """Create tables if they don't exist."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    
    async def save_artifact(self, artifact: ArtifactSchema) -> UUID:
        """Save artifact and return its ID."""
    
    async def get_artifact(self, artifact_id: str) -> Optional[ArtifactSchema]:
        """Retrieve artifact by ID."""
    
    async def get_descendants(
        self, 
        artifact_id: UUID, 
        depth: Optional[int] = None
    ) -> list[ArtifactSchema]:
        """Get all descendant artifacts."""
    
    async def get_ancestors(self, artifact_id: UUID) -> list[ArtifactSchema]:
        """Get all ancestor artifacts."""
```

### Artifact Visitor Pattern (`core/artifact/persist.py`)

```python
class ArtifactPersistentVisitor:
    """Visits artifacts and persists them to storage and database."""
    
    def __init__(self, minio_client: StorageClient, tracker: ArtifactTracker):
        self.minio_client = minio_client
        self.tracker = tracker
    
    def visit_video(self, artifact: VideoArtifact, data: VideoUploadData):
        """Persist video artifact."""
    
    def visit_autoshot(self, artifact: AutoshotArtifact, data: list[Shot]):
        """Persist autoshot segments."""
    
    def visit_asr(self, artifact: ASRArtifact, data: ASRResult):
        """Persist ASR transcript."""
    
    def visit_image(self, artifact: ImageArtifact, image_bytes: bytes):
        """Persist extracted image."""
    
    def visit_segment_caption(
        self, 
        artifact: SegmentCaptionArtifact, 
        caption_data: SegmentCaptionData
    ):
        """Persist segment caption."""
    
    def visit_image_caption(
        self, 
        artifact: ImageCaptionArtifact, 
        caption_data: ImageCaptionData
    ):
        """Persist image caption."""
    
    def visit_image_embedding(
        self, 
        artifact: ImageEmbeddingArtifact, 
        embedding: np.ndarray
    ):
        """Persist image embedding."""
    
    def visit_segment_caption_embedding(
        self, 
        artifact: TextCapSegmentEmbedArtifact, 
        embedding: np.ndarray
    ):
        """Persist segment caption embedding."""
```

---

## Microservices Architecture

### Shared Architecture (`prefect_agent/shared/`)

All microservices inherit from shared base classes, ensuring consistency:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        SHARED CORE LAYER                                     │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     BaseService<InputT, OutputT>                     │   │
│  │                                                                     │   │
│  │  Attributes:                                                         │   │
│  │  • service_config: ServiceConfig                                     │   │
│  │  • log_config: LogConfig                                             │   │
│  │  • metrics: ServiceMetrics                                           │   │
│  │  • loaded_model: Optional[BaseModelHandler]                          │   │
│  │  • loaded_model_info: Optional[ModelInfo]                            │   │
│  │                                                                     │   │
│  │  Methods:                                                            │   │
│  │  • async load_model(model_name, device) → ModelInfo                 │   │
│  │  • async unload_model()                                              │   │
│  │  • async infer(input_data: InputT) → OutputT                        │   │
│  │  • async get_status() → dict                                         │   │
│  │  • get_available_models() → list[str]                               │   │
│  │  • update_system_metrics()                                           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    ▲                                         │
│                                    │ inherits                                │
│                                    │                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    SHARED COMPONENTS                                 │   │
│  │                                                                     │   │
│  │  ┌──────────────┐  ┌────────────────┐  ┌────────────────────────┐   │   │
│  │  │ Config       │  │ Logger         │  │ Metrics                │   │   │
│  │  │ (pydantic)   │  │ (Loguru)       │  │ (Prometheus)           │   │   │
│  │  └──────────────┘  └────────────────┘  └────────────────────────┘   │   │
│  │                                                                     │   │
│  │  ┌──────────────┐  ┌────────────────┐  ┌────────────────────────┐   │   │
│  │  │ ServiceReg.  │  │ Storage        │  │ Retry                  │   │   │
│  │  │ (Consul)     │  │ (MinIO)        │  │ (tenacity)             │   │   │
│  │  └──────────────┘  └────────────────┘  └────────────────────────┘   │   │
│  │                                                                     │   │
│  │  ┌──────────────┐  ┌────────────────┐  ┌────────────────────────┐   │   │
│  │  │ Schema       │  │ Registry       │  │ Monitor                │   │   │
│  │  │ (Pydantic)   │  │ (Model hooks)  │  │ (System stats)         │   │   │
│  │  └──────────────┘  └────────────────┘  └────────────────────────┘   │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### BaseService Abstract Class

```python
# From prefect_agent/shared/service.py

class BaseService(Generic[InputT, OutputT], ABC):
    """Abstract base class providing common service functionality."""
    
    def __init__(
        self, 
        service_config: ServiceConfig, 
        log_config: LogConfig
    ) -> None:
        # Initialize logging
        setup_service_logger(
            service_name=service_config.service_name,
            service_version=service_config.service_version,
            log_level=log_config.log_level,
        )
        
        # Initialize metrics
        self.metrics = ServiceMetrics(
            service_name=service_config.service_name
        )
        
        # Model state
        self.loaded_model: Optional[BaseModelHandler[InputT, OutputT]] = None
        self.loaded_model_info: Optional[ModelInfo] = None
        
        # Consul registration
        self._consul_registry = ConsulServiceRegistry(
            host=consule_conf.host,
            port=consule_conf.port,
        )
    
    async def load_model(
        self, 
        model_name: str, 
        device: Literal["cuda", "cpu"]
    ) -> ModelInfo:
        """Load model onto specified device."""
        # 1. Validate model availability
        available = self.get_available_models()
        if model_name not in available:
            raise HTTPException(400, f"Model {model_name} not available")
        
        # 2. Check GPU availability
        if device == "cuda" and not is_gpu_available():
            if self.service_config.cpu_fallback:
                device = "cpu"
            else:
                raise HTTPException(400, "No GPU available")
        
        # 3. Get handler from registry
        handler = get_model_handler(model_name, self.service_config)
        
        # 4. Load model
        await handler.load_model_impl(device)
        
        # 5. Store reference
        self.loaded_model = handler
        self.loaded_model_info = handler.get_model_info()
        
        # 6. Update metrics
        self.update_system_metrics()
        
        return self.loaded_model_info
    
    async def infer(self, input_data: InputT) -> OutputT:
        """Execute model inference."""
        if self.loaded_model is None:
            raise HTTPException(400, "No model loaded")
        
        try:
            start_time = time.time()
            
            # Preprocess
            preprocessed = await self.loaded_model.preprocess_input(input_data)
            
            # Inference
            result = await self.loaded_model.run_inference(preprocessed)
            
            # Postprocess
            output = await self.loaded_model.postprocess_output(
                result, 
                input_data
            )
            
            # Metrics
            duration = time.time() - start_time
            self.metrics.observe_request_duration("infer", duration)
            self.metrics.track_request("infer", "success")
            
            return output
        
        except Exception as exc:
            self.metrics.track_error("infer", type(exc).__name__)
            raise HTTPException(500, f"Inference failed: {exc}")
    
    async def unload_model(self) -> dict:
        """Release model resources."""
        if self.loaded_model:
            await self.loaded_model.unload_model_impl()
            self.loaded_model = None
            self.loaded_model_info = None
        return {"status": "unloaded"}
```

### BaseModelHandler Interface

```python
# From prefect_agent/shared/registry.py

class BaseModelHandler(Generic[InputT, OutputT], ABC):
    """Abstract interface for model implementations."""
    
    def __init__(self, model_name: str, config: ServiceConfig) -> None:
        self.model_name = model_name
        self.config = config
    
    @abstractmethod
    async def load_model_impl(
        self, 
        device: Literal["cpu", "cuda"]
    ) -> None:
        """Load model weights into memory."""
    
    @abstractmethod
    async def unload_model_impl(self) -> None:
        """Release model resources."""
    
    @abstractmethod
    def get_model_info(self) -> ModelInfo:
        """Expose metadata for the loaded model."""
    
    @abstractmethod
    async def preprocess_input(self, input_data: InputT) -> Any:
        """Transform request payload into model-ready format."""
    
    @abstractmethod
    async def run_inference(self, preprocessed_data: Any) -> Any:
        """Perform forward pass and return raw outputs."""
    
    @abstractmethod
    async def postprocess_output(
        self, 
        output_data: Any, 
        original_input_data: InputT
    ) -> OutputT:
        """Convert raw outputs into service response schema."""
```

### Model Registry Pattern

```python
# From prefect_agent/shared/registry.py

MODEL_REGISTRY: Dict[str, Type[BaseModelHandler[Any, Any]]] = {}

def register_model(name: str):
    """Decorator that self-registers model handlers."""
    def decorator(cls: Type[BaseModelHandler[Any, Any]]) -> Type[BaseModelHandler[Any, Any]]:
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator

def list_models() -> list[str]:
    """List all registered models."""
    return sorted(MODEL_REGISTRY.keys())

def get_model_handler(
    name: str, 
    config: ServiceConfig
) -> BaseModelHandler[Any, Any]:
    """Get model handler instance by name."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Model {name} not registered")
    return MODEL_REGISTRY[name](name, config)
```

### Service Implementation Examples

**Autoshot Service (`prefect_agent/service_autoshot/`):**

```python
# service_autoshot/core/service.py
from shared.service import BaseService
from service_autoshot.core.config import AutoshotConfig
from service_autoshot.schema import AutoShotRequest, AutoShotResponse

class AutoshotService(BaseService[AutoShotRequest, AutoShotResponse]):
    """Shot boundary detection service using TransNetV2."""
    
    def __init__(self, service_config: AutoshotConfig, log_config: LogConfig) -> None:
        super().__init__(service_config, log_config)
    
    def get_available_models(self) -> list[str]:
        registered = super().get_available_models()
        return [model for model in registered if model == "autoshot"]
```

```python
# service_autoshot/model/registry.py
from shared.registry import BaseModelHandler, register_model
from shared.schema import ModelInfo
from service_autoshot.core.config import AutoshotConfig
from service_autoshot.model.autoshot import AutoShot
from service_autoshot.schema import AutoShotRequest, AutoShotResponse

@register_model("autoshot")
class AutoshotModelHandler(BaseModelHandler[AutoShotRequest, AutoShotResponse]):
    """Model handler for TransNetV2 shot detection."""
    
    async def load_model_impl(self, device: Literal["cpu", "cuda"]) -> None:
        model_path = self.config.autoshot_model_path
        self._model = AutoShot(
            pretrained_path=model_path, 
            device=device
        )
        self._device = device
    
    async def unload_model_impl(self) -> None:
        if self._model is not None:
            del self._model
        self._model = None
    
    def get_model_info(self) -> ModelInfo:
        return ModelInfo(
            model_name=self.model_name,
            model_type="transnet_v2",
            device=self._device
        )
    
    async def preprocess_input(self, input_data: AutoShotRequest) -> str:
        # Fetch video from S3
        video_path = await fetch_object_from_s3(
            input_data.s3_minio_url, 
            storage=self._storage_client,
            suffix='.mp4'
        )
        return video_path
    
    async def run_inference(self, preprocessed_data: str) -> list[list[int]]:
        return self._model.process_video(preprocessed_data)
    
    async def postprocess_output(
        self, 
        output_data: list[list[int]], 
        original_input_data: AutoShotRequest
    ) -> AutoShotResponse:
        scenes = [tuple(s) for s in output_data]
        return AutoShotResponse(
            scenes=scenes,
            total_scenes=len(scenes),
            status="success",
        )
```

**ASR Service (`prefect_agent/service_asr/`):**

```python
# service_asr/model/asr_core.py
from shared.registry import BaseModelHandler, register_model
from service_asr.core.config import ASRServiceConfig
from service_asr.schema import ASRInferenceRequest, ASRInferenceResponse

@register_model("chunkformer")
class ASRModelHandler(BaseModelHandler[ASRInferenceRequest, ASRInferenceResponse]):
    """Chunkformer RNN-T ASR model handler."""
    
    async def load_model_impl(self, device: Literal["cpu", "cuda"]) -> None:
        # Load Chunkformer checkpoint
        self._processor = AutoProcessor.from_pretrained(
            self.config.model_name
        )
        self._model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.config.model_name
        ).to(device)
        self._device = device
    
    async def run_inference(self, audio_path: str) -> dict:
        # Extract audio and run ASR
        audio = extract_audio(audio_path, sample_rate=16000)
        inputs = self._processor(
            audio, 
            sampling_rate=16000, 
            return_tensors="pt"
        ).to(self._device)
        
        with torch.no_grad():
            generated_ids = self._model.generate(
                inputs.input_features,
                max_new_tokens=256,
            )
        
        transcript = self._processor.batch_decode(
            generated_ids, 
            skip_special_tokens=True
        )[0]
        
        return {
            "text": transcript,
            "confidence": 0.95,
        }
```

**LLM Service (`prefect_agent/service_llm/`):**

```python
# service_llm/model/gemini.py
from shared.registry import BaseModelHandler, register_model
from service_llm.core.config import LLMServiceConfig
from service_llm.schema import LLMRequest, LLMResponse

@register_model("gemini_api")
class GeminiAPIHandler(BaseModelHandler[LLMRequest, LLMResponse]):
    """Google Gemini API handler."""
    
    def __init__(self, model_name: str, config: LLMServiceConfig):
        super().__init__(model_name, config)
        self._client = None
    
    async def load_model_impl(self, device: Literal["cpu", "cuda"]) -> None:
        import google.generativeai as genai
        genai.configure(api_key=self.config.gemini_api_key)
        self._model = genai.GenerativeModel(
            model_name=self.config.gemini_model_name
        )
    
    async def run_inference(self, prompt_data: dict) -> LLMResponse:
        response = self._model.generate_content(
            prompt_data["prompt"],
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=prompt_data.get("max_tokens", 1024),
                temperature=prompt_data.get("temperature", 0.7),
            )
        )
        
        return LLMResponse(
            text=response.text,
            model=self.config.gemini_model_name,
            usage=TokenUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            )
        )
```

---

## Service Discovery & Communication

### Consul Integration (`core/pipeline/service_registry.py`)

```python
from consul.aio import Consul
from dataclasses import dataclass

@dataclass
class ServiceInfo:
    service_id: str
    service_name: str
    address: str
    port: int
    tags: list[str]
    meta: dict[str, str]
    health_status: str = "unknown"

class ConsulServiceRegistry:
    """Consul service discovery integration."""
    
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8500,
        datacenter: str = "dc1",
        check_interval: str = "10s",
        check_timeout: str = "5s",
    ):
        self.consul = Consul(host=host, port=port)
        self.datacenter = datacenter
        self.check_interval = check_interval
        self.check_timeout = check_timeout
    
    async def register_service(
        self,
        service_name: str,
        address: str,
        port: int,
        service_id: str | None = None,
        tags: list[str] | None = None,
        health_check_url: str | None = None,
        health_check_tcp: str | None = None
    ):
        """Register service with Consul."""
        service_id = service_id or f"{service_name}-{address}-{port}"
        
        check = {}
        if health_check_url:
            check = {
                "http": health_check_url,
                "interval": self.check_interval,
                "timeout": self.check_timeout,
                "DeregisterCriticalServiceAfter": "30s",
            }
        elif health_check_tcp:
            check = {
                "tcp": health_check_tcp,
                "interval": self.check_interval,
                "timeout": self.check_timeout,
                "DeregisterCriticalServiceAfter": "30s",
            }
        
        await self.consul.agent.service.register(
            name=service_name,
            service_id=service_id,
            address=address,
            port=port,
            tags=tags or [],
            check=check or None
        )
    
    async def discover_service(self, service_name: str) -> list[ServiceInfo]:
        """Discover all instances of a service."""
        index, nodes = await self.consul.catalog.service(
            service_name, 
            dc=self.datacenter
        )
        
        services = []
        for node in nodes:
            health_status = await self._get_health_status(node["ServiceID"])
            services.append(
                ServiceInfo(
                    service_id=node["ServiceID"],
                    service_name=node["ServiceName"],
                    address=node["ServiceAddress"] or node["Address"],
                    port=node["ServicePort"],
                    tags=node.get("ServiceTags", []),
                    meta=node.get("ServiceMeta", {}),
                    health_status=health_status,
                )
            )
        return services
    
    async def get_healthy_service(self, service_name: str) -> Optional[ServiceInfo]:
        """Get first healthy service instance."""
        services = await self.discover_service(service_name)
        healthy = [s for s in services if s.health_status == "passing"]
        return healthy[0] if healthy else None
    
    async def _get_health_status(self, service_id: str) -> str:
        """Get health status of a service instance."""
        _, checks = await self.consul.health.checks(service_id)
        statuses = [c["Status"] for c in checks]
        
        if "critical" in statuses:
            return "critical"
        elif "warning" in statuses:
            return "warning"
        elif "passing" in statuses:
            return "passing"
        return "unknown"
```

### Base Service Client (`core/clients/base.py`)

```python
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Literal
from urllib.parse import urljoin
import httpx
from tenacity import (
    AsyncRetrying,
    RetryError,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

class BaseServiceClient(ABC, Generic[TRequest, TResponse]):
    """
    Base client for microservices with Consul discovery, retries, and timeouts.
    
    Features:
    - Service discovery via Consul
    - Automatic retry with exponential backoff
    - Configurable timeouts
    - Response validation
    """
    
    def __init__(self, config: ClientConfig):
        self.config = config
        self.http_client: httpx.AsyncClient | None = None
        self.consul: ConsulServiceRegistry | None = None
    
    @property
    @abstractmethod
    def service_name(self) -> str:
        """Service name registered in Consul."""
        raise NotImplementedError
    
    @property
    @abstractmethod
    def inference_endpoint(self) -> str:
        """Inference endpoint path."""
        raise NotImplementedError
    
    @property
    @abstractmethod
    def load_endpoint(self) -> str:
        """Model loading endpoint path."""
        raise NotImplementedError
    
    @property
    @abstractmethod
    def unload_endpoint(self) -> str:
        """Model unloading endpoint path."""
        raise NotImplementedError
    
    async def __aenter__(self) -> BaseServiceClient:
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    async def connect(self) -> None:
        """Initialize HTTP client and Consul connection."""
        self.http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.config.timeout_seconds),
            follow_redirects=True
        )
        
        self.consul = ConsulServiceRegistry(
            host=self.config.consul_host,
            port=self.config.consul_port,
        )
    
    async def close(self) -> None:
        """Close HTTP client."""
        if self.http_client:
            await self.http_client.aclose()
    
    async def get_service_url(self) -> str | None:
        """
        Discover service URL via Consul.
        
        Returns:
            Service URL (e.g., "http://autoshot:8001") or None if unavailable.
        """
        if self.consul:
            try:
                service_info = await self.consul.get_healthy_service(
                    self.service_name
                )
                if service_info:
                    url = f"http://{service_info.address}:{service_info.port}"
                    return url
                
                logger.warning(
                    f"No healthy service found: {self.service_name}"
                )
            
            except Exception as e:
                logger.error(
                    f"Consul discovery failed for {self.service_name}",
                    error=str(e)
                )
                raise ServiceUnavailableError(
                    f"No healthy {self.service_name} service available"
                )
    
    async def make_request(
        self,
        method: str,
        endpoint: str,
        request_data: BaseModel | None = None,
        **kwargs
    ):
        """
        Make HTTP request with retry logic.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            request_data: Optional request body
            **kwargs: Additional httpx request kwargs
        
        Returns:
            Parsed JSON response
        
        Raises:
            ClientError: After max retries exhausted
        """
        if self.http_client is None:
            raise ClientError("Client not connected")
        
        async def _attempt_request():
            base_url = await self.get_service_url()
            url = urljoin(base_url, endpoint)
            
            request_kwargs = {**kwargs}
            if request_data:
                request_kwargs['json'] = request_data.model_dump(mode='json')
            
            response = await self.http_client.request(
                method, 
                url, 
                **request_kwargs
            )
            response.raise_for_status()
            return response.json()
        
        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(self.config.max_retries),
                wait=wait_exponential(
                    min=self.config.retry_min_wait,
                    max=self.config.retry_max_wait
                ),
                retry=retry_if_exception_type(Exception),
                reraise=True
            ):
                with attempt:
                    return await _attempt_request()
        
        except RetryError as e:
            logger.exception(
                f"{self.service_name} request failed after retries",
                error=str(e),
                endpoint=endpoint
            )
            raise ClientError(
                f"Request to {self.service_name} failed after "
                f"{self.config.max_retries} retries"
            )
    
    async def invoke(self, request: TRequest) -> TResponse:
        """
        Execute inference request.
        
        Args:
            request: Inference request object
        
        Returns:
            Inference response object
        """
        response = await self.make_request(
            method="POST",
            endpoint=self.inference_endpoint,
            request_data=request,
        )
        
        if response is None:
            raise ClientError(
                f"Received empty response from {self.service_name}"
            )
        
        return response
    
    async def load_model(
        self,
        model_name: str,
        device: Literal['cuda', 'cpu'] = "cuda"
    ):
        """Load model onto specified device."""
        request = LoadModelRequest(model_name=model_name, device=device)
        
        response = await self.make_request(
            method="POST",
            endpoint=self.load_endpoint,
            request_data=request,
        )
        
        if response is None:
            raise ClientError(f"Failed to load model {model_name}")
        
        logger.info(
            f"{self.service_name}_model_loaded",
            model_name=model_name,
            device=device
        )
        
        return response
    
    async def unload_model(self, cleanup_memory: bool = True) -> dict[str, str]:
        """Unload model and release resources."""
        request = UnloadModelRequest(cleanup_memory=cleanup_memory)
        
        if self.http_client is None:
            raise ClientError("Client not connected")
        
        base_url = await self.get_service_url()
        url = urljoin(base_url, self.unload_endpoint)
        
        response = await self.http_client.post(
            url,
            json=request.model_dump(mode='json')
        )
        response.raise_for_status()
        
        result = response.json()
        logger.info(f"{self.service_name}_model_unloaded")
        
        return result
```

### Service-Specific Clients

**Autoshot Client (`core/clients/autoshot_client.py`):**

```python
from core.clients.base import BaseServiceClient, ClientConfig
from prefect_agent.service_autoshot.schema import AutoShotRequest, AutoShotResponse

class AutoshotClient(BaseServiceClient[AutoShotRequest, AutoShotResponse]):
    """Client for Autoshot microservice."""
    
    @property
    def service_name(self) -> str:
        # Must match service name registered in Consul
        return "service-autoshot"
    
    @property
    def inference_endpoint(self) -> str:
        return '/autoshot/infer'
    
    @property
    def load_endpoint(self) -> str:
        return '/autoshot/load'
    
    @property
    def unload_endpoint(self) -> str:
        return '/autoshot/unload'
    
    @property
    def models_endpoint(self) -> str:
        return '/autoshot/models'
    
    @property
    def status_endpoint(self) -> str:
        return '/autoshot/status'
```

**ASR Client (`core/clients/asr_client.py`):**

```python
from core.clients.base import BaseServiceClient, ClientConfig
from prefect_agent.service_asr.schema import ASRInferenceRequest, ASRInferenceResponse

class ASRClient(BaseServiceClient[ASRInferenceRequest, ASRInferenceResponse]):
    """Client for ASR microservice."""
    
    @property
    def service_name(self) -> str:
        return "service-asr"
    
    @property
    def inference_endpoint(self) -> str:
        return '/asr/infer'
    
    @property
    def load_endpoint(self) -> str:
        return '/asr/load'
    
    @property
    def unload_endpoint(self) -> str:
        return '/asr/unload'
    
    @property
    def models_endpoint(self) -> str:
        return '/asr/models'
    
    @property
    def status_endpoint(self) -> str:
        return '/asr/status'
```

**LLM Client (`core/clients/llm_client.py`):**

```python
from core.clients.base import BaseServiceClient, ClientConfig
from prefect_agent.service_llm.schema import LLMRequest, LLMResponse

class LLMClient(BaseServiceClient[LLMRequest, LLMResponse]):
    """Client for LLM microservice."""
    
    @property
    def service_name(self) -> str:
        return "service-llm"
    
    @property
    def inference_endpoint(self) -> str:
        return '/llm/infer'
    
    @property
    def load_endpoint(self) -> str:
        return '/llm/load'
    
    @property
    def unload_endpoint(self) -> str:
        return '/llm/unload'
    
    @property
    def models_endpoint(self) -> str:
        return '/llm/models'
    
    @property
    def status_endpoint(self) -> str:
        return '/llm/status'
```

---

## Data Flow Through Microservices

### Complete Request Flow

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                         PREFECT TASK (e.g., autoshot_task)                          │
│                                                                                     │
│  async def autoshot_task(videos: list[VideoArtifact]):                             │
│      client_config = AppState().base_client_config                                  │
│                                                                                     │
│      async with AutoshotClient(config=client_config) as client:                     │
│          await client.load_model(model_name="autoshot", device="cuda")              │
│                                                                                     │
│          for video in videos:                                                       │
│              request = AutoShotRequest(s3_minio_url=video.minio_url)                │
│              response = await client.invoke(request)                                │
│                                                                                     │
│          await client.unload_model()                                                │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                         BaseServiceClient.make_request()                            │
│                                                                                     │
│  1. get_service_url() → Consul discovery                                             │
│     • consul.get_healthy_service("service-autoshot")                               │
│     • Returns: http://autoshot:8001                                                 │
│                                                                                     │
│  2. _attempt_request() → HTTP POST                                                  │
│     • URL: http://autoshot:8001/autoshot/infer                                     │
│     • Body: {"s3_minio_url": "s3://videos/test.mp4"}                               │
│                                                                                     │
│  3. Retry with exponential backoff                                                  │
│     • max_retries: 3                                                               │
│     • wait: exponential(min=1s, max=10s)                                            │
│                                                                                     │
│  4. Response validation                                                             │
│     • Parse JSON response                                                          │
│     • Validate against AutoShotResponse schema                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                         MICROSERVICE (Autoshot Service)                             │
│                                                                                     │
│  main.py:                                                                         │
│  app = FastAPI()                                                                   │
│  app.include_router(autoshot_router)                                               │
│                                                                                     │
│  @app.post("/autoshot/infer")                                                      │
│  async def infer(request: AutoShotRequest):                                         │
│      return await autoshot_service.infer(request)                                   │
│                                                                                     │
│  autoshot_service.infer():                                                         │
│      1. Preprocess: fetch video from S3                                            │
│      2. Run inference: TransNetV2.process_video()                                  │
│      3. Postprocess: format scenes response                                        │
│      4. Return: AutoShotResponse                                                   │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                         BaseService.infer()                                         │
│                                                                                     │
│  async def infer(self, input_data: InputT) -> OutputT:                            │
│      if self.loaded_model is None:                                                  │
│          raise HTTPException(400, "No model loaded")                                │
│                                                                                     │
│      try:                                                                          │
│          start_time = time.time()                                                   │
│                                                                                     │
│          # Preprocess                                                               │
│          preprocessed = await self.loaded_model.preprocess_input(input_data)       │
│                                                                                     │
│          # Inference                                                                │
│          result = await self.loaded_model.run_inference(preprocessed)              │
│                                                                                     │
│          # Postprocess                                                              │
│          output = await self.loaded_model.postprocess_output(result, input_data)   │
│                                                                                     │
│          # Metrics                                                                  │
│          duration = time.time() - start_time                                        │
│          self.metrics.observe_request_duration("infer", duration)                  │
│          self.metrics.track_request("infer", "success")                            │
│                                                                                     │
│          return output                                                              │
│                                                                                     │
│      except Exception as exc:                                                       │
│          self.metrics.track_error("infer", type(exc).__name__)                      │
│          raise HTTPException(500, f"Inference failed: {exc}")                       │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                    AutoshotModelHandler (TransNetV2)                                │
│                                                                                     │
│  async def preprocess_input(self, input_data: AutoShotRequest) -> str:            │
│      # Download video from S3                                                       │
│      video_path = await fetch_object_from_s3(                                      │
│          input_data.s3_minio_url,                                                  │
│          storage=self._storage_client,                                             │
│          suffix='.mp4'                                                             │
│      )                                                                              │
│      return video_path                                                             │
│                                                                                     │
│  async def run_inference(self, video_path: str) -> list[list[int]]:               │
│      # TransNetV2 prediction                                                        │
│      predictions = self._model.process_video(video_path)                           │
│      return predictions                                                            │
│                                                                                     │
│  async def postprocess_output(                                                     │
│      self,                                                                         │
│      output_data: list[list[int]],                                                 │
│      original_input_data: AutoShotRequest                                          │
│  ) -> AutoShotResponse:                                                            │
│      scenes = [tuple(s) for s in output_data]                                      │
│      return AutoShotResponse(                                                      │
│          scenes=scenes,                                                            │
│          total_scenes=len(scenes),                                                 │
│          status="success",                                                         │
│      )                                                                              │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Artifact Schema & Storage Layout

### Artifact Class Hierarchy (`core/artifact/schema.py`)

```python
from abc import ABC
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class BaseArtifact(BaseModel, ABC):
    """Abstract base class for all artifacts."""
    
    artifact_id: str = Field(..., description="Unique artifact identifier")
    minio_url: str = Field(..., description="Full MinIO URL")
    minio_url_path: str = Field(..., description="MinIO object key")
    related_video_id: str = Field(..., description="Parent video ID")
    task_name: str = Field(..., description="Task that created this artifact")
    created_at: datetime = Field(default_factory=datetime.now)
    
    @property
    @abstractmethod
    def object_key(self) -> str:
        """Generate MinIO object key."""
        raise NotImplementedError
    
    def accept_upload(self, visitor: ArtifactPersistentVisitor, data: Any):
        """Accept visitor for persistence."""
        raise NotImplementedError
    
    def accept_check_exist(self, visitor: ArtifactPersistentVisitor) -> bool:
        """Check if artifact already exists."""
        raise NotImplementedError

class VideoArtifact(BaseArtifact):
    """Original video artifact."""
    video_name: str
    video_extension: str
    file_size: Optional[int] = None
    
    @property
    def object_key(self) -> str:
        return f"videos/{self.video_name}.{self.video_extension}"

class AutoshotArtifact(BaseArtifact):
    """Shot boundary detection artifact."""
    segment_count: int
    
    @property
    def object_key(self) -> str:
        return f"autoshot/{self.video_name}.json"

class ASRArtifact(BaseArtifact):
    """ASR transcription artifact."""
    duration_seconds: float
    language: Optional[str] = None
    
    @property
    def object_key(self) -> str:
        return f"asr/{self.video_name}.json"

class ImageArtifact(BaseArtifact):
    """Extracted frame artifact."""
    frame_index: int
    segment_start: int
    segment_end: int
    image_format: str = "webp"
    
    @property
    def object_key(self) -> str:
        return f"images/{self.video_name}/{self.frame_index}.webp"

class SegmentCaptionArtifact(BaseArtifact):
    """Segment caption artifact."""
    caption_text: str
    segment_start: int
    segment_end: int
    
    @property
    def object_key(self) -> str:
        return f"caption/segment/{self.video_name}/{self.segment_start}_{self.segment_end}.json"

class ImageCaptionArtifact(BaseArtifact):
    """Image caption artifact."""
    caption_text: str
    frame_index: int
    
    @property
    def object_key(self) -> str:
        return f"caption/image/{self.video_name}/{self.frame_index}.json"

class ImageEmbeddingArtifact(BaseArtifact):
    """Image embedding artifact."""
    embedding_dimension: int
    frame_index: int
    
    @property
    def object_key(self) -> str:
        return f"embedding/image/{self.video_name}/{self.frame_index}.npy"

class TextCaptionEmbeddingArtifact(BaseArtifact):
    """Image caption text embedding artifact."""
    embedding_dimension: int
    frame_index: int
    
    @property
    def object_key(self) -> str:
        return f"embedding/image_caption/{self.video_name}/{self.frame_index}.npy"

class TextCapSegmentEmbedArtifact(BaseArtifact):
    """Segment caption text embedding artifact."""
    embedding_dimension: int
    segment_start: int
    segment_end: int
    
    @property
    def object_key(self) -> str:
        return f"embedding/caption_segment/{self.video_name}/{self.segment_start}_{self.segment_end}.npy"
```

---

## Vector Database Integration

### Milvus Collections

**Image Collection Schema:**

```python
# From core/clients/milvus_client.py

class ImageMilvusClient(BaseMilvusClient):
    """Milvus client for image embeddings."""
    
    @property
    def visual_embedding_field(self) -> str:
        return "visual_embedding"
    
    @property
    def caption_embedding_field(self) -> str:
        return "caption_embedding"
    
    @property
    def caption_sparse_embedding_field(self) -> str:
        return "caption_sparse_embedding"
    
    @property
    def caption_text_field(self) -> str:
        return "caption_text"
    
    def get_schema(self) -> CollectionSchema:
        fields = [
            FieldSchema(
                name="id",
                dtype=DataType.VARCHAR,
                is_primary=True,
                auto_id=False,
                max_length=36,
            ),
            FieldSchema(
                name="artifact_id",
                dtype=DataType.VARCHAR,
                max_length=36,
            ),
            FieldSchema(
                name="video_id",
                dtype=DataType.VARCHAR,
                max_length=36,
            ),
            FieldSchema(
                name="visual_embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=512,  # OpenCLIP/BEiT3 output dimension
            ),
            FieldSchema(
                name="caption_embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=384,  # Sentence-BERT output dimension
            ),
            FieldSchema(
                name="caption_sparse_embedding",
                dtype=DataType.SPARSE_FLOAT_VECTOR,
            ),
            FieldSchema(
                name="caption_text",
                dtype=DataType.VARCHAR,
                max_length=2000,
            ),
            FieldSchema(
                name="frame_index",
                dtype=DataType.INT64,
            ),
            FieldSchema(
                name="segment_start",
                dtype=DataType.INT64,
            ),
            FieldSchema(
                name="segment_end",
                dtype=DataType.INT64,
            ),
        ]
        return CollectionSchema(fields=fields)
```

**Index Configuration:**

```python
# From core/config/milvus_index_config.py

class MilvusIndexBaseConfig(BaseModel):
    index_type: str = "HNSW"
    metric_type: str = "COSINE"
    index_params: dict = Field(default_factory=lambda: {"M": 16, "efConstruction": 64})

# Visual embeddings (512-dim, HNSW)
image_visual_dense_conf = MilvusIndexBaseConfig(
    index_type="HNSW",
    metric_type="COSINE",
    index_params={"M": 16, "efConstruction": 64}
)

# Caption embeddings (384-dim, HNSW)
image_caption_dense_conf = MilvusIndexBaseConfig(
    index_type="HNSW",
    metric_type="COSINE", 
    index_params={"M": 16, "efConstruction": 64}
)

# Sparse embeddings for BM25
image_caption_sparse_conf = MilvusIndexBaseConfig(
    index_type="SPARSE_INVERTED_INDEX",
    metric_type="BM25",
    index_params={"bm25_k1": 1.2, "bm25_b": 0.75}
)
```

**Milvus Persistence Task:**

```python
# From task/milvus_persist_task/main.py

class ImageEmbeddingMilvusTask(BaseTask):
    """Persist image embeddings to Milvus."""
    
    def __init__(
        self,
        artifact_visitor: ArtifactPersistentVisitor,
        ingest_batch_size: int = 500,
    ):
        self.artifact_visitor = artifact_visitor
        self.ingest_batch_size = ingest_batch_size
    
    async def preprocess(
        self,
        input_data: tuple[list[ImageEmbeddingArtifact], list[TextCaptionEmbeddingArtifact]]
    ) -> list[dict]:
        """Prepare embedding data for Milvus insertion."""
        image_embeddings, text_embeddings = input_data
        
        # Group by video_id
        video_groups = defaultdict(list)
        
        for emb in image_embeddings:
            video_groups[emb.related_video_id].append({
                "artifact_id": emb.artifact_id,
                "visual_embedding": load_npy(emb.minio_url),
            })
        
        for emb in text_embeddings:
            video_groups[emb.related_video_id].append({
                "artifact_id": emb.artifact_id,
                "caption_embedding": load_npy(emb.minio_url),
            })
        
        return video_groups
    
    async def execute(
        self,
        preprocessed: list[dict],
        client: BaseMilvusClient
    ) -> list[tuple]:
        """Insert embeddings into Milvus."""
        results = []
        
        for batch in batched(preprocessed, self.ingest_batch_size):
            await client.create_collection_if_not_exists()
            
            ids = await client.insert_vectors(batch)
            results.extend(ids)
        
        return results
```

---

## Configuration Management

### Environment Variables

**Core Configuration (`ingestion/.env`):**

```bash
# MinIO Configuration
MINIO_HOST=localhost
MINIO_PORT=9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_USER=minioadmin
MINIO_PASSWORD=minioadmin
MINIO_SECURE=False
MINIO_API_PORT=9000
MINIO_CONSOLE_PORT=9001

# PostgreSQL Configuration
POSTGRE_DATABASE_URL=postgresql+asyncpg://prefect:prefect@localhost:5432/prefect

# Milvus Configuration
MILVUS_HOST=standalone
MILVUS_PORT=19530
MILVUS_USER=
MILVUS_PASSWORD=
MILVUS_DB_NAME=default

# Consul Configuration
CONSUL_HOST=localhost
CONSUL_PORT=8500

# Service Contexts (paths to microservice code)
AUTOSHOT_CONTEXT=./prefect_agent/service_autoshot
ASR_CONTEXT=./prefect_agent/service_asr
IMAGE_EMBEDDING_CONTEXT=./prefect_agent/service_image_embedding
LLM_CONTEXT=./prefect_agent/service_llm
TEXT_EMBEDDING_CONTEXT=./prefect_agent/service_text_embedding

# Local Data Directory
LOCAL_DATA=service_data
```

**Microservice Configuration (`prefect_agent/service_*/.env`):**

```bash
# Service Autoshot (.env)
SERVICE_NAME=service-autoshot
SERVICE_VERSION=0.1.0
HOST=autoshot
PORT=8001
CPU_FALLBACK=FALSE
AUTOSHOT_MODEL_PATH=/app/weight/transnetv2-pytorch-weights.pth
LOG_LEVEL=DEBUG

# Service ASR (.env)
SERVICE_NAME=service-asr
SERVICE_VERSION=0.1.0
HOST=asr
PORT=8002
CPU_FALLBACK=FALSE
CHUNKFORMER_MODEL_PATH=/app/weight/chunkformer-large-vie
TEMP_DIR=/tmp/asr
DEFAULT_CHUNK_SIZE=300000
DEFAULT_SAMPLE_RATE=16000
LOG_LEVEL=DEBUG

# Service LLM (.env)
SERVICE_NAME=service-llm
SERVICE_VERSION=0.1.0
HOST=llm
PORT=8004
CPU_FALLBACK=TRUE
GEMINI_API_KEY=<api-key>
GEMINI_MODEL_NAME=gemini-1.5-flash
OPENROUTER_API_KEY=<api-key>
OPENROUTER_MODEL_NAME=google/gemini-2.5-flash-lite-preview-09-2025
LOG_LEVEL=DEBUG

# Service Image Embedding (.env)
SERVICE_NAME=service-image-embedding
SERVICE_VERSION=0.1.0
HOST=image-embedding
PORT=8000
CPU_FALLBACK=FALSE
BEIT3_MODEL_CHECKPOINT=/app/weight/beit3/beit3_large_patch16_384_f30k_retrieval.pth
BEIT3_TOKENIZER_CHECKPOINT=/app/weight/beit3/beit3.spm
OPEN_CLIP_MODEL_NAME=ViT-L-14
OPEN_CLIP_PRETRAINED=laion2b_s32b_b82k
LOG_LEVEL=DEBUG

# Service Text Embedding (.env)
SERVICE_NAME=service-text-embedding
SERVICE_VERSION=0.1.0
HOST=text-embedding
PORT=8003
CPU_FALLBACK=FALSE
SENTENCE_TRANSFORMER_MODEL=all-MiniLM-L6-v2
MMBERT_MODEL_NAME=bert-base-multilingual-cased
LOG_LEVEL=DEBUG
```

### Task Configuration (`core/config/task_config.py`)

```python
from pydantic import BaseModel, Field
from typing import Literal

class TaskConfig(BaseModel):
    """Base configuration for tasks."""
    model_name: str
    device: Literal["cuda", "cpu"] = "cuda"
    batch_size: int = 1

class AutoshotSettings(TaskConfig):
    """Autoshot task settings."""
    model_name: str = "autoshot"
    device: Literal["cuda", "cpu"] = "cuda"

class ASRSettings(TaskConfig):
    """ASR task settings."""
    model_name: str = "chunkformer"
    device: Literal["cuda", "cpu"] = "cuda"

class LLMCaptionSettings(TaskConfig):
    """LLM captioning settings."""
    model_name: str = "gemini_api"
    device: Literal["cuda", "cpu"] = "cuda"
    batch_size: int = 1
    image_per_segments: int = 3

class ImageEmbeddingSettings(TaskConfig):
    """Image embedding settings."""
    model_name: str = "open_clip"
    device: Literal["cuda", "cpu"] = "cuda"
    batch_size: int = 16

class TextEmbeddingSettings(TaskConfig):
    """Text embedding settings."""
    model_name: str = "sentence_transformers"
    device: Literal["cuda", "cpu"] = "cuda"
    batch_size: int = 32

class ImageProcessingSettings(BaseModel):
    """Image extraction settings."""
    num_img_per_segment: int = 5
    upload_concurrency: int = 10

class ConsulSettings(BaseModel):
    """Consul client settings."""
    host: str = "localhost"
    port: int = 8500
    timeout_seconds: float = 30.0
    max_retries: int = 3
    retry_min_wait: float = 1.0
    retry_max_wait: float = 10.0
```

---

## Error Handling & Resilience

### Retry Configuration

```python
# Default retry settings from ConsulSettings
consul_conf = ConsulSettings(
    host="localhost",
    port=8500,
    timeout_seconds=30.0,
    max_retries=3,
    retry_min_wait=1.0,
    retry_max_wait=10.0,
)

# Client configuration built from ConsulSettings
base_client_config = ClientConfig(
    timeout_seconds=consul_conf.timeout_seconds,
    max_retries=consul_conf.max_retries,
    retry_min_wait=consul_conf.retry_min_wait,
    retry_max_wait=consul_conf.retry_max_wait,
    consul_host=consul_conf.host,
    consul_port=consul_conf.port,
)
```

### Exception Hierarchy

```python
# From core/clients/base.py

class ClientError(Exception):
    """Base exception for client errors."""
    pass

class ServiceUnavailableError(ClientError):
    """Raised when service is not available."""
    pass

class MilvusClientError(Exception):
    """Raised when Milvus operations fail."""
    pass
```

### Health Check Integration

```python
# Consul health check registration
await consul.register_service(
    service_name="service-autoshot",
    address="autoshot",
    port=8001,
    health_check_url="http://autoshot:8001/health",
    tags=["gpu", "transnetv2"],
)
```

### GPU Fallback Mechanism

```python
# From prefect_agent/shared/service.py

async def load_model(self, model_name: str, device: str):
    # Check GPU availability
    if device == "cuda" and not is_gpu_available():
        if self.service_config.cpu_fallback:
            logger.warning(
                f"GPU not available, falling back to CPU",
                model_name=model_name
            )
            device = "cpu"
        else:
            raise HTTPException(
                400, 
                "GPU required but not available. "
                "Set CPU_FALLBACK=true to use CPU."
            )
    
    # Load model
    handler = get_model_handler(model_name, self.service_config)
    await handler.load_model_impl(device)
```

---

## Monitoring & Observability

### Logging (`core/config/logging.py`)

```python
from loguru import logger
import sys

def setup_logger(
    service_name: str,
    log_level: str = "INFO",
    log_format: str = "console",
    log_file: Optional[str] = None,
):
    """Configure Loguru logger with service context."""
    
    logger.remove()  # Remove default handler
    
    # Console handler
    logger.add(
        sys.stdout,
        format=log_format,
        level=log_level,
    )
    
    # File handler (if specified)
    if log_file:
        logger.add(
            log_file,
            format="{time} | {level} | {message}",
            level=log_level,
            rotation="100 MB",
            retention="10 days",
        )
    
    # Add service context
    logger = logger.bind(service=service_name)
```

### Metrics (`prefect_agent/shared/metrics.py`)

```python
from prometheus_client import Counter, Histogram, Gauge

class ServiceMetrics:
    """Prometheus metrics for service monitoring."""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        
        # Request metrics
        self.requests_total = Counter(
            'service_requests_total',
            'Total number of requests',
            ['endpoint', 'status']
        )
        
        self.request_duration = Histogram(
            'service_request_duration_seconds',
            'Request duration in seconds',
            ['endpoint'],
            buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
        )
        
        self.errors_total = Counter(
            'service_errors_total',
            'Total number of errors',
            ['endpoint', 'error_type']
        )
        
        # System metrics
        self.cpu_usage = Gauge(
            'service_cpu_usage_percent',
            'CPU usage percentage'
        )
        
        self.memory_bytes = Gauge(
            'service_memory_bytes',
            'Memory usage in bytes'
        )
        
        self.gpu_memory_used = Gauge(
            'service_gpu_memory_used_bytes',
            'GPU memory used in bytes',
            ['gpu_id']
        )
        
        self.gpu_utilization = Gauge(
            'service_gpu_utilization_percent',
            'GPU utilization percentage',
            ['gpu_id']
        )
    
    def track_request(self, endpoint: str, status: str):
        """Record request completion."""
        self.requests_total.labels(
            endpoint=endpoint, 
            status=status
        ).inc()
    
    def observe_request_duration(self, endpoint: str, duration: float):
        """Record request duration."""
        self.request_duration.labels(endpoint=endpoint).observe(duration)
    
    def track_error(self, endpoint: str, error_type: str):
        """Record error occurrence."""
        self.errors_total.labels(
            endpoint=endpoint, 
            error_type=error_type
        ).inc()
    
    def update_system_metrics(self):
        """Update system resource metrics."""
        import psutil
        process = psutil.Process()
        
        self.cpu_usage.set(process.cpu_percent())
        self.memory_bytes.set(process.memory_info().rss)
        
        # GPU metrics (if available)
        if is_gpu_available():
            for gpu_id in get_gpu_ids():
                memory = get_gpu_memory(gpu_id)
                utilization = get_gpu_utilization(gpu_id)
                
                self.gpu_memory_used.labels(gpu_id=gpu_id).set(memory)
                self.gpu_utilization.labels(gpu_id=gpu_id).set(utilization)
```

### Progress Tracking

```python
# From core/management/progress.py

class ProcessingStage(Enum):
    VIDEO_INGEST = "video_ingest"
    AUTOSHOT_SEGMENTATION = "autoshot_segmentation"
    ASR_TRANSCRIPTION = "asr_transcription"
    SEGMENT_CAPTIONING = "segment_captioning"
    IMAGE_EXTRACTION = "image_extraction"
    IMAGE_CAPTIONING = "image_captioning"
    IMAGE_EMBEDDING = "image_embedding"
    TEXT_CAP_IMAGE_EMBEDDING = "text_cap_image_embedding"
    TEXT_CAP_SEGMENT_EMBEDDING = "text_cap_segment_embedding"
    IMAGE_MILVUS = "image_milvus"
    TEXT_CAP_SEGMENT_MILVUS = "text_cap_segment_milvus"

class HTTPProgressTracker:
    """Track processing progress via HTTP callbacks."""
    
    def __init__(self, base_url: str, endpoint: str):
        self.base_url = base_url
        self.endpoint = endpoint
        self._cache = {}
    
    async def start_video(self, video_id: str):
        """Mark video processing as started."""
        await self._send_callback(video_id, ProcessingStage.VIDEO_INGEST, 0, 1)
    
    async def update_stage_progress(
        self,
        video_id: str,
        stage: ProcessingStage,
        total_items: int,
        completed_items: int,
        details: dict,
    ):
        """Update progress for a processing stage."""
        progress = completed_items / total_items if total_items > 0 else 0
        await self._send_callback(video_id, stage, completed_items, total_items, details)
    
    async def trigger_http_not_throttle(self, video_id: str):
        """Trigger final notification."""
        # Implementation depends on external callback service
        pass
```

---

## Deployment Architecture

### Docker Compose Services

```yaml
# From docker-compose.yml

services:
  # Infrastructure
  etcd:
    image: quay.io/coreos/etcd:v3.5.18
    # Milvus dependency
    
  standalone:
    image: milvusdb/milvus:v2.6.3
    ports:
      - "19530:19530"
      - "9091:9091"
    depends_on:
      - etcd
      - minio
  
  postgres:
    image: postgres:14
    ports:
      - "5432:5432"
  
  redis:
    image: redis:7
  
  # Prefect Server
  prefect-server:
    image: prefecthq/prefect:3-latest
    ports:
      - "4200:4200"
    depends_on:
      - postgres
      - redis
  
  # Prefect Services
  prefect-services:
    image: prefecthq/prefect:3-latest
    command: prefect server services start
    depends_on:
      - postgres
      - redis
  
  # Prefect Worker
  prefect-worker:
    build:
      context: ./
      dockerfile: Dockerfile
    depends_on:
      - prefect-server
    volumes:
      - ./api:/app/api
      - ./core:/app/core
      - ./flow:/app/flow
      - ./task:/app/task
      - ./main.py:/app/main.py
      - ./prefect_agent/shared:/app/prefect_agent/shared
      - ./prefect_agent/service_asr:/app/prefect_agent/service_asr
      - ./prefect_agent/service_autoshot:/app/prefect_agent/service_autoshot
      - ./prefect_agent/service_llm:/app/prefect_agent/service_llm
      - ./prefect_agent/service_image_embedding:/app/prefect_agent/service_image_embedding
      - ./prefect_agent/service_text_embedding:/app/prefect_agent/service_text_embedding
    environment:
      PREFECT_API_URL: http://prefect-server:4200/api
      NVIDIA_VISIBLE_DEVICES: all
  
  # Microservices
  minio:
    image: minio/minio
    command: server /data --console-address ":9001"
    ports:
      - "9000:9000"
      - "9001:9001"
  
  consul:
    image: consul:latest
    ports:
      - "8500:8500"
  
  autoshot:
    build:
      context: ./prefect_agent/service_autoshot
    ports:
      - "8001:8001"
    environment:
      - SERVICE_NAME=service-autoshot
      - HOST=autoshot
      - PORT=8001
      - CPU_FALLBACK=FALSE
    volumes:
      - ./prefect_agent/weight:/app/weight
    depends_on:
      - consul
      - minio
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
  
  asr:
    build:
      context: ./prefect_agent/service_asr
    ports:
      - "8002:8002"
    environment:
      - SERVICE_NAME=service-asr
      - HOST=asr
      - PORT=8002
      - CPU_FALLBACK=FALSE
    volumes:
      - ./prefect_agent/weight:/app/weight
    depends_on:
      - consul
      - minio
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
  
  llm:
    build:
      context: ./prefect_agent/service_llm
    ports:
      - "8004:8004"
    environment:
      - SERVICE_NAME=service-llm
      - HOST=llm
      - PORT=8004
      - CPU_FALLBACK=TRUE
      - GEMINI_API_KEY=${GEMINI_API_KEY}
      - OPENROUTER_API_KEY=${OPENROUTER_API_KEY}
    depends_on:
      - consul
      - minio
  
  image-embedding:
    build:
      context: ./prefect_agent/service_image_embedding
    ports:
      - "8000:8000"
    environment:
      - SERVICE_NAME=service-image-embedding
      - HOST=image-embedding
      - PORT=8000
      - CPU_FALLBACK=FALSE
    volumes:
      - ./prefect_agent/weight:/app/weight
    depends_on:
      - consul
      - minio
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
  
  text-embedding:
    build:
      context: ./prefect_agent/service_text_embedding
    ports:
      - "8003:8003"
    environment:
      - SERVICE_NAME=service-text-embedding
      - HOST=text-embedding
      - PORT=8003
      - CPU_FALLBACK=FALSE
    volumes:
      - ./prefect_agent/weight:/app/weight
    depends_on:
      - consul
      - minio
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

---

## Extension Points

### Adding a New Microservice

1. **Create service directory:**
   ```bash
   cp -r prefect_agent/service_autoshot prefect_agent/service_new
   ```

2. **Implement service class:**
   ```python
   # service_new/core/service.py
   from shared.service import BaseService
   from service_new.core.config import NewConfig
   from service_new.schema import NewRequest, NewResponse
   
   class NewService(BaseService[NewRequest, NewResponse]):
       def __init__(self, config: NewConfig, log_config: LogConfig):
           super().__init__(config, log_config)
       
       def get_available_models(self) -> list[str]:
           return ["new_model"]
   ```

3. **Implement model handler:**
   ```python
   # service_new/model/registry.py
   from shared.registry import BaseModelHandler, register_model
   
   @register_model("new_model")
   class NewModelHandler(BaseModelHandler[NewRequest, NewResponse]):
       async def load_model_impl(self, device):
           # Load model weights
           pass
       
       async def run_inference(self, preprocessed):
           # Run model inference
           pass
   ```

4. **Register with Consul:**
   ```python
   # In service_new/main.py
   await registry.register_service(
       service_name="service-new",
       address="new",
       port=8005,
       health_check_url="http://new:8005/health",
   )
   ```

5. **Add to docker-compose.yml:**
   ```yaml
   new:
     build:
       context: ./prefect_agent/service_new
     ports:
       - "8005:8005"
     depends_on:
       - consul
       - minio
   ```

### Adding a New Artifact Type

1. **Define schema:**
   ```python
   # core/artifact/schema.py
   class NewArtifact(BaseArtifact):
       new_field: str
       
       @property
       def object_key(self) -> str:
           return f"new/{self.video_name}.json"
   ```

2. **Implement visitor method:**
   ```python
   # core/artifact/persist.py
   def visit_new(self, artifact: NewArtifact, data: NewData):
       # Upload to MinIO
       await self.minio_client.put_json(
           bucket="video-artifacts",
           object_key=artifact.object_key,
           data=data.model_dump()
       )
       
       # Create database record
       artifact_id = await self.tracker.save_artifact(artifact)
       
       # Create lineage edge
       if artifact.parent_artifact_id:
           await self.tracker.create_lineage(
               parent_id=artifact.parent_artifact_id,
               child_id=artifact_id,
               transformation_type="new_processing"
           )
   ```

3. **Add to pipeline:**
   ```python
   # flow/video_processing.py
   @task
   async def new_task(input_data):
       # Implement task logic
       pass
   
   # In video_processing_flow:
   new_artifacts = new_task.submit(previous_output)
   ```

---

## Summary

The ingestion service architecture provides:

1. **Modular Design** - Each processing stage is independent and replaceable
2. **Scalable Microservices** - Individual services can be scaled based on load
3. **Service Discovery** - Dynamic service location via Consul
4. **Fault Tolerance** - Automatic retries, health checks, and fallbacks
5. **Complete Lineage** - Full traceability from source to derived artifacts
6. **Extensibility** - Clear patterns for adding new services and artifact types
7. **Observability** - Comprehensive logging, metrics, and progress tracking
8. **GPU Optimization** - Dynamic model loading/unloading for efficient resource use

This architecture enables production-grade video processing with enterprise reliability and scalability.
