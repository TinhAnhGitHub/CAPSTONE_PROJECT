# VideoDeepSearch

VideoDeepSearch is the retrieval and reasoning service for the Capstone video system. It receives natural-language questions over WebSocket, builds a multi-agent search workflow, calls retrieval tools over indexed video artifacts, and streams intermediate tool events plus the final answer back to the client.

VideoDeepSearch does not ingest videos itself. It expects data produced by `video_pipeline`: artifacts in PostgreSQL and MinIO, vectors in Qdrant, OCR documents in Elasticsearch, and knowledge graph records in ArangoDB.

## Table Of Contents

- [What This Service Does](#what-this-service-does)
- [Runtime Architecture](#runtime-architecture)
- [Request Flow](#request-flow)
- [Agent Architecture](#agent-architecture)
- [Toolkits](#toolkits)
- [Storage And Retrieval Backends](#storage-and-retrieval-backends)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Local Development](#local-development)
- [Docker Development](#docker-development)
- [Evaluation And Tracing](#evaluation-and-tracing)
- [Operations And Troubleshooting](#operations-and-troubleshooting)
- [Project Layout](#project-layout)

## What This Service Does

VideoDeepSearch answers questions such as:

```text
Find the moment where the speaker discusses the Apollo mission.
Show the frames where a red object appears.
Which videos contain traffic accident scenes?
Summarize the timeline of video v1.
Find entities related to a kitchen scene and explain what happened.
```

It combines several retrieval modes:

| Retrieval mode | Backend | Best for |
|----------------|---------|----------|
| Image multimodal search | Qdrant `_image` collection with QwenVL vectors | Visual appearance, objects, scenes, frame-level search. |
| Image caption search | Qdrant `_image_caption` collection with mmBERT/SPLADE | Caption semantics and text descriptions of frames. |
| Segment multimodal search | Qdrant `_segment` collection with QwenVL vectors | Multi-frame scene or event retrieval. |
| Segment caption search | Qdrant `_segment_caption` collection with mmBERT/SPLADE | Event and action descriptions. |
| Audio transcript search | Qdrant `_audio_transcript` collection with mmBERT/SPLADE | Spoken content and dialogue. |
| OCR text search | Elasticsearch | Visible text in frames. The toolkit is present but disabled in the active worker registry. |
| Metadata lookup | PostgreSQL and MinIO | Video details, timelines, adjacent frames/segments, ASR context. |
| Knowledge graph search | ArangoDB | Entity, event, micro-event, graph traversal, BM25, and hybrid graph retrieval. |

## Runtime Architecture

```text
Frontend or API client
  |
  | WebSocket /ws/start_workflow
  v
FastAPI app on :8080
  |
  | lifespan initializes clients and OpenRouter models
  v
Greeter team
  |
  | routes simple messages or search requests
  v
Orchestrator team
  |
  | asks planning agent when useful
  | spawns isolated worker agents
  v
Worker agents
  |
  | use selected tool subsets
  v
Qdrant + PostgreSQL + MinIO + Elasticsearch + ArangoDB + inference services
```

Main runtime files:

| File | Responsibility |
|------|----------------|
| `main.py` | FastAPI app, CORS, router registration, Uvicorn dev entrypoint. |
| `src/videodeepsearch/api/stream.py` | WebSocket endpoint and stream loop. |
| `src/videodeepsearch/api/health.py` | Readiness and status endpoints. |
| `src/videodeepsearch/core/lifespan.py` | Initializes all storage clients, inference clients, Agno DB, and OpenRouter models. |
| `src/videodeepsearch/core/settings.py` | Loads `config/settings.yaml` and environment overrides. |
| `src/videodeepsearch/agent/team.py` | Builds the full video search team and converts Agno stream events to JSON. |
| `src/videodeepsearch/toolkit/` | Retrieval tools available to worker agents. |

## Request Flow

1. A client connects to `WS /ws/start_workflow`.
2. The client sends JSON with `user_id`, `video_ids`, `user_demand`, and `session_id`.
3. The WebSocket handler reads initialized clients and models from `app.state`.
4. `ignite_workflow()` builds a session-scoped Agno team.
5. The Greeter team receives the user demand.
6. Simple greeting or capability questions can be answered directly.
7. Retrieval questions are delegated to the Orchestrator team.
8. The Orchestrator can inspect available worker tools and available worker models.
9. The Planning Agent can produce a step-by-step tool execution plan.
10. The Orchestrator calls `spawn_and_run_worker()` one or more times.
11. Each worker receives a selected model and selected tool subset.
12. Worker tool calls query Qdrant, PostgreSQL, MinIO, Elasticsearch, ArangoDB, and inference services.
13. Agno stream events are serialized and emitted to the WebSocket client.
14. The final team output is emitted after the streaming run completes.

## Agent Architecture

### Greeter Team

Code: `src/videodeepsearch/agent/supervisor/greeter/`

The Greeter is the outer team. It receives the user message first and decides whether to answer directly or route the request to the Orchestrator.

Configured model key: `llm_provider.agents.greeter`.

### Orchestrator Team

Code: `src/videodeepsearch/agent/supervisor/orchestrator/`

The Orchestrator coordinates retrieval. It has access to the `SpawnWorkerToolkit`, which exposes:

| Tool | Purpose |
|------|---------|
| `get_available_worker_tools()` | Lists all worker tools grouped by toolkit alias. |
| `get_available_models()` | Lists configured worker models and strengths. |
| `spawn_and_run_worker()` | Creates a worker agent with selected model and selected tool names. |

Configured model key: `llm_provider.agents.orchestrator`.

### Planning Agent

Code: `src/videodeepsearch/agent/member/planning/`

The Planning Agent receives the tool registry context and can produce explicit plans using available tool names. It does not directly query video storage; it helps the Orchestrator decide what workers should do.

Configured model key: `llm_provider.agents.planning`.

### Worker Agents

Code: `src/videodeepsearch/agent/member/worker/`

Workers are stateless, isolated agents created dynamically by the Orchestrator. Each worker receives:

| Input | Meaning |
|-------|---------|
| `agent_name` | Unique worker name for the run. |
| `description` | Short role description. |
| `task` | Scoped task to complete. |
| `detail_plan` | Step-by-step plan from the Orchestrator or Planning Agent. |
| `user_demand` | Original user request. |
| `model_name` | Key from `llm_provider.workers`. |
| `tool_names` | Tool names such as `search.get_images_from_qwenvl_query`. Empty list means all tools. |

Worker run metrics are stored in session state and aggregated by the Orchestrator post hook.

## Toolkits

Toolkits are created in `src/videodeepsearch/toolkit/factories.py` and registered in `agent/team.py`. Active worker toolkit aliases are `search`, `utility`, `video`, and `kg`. OCR and generic LLM toolkits exist in code but are currently commented out in the active selector and registry.

### VideoSearchToolkit

Alias: `search`

Code: `src/videodeepsearch/toolkit/search.py`

| Tool | Backend | Purpose |
|------|---------|---------|
| `get_images_from_qwenvl_query` | Qdrant image collection and QwenVL text embedding | Search extracted frames using unified multimodal embeddings. |
| `get_images_from_caption_query_mmbert` | Qdrant image-caption collection, mmBERT, optional SPLADE | Search frame captions by semantic or hybrid text query. |
| `get_segments_from_qwenvl_query` | Qdrant segment collection and QwenVL text embedding | Search temporal segments using multimodal segment vectors. |
| `get_segments_from_event_query_mmbert` | Qdrant segment-caption collection, mmBERT, optional SPLADE | Search event/scene captions. |
| `get_audio_from_query_dense` | Qdrant audio-transcript collection and mmBERT | Search spoken content semantically. |
| `get_audio_from_query_hybrid` | Qdrant audio-transcript collection, mmBERT, SPLADE | Search spoken content with dense plus sparse hybrid retrieval. |

Common filters:

| Parameter | Meaning |
|-----------|---------|
| `top_k` | Number of results. Defaults to 10. |
| `video_ids` | Optional video filter. Can be a Python list or JSON string. Defaults to the request's `video_ids`. |
| `user_id` | Applied from the active session context. |

### UtilityToolkit

Alias: `utility`

Code: `src/videodeepsearch/toolkit/utility.py`

| Tool | Purpose |
|------|---------|
| `get_related_asr_from_segment` | Gets ASR transcript context around a segment time range. |
| `get_related_asr_from_image` | Gets ASR transcript context around a frame timestamp. |
| `get_adjacent_segments` | Navigates forward or backward from a reference segment. |
| `get_adjacent_images` | Navigates forward or backward from a reference frame. |

These tools query artifact metadata from PostgreSQL and can reconstruct temporal context around vector search results.

### VideoMetadataToolkit

Alias: `video`

Code: `src/videodeepsearch/toolkit/video_metadata.py`

| Tool | Purpose |
|------|---------|
| `get_video_metadata` | Reads a `VideoArtifact` and returns FPS, duration, extension, MinIO URL, creation time, and metadata. |
| `get_video_timeline` | Returns a segment, shot, or minute-level timeline for a video. |

Timeline granularities:

| Granularity | Source |
|-------------|--------|
| `segment` | `SegmentCaptionArtifact` children. |
| `shot` | `AutoshotArtifact.metadata.segments`. |
| `minute` | Segment captions grouped by start minute. |

### KGSearchToolkit

Alias: `kg`

Code: `src/videodeepsearch/toolkit/kg_retrieval.py`

| Tool | Backend | Purpose |
|------|---------|---------|
| `search_entities_semantic` | ArangoDB vector functions and mmBERT query embedding | Finds canonical entities by semantic similarity. |
| `search_events` | ArangoDB vector functions and mmBERT query embedding | Finds segment-level event nodes. |
| `search_micro_events` | ArangoDB vector functions and mmBERT query embedding | Finds fine-grained micro-event nodes. |
| `traverse_from_entity` | ArangoDB graph traversal | Explores connected entities, events, and micro-events from a seed entity key. |
| `multi_granularity_search` | ArangoDB vector functions | Searches entities, events, and micro-events in one pass. |
| `search_bm25` | ArangoSearch view | Keyword search over entities, events, and micro-events. |
| `triple_hybrid_search` | ArangoSearch, vector similarity, graph expansion, RRF | Combines BM25, semantic vector search, and graph-based expansion. |

Graph collections expected from `video_pipeline`:

| Collection | Meaning |
|------------|---------|
| `entities` | Canonical people, objects, locations, concepts. |
| `events` | Segment-level events. |
| `micro_events` | Atomic event captions. |
| `entity_relations` | Entity-to-entity relationships. |
| `event_sequences` | Event-to-event edges. |
| `event_entities` | Event-to-entity links. |
| `micro_event_sequences` | Micro-event-to-micro-event edges. |
| `micro_event_parents` | Micro-event to parent event links. |
| `micro_event_entities` | Micro-event-to-entity links. |

### Disabled Toolkits

The codebase includes `OCRSearchToolkit` and `LLMToolkit`, but they are currently commented out in `_build_tool_selector()` and `_build_tool_registry()` in `agent/team.py`. They are not available to workers unless those registrations are enabled.

## Storage And Retrieval Backends

VideoDeepSearch reads data created by Video Pipeline.

| Backend | Client code | Data read |
|---------|-------------|-----------|
| PostgreSQL | `clients/storage/postgre/` | Artifact metadata, lineage, Agno session storage. |
| MinIO | `clients/storage/minio/` | Raw files and stored artifacts when a tool needs object bytes or URLs. |
| Qdrant | `clients/storage/qdrant/` | Dense and hybrid vector collections. |
| Elasticsearch | `clients/storage/elasticsearch/` | OCR documents. Initialized by lifespan, currently not exposed to active workers. |
| ArangoDB | `clients/storage/arangodb/` | Knowledge graph collections and ArangoSearch view. |

Qdrant base collection configuration comes from `storage.qdrant.collection_name`. Specialized clients append suffixes:

| Client | Collection suffixes |
|--------|---------------------|
| `ImageQdrantClient` | `_image`, `_image_caption` |
| `SegmentQdrantClient` | `_segment`, `_segment_caption` |
| `AudioQdrantClient` | `_audio_transcript` |

Inference clients initialized at startup:

| Client | Config section | Used for |
|--------|----------------|----------|
| `QwenVLEmbeddingClient` | `inference.qwenvl.base_url` | Text queries into QwenVL multimodal vector space. |
| `MMBertClient` | `inference.mmbert.base_url` | Dense text embeddings for captions, audio, KG semantic search. |
| `SpladeClient` | `inference.splade.url` | Sparse text vectors for hybrid Qdrant search. |

## API Reference

The service runs on port `8080` by default.

### Root

```http
GET /
```

Response:

```json
{
  "status": "ok",
  "service": "video-agent-workflow"
}
```

### Health Status

```http
GET /health/status
```

Returns booleans for initialized clients and lists configured agent model keys:

```json
{
  "postgres": true,
  "minio": true,
  "qdrant_image": true,
  "qdrant_segment": true,
  "qdrant_audio": true,
  "elasticsearch": true,
  "arangodb": true,
  "qwenvl": true,
  "mmbert": true,
  "splade": true,
  "models": ["greeter", "orchestrator", "planning", "llm_tool", "summarizer"],
  "worker_models": ["qwen/qwen3.6-plus"]
}
```

### Readiness

```http
GET /health/ready
```

Returns:

```json
{
  "ready": true,
  "checks": {
    "postgres": true,
    "minio": true,
    "qdrant": true,
    "elasticsearch": true,
    "arangodb": true,
    "inference": true,
    "models": true,
    "worker_models": true
  }
}
```

### Workflow WebSocket

```text
WS /ws/start_workflow
```

Required message:

```json
{
  "user_id": "user-abc",
  "video_ids": ["vid-001", "vid-002"],
  "user_demand": "Find scenes where someone opens a door and explain the surrounding context.",
  "session_id": "chat-session-001"
}
```

Optional fields such as `chat_history` may be sent by clients, but the current handler only extracts the required fields above.

Streaming output contains serialized Agno events. Common event types include:

| Event | Meaning |
|-------|---------|
| `RunContent` or `TeamRunContent` | Natural-language content from an agent or team. |
| `ToolCallStarted` or `TeamToolCallStarted` | Tool call has started. |
| `ToolCallCompleted` or `TeamToolCallCompleted` | Tool call completed and returned data. |
| `TeamRunContentCompleted` | Final team output. |
| `error` | Validation or runtime error emitted by the WebSocket handler. |

Minimal browser example:

```javascript
const ws = new WebSocket("ws://localhost:8080/ws/start_workflow");

ws.onopen = () => {
  ws.send(JSON.stringify({
    user_id: "user-abc",
    video_ids: ["vid-001"],
    user_demand: "What happens around the kitchen scene?",
    session_id: "session-001"
  }));
};

ws.onmessage = (event) => {
  console.log(JSON.parse(event.data));
};
```

## Configuration

Default config file:

```text
config/settings.yaml
```

Override config path:

```env
CONFIG_PATH=/app/config/settings.yaml
```

Required secret:

```env
OPENROUTER_API_KEY=your-openrouter-key
```

Settings sections:

| Section | Purpose |
|---------|---------|
| `llm_provider.api_key` | OpenRouter API key or `${OPENROUTER_API_KEY}` reference. |
| `llm_provider.agents.greeter` | Outer routing team model. |
| `llm_provider.agents.orchestrator` | Coordinator model. |
| `llm_provider.agents.planning` | Planning agent model. |
| `llm_provider.agents.llm_tool` | Optional model for generic LLM tools. Falls back to planning model when missing. |
| `llm_provider.agents.summarizer` | Optional summary model. Falls back to planning model when missing. |
| `llm_provider.workers` | Pool of worker models with descriptions and strengths. |
| `storage.postgres` | Artifact database and Agno session DB. |
| `storage.minio` | Object storage. |
| `storage.qdrant` | Vector database and base collection name. |
| `storage.elasticsearch` | OCR search backend. |
| `storage.arangodb` | KG database, graph name, and search view. |
| `inference.qwenvl` | QwenVL embedding endpoint. |
| `inference.mmbert` | mmBERT embedding endpoint. |
| `inference.splade` | SPLADE endpoint. |
| `cache` | Cache settings for future/utility usage. |
| `server` | Host, port, and CORS origins. |
| `mlflow` | MLflow tracking settings. |

Environment overrides are applied in `core/settings.py`. Common overrides:

| Environment variable | Overrides |
|----------------------|-----------|
| `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_URL` | PostgreSQL connection. |
| `MINIO_HOST`, `MINIO_PORT`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`, `MINIO_SECURE` | MinIO connection. |
| `QDRANT_HOST`, `QDRANT_PORT`, `QDRANT_GRPC_PORT`, `QDRANT_COLLECTION` | Qdrant connection and collection base. |
| `ELASTICSEARCH_HOST`, `ELASTICSEARCH_PORT`, `ELASTICSEARCH_USER`, `ELASTICSEARCH_PASSWORD`, `ELASTICSEARCH_INDEX` | Elasticsearch connection. |
| `ARANGO_HOST`, `ARANGO_DB`, `ARANGO_USERNAME`, `ARANGO_PASSWORD` | ArangoDB connection. |
| `QWENVL_BASE_URL`, `MMBERT_BASE_URL`, `SPLADE_URL` | Inference endpoints. |

## Local Development

Install dependencies:

```bash
uv sync
```

Run the API locally:

```bash
uv run uvicorn main:app --reload --port 8080
```

Alternative direct script run:

```bash
uv run python main.py
```

Useful local endpoints:

```text
http://localhost:8080/
http://localhost:8080/health/status
http://localhost:8080/health/ready
ws://localhost:8080/ws/start_workflow
```

Local runs require reachable storage and inference services. The easiest setup is usually to run the shared infrastructure from `video_pipeline/docker/docker-compose.yaml` and point VideoDeepSearch to the same `video_shared_net` services.

## Docker Development

Compose file:

```text
videodeepsearch/docker-compose.yml
```

The app service expects two external Docker networks:

| Network | Purpose |
|---------|---------|
| `video_shared_net` | Shared infrastructure from `video_pipeline`: PostgreSQL, MinIO, Qdrant, Elasticsearch, ArangoDB, inference services. |
| `mlflow-network` | MLflow backend stack. |

Create networks if they do not exist:

```bash
docker network create video_shared_net
docker network create mlflow-network
```

Create `videodeepsearch/.env` with at least:

```env
OPENROUTER_API_KEY=your-openrouter-key
DOCKER_LOCAL_VOLUME=/absolute/path/for/videodeepsearch-data
```

Start VideoDeepSearch and MLflow stack:

```bash
cd videodeepsearch
docker compose up -d --build
```

Start only the MLflow stack:

```bash
docker compose up -d mlflow-postgres mlflow-storage mlflow-create-bucket mlflow
```

Start only the app after dependencies are available:

```bash
docker compose up -d --build videodeepsearch
```

Compose service ports:

| Service | Port | Purpose |
|---------|------|---------|
| `videodeepsearch` | `8080` | FastAPI/WebSocket API. |
| `mlflow` | `5000` | MLflow tracking UI/API. |
| `mlflow-storage` | `9010`, `9011` | S3-compatible artifact storage and console. |
| `mlflow-postgres` | internal | MLflow backend store. |

The app container sets service hostnames such as `postgres`, `minio`, `qdrant`, `elasticsearch`, and `arangodb`; those names resolve only if the corresponding services are on `video_shared_net`.

## Evaluation And Tracing

Evaluation code lives under `src/videodeepsearch/evaluation`.

| Directory | Purpose |
|-----------|---------|
| `datasets/` | Dataset schemas, loaders, and builders. |
| `scorers/` | Metric presets and scorer configuration. |
| `runners/` | Evaluation and validation runners. |
| `util/` | MLflow helpers and team preparation. |

The README's previous evaluation setup follows an AVHaystacksQA/MAGNET-style workflow with video recall, answer correctness, and temporal grounding metrics.

Tracing:

| Component | Behavior |
|-----------|----------|
| `tracing/decorator.py` | Provides `@traced_tool()` around toolkit methods. |
| `ignite_workflow()` | Enables `mlflow.agno.autolog()` and starts an MLflow run per workflow. |
| MLflow Compose stack | Provides local tracking and artifact storage. |

Note: `ignite_workflow()` currently sets `mlflow.set_tracking_uri("http://100.113.186.28:5000")` directly in code. If running in a different environment, update that value or make it respect `settings.mlflow.tracking_uri` before relying on MLflow traces.

## Operations And Troubleshooting

Check readiness:

```bash
curl http://localhost:8080/health/ready
curl http://localhost:8080/health/status
```

Common issues:

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `/health/ready` reports `models=false` | OpenRouter key missing or config invalid. | Set `OPENROUTER_API_KEY` and verify `config/settings.yaml`. |
| `/health/ready` reports `qdrant=false` | Qdrant host/port not reachable. | Start Qdrant on `video_shared_net` or override `QDRANT_HOST`. |
| Tool searches return no results | Video was not processed or collection base does not match ingestion. | Verify `video_pipeline` completed and `QDRANT_COLLECTION` matches pipeline `QDRANT_COLLECTION_BASE`. |
| KG tools fail with collection/view errors | ArangoDB graph or search view is missing. | Run the KG and Arango indexing stages and verify Arango schema/index setup. |
| WebSocket returns missing fields error | Request JSON lacks required keys. | Send `user_id`, `video_ids`, `user_demand`, and `session_id`. |
| Worker cannot resolve requested tool | Tool name does not match `alias.function_name`. | Call `get_available_worker_tools()` or use aliases documented above. |
| MLflow traces do not appear | Tracking URI is hardcoded or MLflow service is not reachable. | Start MLflow and align tracking URI. |

Operational checks against indexed data:

1. Confirm `video_pipeline` completed for the target `video_id`.
2. Check PostgreSQL has a `VideoArtifact` with that ID.
3. Check Qdrant collections with suffixes `_image`, `_segment`, `_audio_transcript`, `_image_caption`, and `_segment_caption`.
4. Check ArangoDB has `entities`, `events`, and `micro_events` for the same `video_id`.
5. Check Elasticsearch index `video_ocr_docs_dev` if OCR search is needed.

## Project Layout

```text
videodeepsearch/
|-- main.py
|-- Dockerfile
|-- docker-compose.yml
|-- config/
|   `-- settings.yaml
|-- src/videodeepsearch/
|   |-- agent/
|   |   |-- team.py
|   |   |-- supervisor/
|   |   |   |-- greeter/
|   |   |   `-- orchestrator/
|   |   |-- member/
|   |   |   |-- planning/
|   |   |   `-- worker/
|   |   `-- synthetic/
|   |-- api/
|   |   |-- stream.py
|   |   `-- health.py
|   |-- clients/
|   |   |-- inference/
|   |   `-- storage/
|   |       |-- arangodb/
|   |       |-- elasticsearch/
|   |       |-- minio/
|   |       |-- postgre/
|   |       `-- qdrant/
|   |-- core/
|   |   |-- settings.py
|   |   |-- lifespan.py
|   |   `-- dependencies.py
|   |-- schemas/
|   |   `-- artifacts.py
|   |-- toolkit/
|   |   |-- search.py
|   |   |-- utility.py
|   |   |-- video_metadata.py
|   |   |-- kg_retrieval.py
|   |   |-- ocr.py
|   |   |-- llm.py
|   |   |-- factories.py
|   |   `-- registry.py
|   |-- evaluation/
|   |   |-- datasets/
|   |   |-- scorers/
|   |   |-- runners/
|   |   `-- util/
|   `-- tracing/
|       `-- decorator.py
|-- pyproject.toml
`-- uv.lock
```

## Relationship To Video Pipeline

VideoDeepSearch is the query-time service. Video Pipeline is the ingest-time service.

| Concern | Video Pipeline | VideoDeepSearch |
|---------|----------------|-----------------|
| Video submission | Yes | No |
| Artifact creation | Yes | No |
| Embedding creation | Yes | Query embeddings only |
| Vector indexing | Yes | Reads Qdrant |
| OCR indexing | Yes | Reads Elasticsearch when enabled |
| KG creation | Yes | Reads ArangoDB |
| Agentic search | No | Yes |
| WebSocket streaming | No | Yes |

Run `video_pipeline` first to ingest videos, then query those processed `video_ids` through VideoDeepSearch.
