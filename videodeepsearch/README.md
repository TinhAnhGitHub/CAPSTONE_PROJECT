# VideoDeepSearch

Natural-language video retrieval and understanding system. Users query video content via WebSocket, and a multi-agent system searches indexed frames, transcripts, captions, and knowledge graphs to return relevant moments with timestamps and confidence scores.

Built on [Agno](https://github.com/agno-agi/agno) multi-agent framework, [FastAPI](https://fastapi.tiangolo.com/), and [OpenRouter](https://openrouter.ai/) for LLM inference.

## Architecture

```
User Query (WebSocket JSON)
    |
    v
Greeter Team (Qwen 3.6 Plus)
    |-- greetings / capability questions -> direct response
    |-- video search queries -> Orchestrator
    |
    v
Orchestrator (GLM 5.1)
    |-- Phase 1: spawn Worker agents directly
    |-- Phase 2 (if needed): Planning Agent creates execution plan
    |-- Phase 3: deduplicate and merge worker results
    |
    +-- Planning Agent (Qwen 3.6 Plus)
    |     generates sequential tool execution plans
    |
    +-- Worker Agent #1 (selected model + subset of tools)
    |     executes scoped search task
    |
    +-- Worker Agent #2 ...
```

Worker agents are stateless, dynamically assigned a model from a pool of 8 models, and receive only the tools relevant to their task.

### Storage Architecture

| Backend | Purpose |
|---------|---------|
| Qdrant | Vector embeddings (dense: QwenVL/MMBert, sparse: SPLADE) for images, segments, audio |
| Elasticsearch | BM25 full-text search over OCR text extracted from video frames |
| ArangoDB | Knowledge graph (entities, events, micro_events, communities, relations) |
| PostgreSQL | Artifact metadata and lineage tracking, agent session storage |
| MinIO | Raw video files, extracted frames, ASR transcripts |

## Getting Started

### Prerequisites

- Python 3.11+
- Docker and Docker Compose
- OpenRouter API key

### Docker

```bash
docker compose up -d --build videodeepsearch
```

The API is available at `http://localhost:8080`.

### Local Development

```bash
uv sync
uv run uvicorn main:app --reload --port 8080
```

### Background Services (MLflow stack)

```bash
docker compose up -d mlflow mlflow-postgres mlflow-storage mlflow-create-bucket
```

## Configuration

All settings are in `config/settings.yaml`. Environment variables can be referenced as `${VAR_NAME}`.

Key settings:

| Section | Description |
|---------|-------------|
| `llm_provider.agents` | Model assignments for greeter, orchestrator, planning, llm_tool, summarizer |
| `llm_provider.workers` | Pool of worker models with descriptions and strengths |
| `storage` | PostgreSQL, MinIO, Qdrant, Elasticsearch, ArangoDB connection params |
| `inference` | QwenVL, MMBert, SPLADE embedding server endpoints |
| `server` | Host, port, CORS origins |
| `mlflow` | Tracking URI, experiment name |

Required environment variable:

```
OPENROUTER_API_KEY=sk-or-...
```

## API

### WebSocket

```
WS /ws/start_workflow
```

Send JSON:

```json
{
  "user_id": "abc123",
  "video_ids": ["v1", "v2"],
  "user_demand": "Find scenes showing traffic accidents",
  "session_id": "s1"
}
```

Events stream back with types: `RunContent`, `ToolCallStarted`, `ToolCallCompleted`, `TeamRunContentCompleted`.

### Health

```
GET /health/status   -> per-service readiness breakdown
GET /health/ready    -> {"ready": true/false}
GET /                -> {"status": "ok"}
```

## Project Structure

```
videodeepsearch/
├── main.py                              # FastAPI app, CORS, uvicorn entry point
├── config/settings.yaml                 # All configuration
├── Dockerfile                           # 2-stage build (uv)
├── docker-compose.yml                   # MLflow stack + app service
├── pyproject.toml
└── src/videodeepsearch/
    ├── agent/
    │   ├── team.py                      # ignite_workflow(), build_video_search_team()
    │   ├── supervisor/
    │   │   ├── greeter/                 # Entry-point team, routes queries
    │   │   └── orchestrator/            # Spawns workers, aggregates results
    │   ├── member/
    │   │   ├── planning/                # Strategic planning agent
    │   │   └── worker/                  # Stateless worker agent + tool selector
    │   └── synthetic/                   # Q&A generation for evaluation datasets
    ├── api/
    │   ├── stream.py                    # WebSocket endpoint
    │   └── health.py                    # Health/readiness endpoints
    ├── clients/
    │   ├── storage/                     # PostgreSQL, MinIO, Qdrant, Elasticsearch, ArangoDB
    │   └── inference/                   # QwenVL, MMBert, SPLADE clients
    ├── core/
    │   ├── settings.py                  # Pydantic settings loader
    │   ├── lifespan.py                  # Client/model initialization on startup
    │   └── dependencies.py              # FastAPI DI helpers
    ├── toolkit/
    │   ├── search.py                    # VideoSearchToolkit (6 tools)
    │   ├── utility.py                   # UtilityToolkit (4 tools)
    │   ├── video_metadata.py            # VideoMetadataToolkit (2 tools)
    │   ├── kg_retrieval.py              # KGSearchToolkit (8 tools)
    │   ├── ocr.py                       # OCRSearchToolkit (disabled)
    │   ├── llm.py                       # LLMToolkit (disabled)
    │   ├── factories.py                 # Factory functions for toolkit instantiation
    │   └── registry.py                  # ToolRegistry for planning context generation
    ├── schemas/
    │   └── artifacts.py                 # ImageInterface, SegmentInterface, AudioInterface
    ├── evaluation/
    │   ├── datasets/                    # EvalRecord, EvalTrace, DatasetBuilder
    │   ├── scorers/                     # MLflow + DeepEval scorer presets
    │   ├── runners/                     # Evaluation execution scripts
    │   └── util/                        # MLflow setup, team initialization
    └── tracing/
        └── decorator.py                 # @traced_tool() MLflow span decorator
```

## Toolkits

### VideoSearchToolkit (6 tools)

| Tool | Description |
|------|-------------|
| `get_images_from_qwenvl_query` | Search images using QwenVL multimodal embeddings |
| `get_images_from_caption_query_mmbert` | Search images by caption (MMBert) |
| `get_segments_from_qwenvl_query` | Search video segments using QwenVL embeddings |
| `get_segments_from_event_query_mmbert` | Search segments by event description (MMBert) |
| `get_audio_from_query_dense` | Search audio transcripts (dense MMBert) |
| `get_audio_from_query_hybrid` | Search audio transcripts (dense + sparse) |

### UtilityToolkit (4 tools)

| Tool | Description |
|------|-------------|
| `get_related_asr_from_segment` | ASR transcript context around a segment |
| `get_related_asr_from_image` | ASR transcript context around a frame |
| `get_adjacent_segments` | Navigate to neighboring segments |
| `get_adjacent_images` | Navigate to neighboring frames |

### VideoMetadataToolkit (2 tools)

| Tool | Description |
|------|-------------|
| `get_video_metadata` | Detailed metadata for a specific video |
| `get_video_timeline` | Visual timeline with timestamps and captions |

### KGSearchToolkit (8 tools)

| Tool | Description |
|------|-------------|
| `search_entities_semantic` | Semantic search for entities (people, objects, locations) |
| `search_events` | Semantic search for segment-level events |
| `search_micro_events` | Semantic search for frame-level micro-events |
| `search_communities` | Semantic search for thematic clusters |
| `traverse_from_entity` | Graph traversal from seed entity |
| `multi_granularity_search` | Parallel search across entities + events + micro_events |
| `search_bm25` | BM25 keyword search via ArangoSearch |
| `triple_hybrid_search` | BM25 + semantic + graph hybrid with RRF fusion |

## Evaluation

Evaluation follows the AVHaystacksQA paradigm from [MAGNET](https://arxiv.org/abs/2506.07016) (Chowdhury et al., 2025) -- a multi-agent framework for multi-video retrieval and temporal grounding.

### Dataset

| Property | Value |
|----------|-------|
| Videos | 60 (subset from AVHaystacks) |
| Topics | cooking, news, public speaking |
| Questions | 70 |
| Answer format | Enumerated steps: `1) ... 2) ... 3) ...` |
| Ground truth | Expected video IDs + step-wise answers |

### Custom Judges

Two primary metrics evaluated via LLM-as-judge (Gemini 3 Flash Lite Preview, minimal reasoning effort):

**VideoRecall** -- Checks whether each ground-truth video appears in the agent response.

- Input: agent response text + ground-truth video IDs
- LLM determines if each ground-truth video is referenced or retrieved in the response
- A video is "recalled" if the LLM confirms it exists in the agent output
- Final score: `recalled_videos / total_ground_truth_videos`

**AnswerCorrectness** -- Checks whether each ground-truth point is satisfied in the agent answer.

- Input: agent enumerated answer + ground-truth enumerated steps
- LLM evaluates each ground-truth point independently: is it addressed in the agent response?
- Final score: `satisfied_points / total_ground_truth_points` (0-100%)
- Example: 2/3 points satisfied -> 66.7%, 1/1 -> 100%
