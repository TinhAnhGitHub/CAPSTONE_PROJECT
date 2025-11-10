# Video Ingestion & Deep Search Tooling Report

## 1. Mission & Scope
- Transform raw videos into richly annotated, queryable indexes so agents can ground responses on precise visual moments.
- Surface those indexed artifacts as callable tools that planners, workers, and post-processing agents can combine to answer multimodal user requests.
- Covering two main codebases: the Prefect-driven ingestion service and the `videodeepsearch` agent/tool runtime.

## 2. Pipeline at a Glance
1. **Video registration** – persist uploads, kick off progress tracking, and store originals in MinIO (`ingestion/flow/video_processing.py:69-103`).
2. **Parallel understanding** – detect shot boundaries (Autoshot) and transcribe audio (ASR) so later stages work on aligned timeline slices (`ingestion/flow/video_processing.py:103-218`).
3. **Derived artifacts** – extract key frames, caption segments and images with the LLM service, and package everything as structured artifacts (`ingestion/flow/video_processing.py:218-427`).
4. **Embedding generation** – create dense/sparse vectors for images and segment captions using dedicated embedding services (`ingestion/flow/video_processing.py:331-453`).
5. **Vector persistence** – batch-write vectors into Milvus collections for later semantic search (`ingestion/flow/video_processing.py:480-560`).
6. **Aggregation** – emit a manifest summarizing produced artifacts and update human-readable Prefect artifacts (`ingestion/flow/video_processing.py:560-627`).

Prefect orchestrates the above inside `video_processing_flow`, wiring the necessary clients (MinIO, Postgres, Milvus, Consul) before the run starts (`ingestion/flow/video_processing.py:640-905`).

## 3. Artifact Inventory & Storage Contracts
| Artifact (schema) | Produced by | Stored in MinIO | Tracked in Postgres | Indexed in Milvus | Purpose |
| --- | --- | --- | --- | --- | --- |
| `VideoArtifact` (`ingestion/core/artifact/schema.py:41-76`) | Video upload task | ✅ original video | ✅ lineage root | ❌ | reference video metadata (fps, extension). |
| `AutoshotArtifact` (`:78-123`) | Autoshot detector | ✅ segments JSON | ✅ child of video | ❌ | timeline of segments with frame/time bounds. |
| `ASRArtifact` (`:125-169`) | ASR service | ✅ transcript JSON | ✅ child of video | ❌ | tokenized speech with timestamps & frame indices. |
| `ImageArtifact` (`:171-223`) | Frame extractor | ✅ still frames | ✅ child of segment | ❌ | frame-level visual evidence. |
| `SegmentCaptionArtifact` (`:225-277`) | Segment caption LLM | ✅ caption JSON | ✅ lineage (video→Autoshot/ASR) | ✅ via `TextCapSegmentEmbedArtifact` | natural-language explanation per segment. |
| `ImageCaptionArtifact` (`:279-323`) | Image caption LLM | ✅ caption JSON | ✅ child of image | ✅ via `TextCaptionEmbeddingArtifact` | textual description per frame. |
| `ImageEmbeddingArtifact` (`:325-369`) | Image embedding service | ✅ vector (.npy/JSON) | ✅ child of image | ✅ (visual dense field) | visual embeddings for similarity search. |
| `TextCaptionEmbeddingArtifact` (`:371-417`) | Text embedding service | ✅ vector | ✅ child of caption | ✅ (caption dense field) | text embeddings for image captions. |
| `TextCapSegmentEmbedArtifact` (`:419-467`) | Text embedding service | ✅ vector | ✅ child of segment caption | ✅ (segment dense/sparse fields) | semantic event retrieval. |

**Persistence workflow** (`ingestion/core/artifact/persist.py`):
- Each artifact’s `accept_upload` routes through `ArtifactPersistentVisitor` to upload payloads to MinIO and record metadata/lineage rows in Postgres (`persist.py:55-189`).
- MinIO paths follow deterministic prefixes (`images/`, `caption/segment/`, `embedding/...`) to let tools rebuild context without touching preprocessing code.

## 4. Runtime Services & Infrastructure
- **MinIO** (`core/storage.py`, configured via `.env`) stores binaries and JSON. Bucket paths appear in artifacts as `s3://bucket/object`. Tools decode these URIs with `extract_s3_minio_url` (`videodeepsearch/tools/type/helper.py:1-22`).
- **PostgreSQL** keeps authoritative metadata and parent/child lineage (`videodeepsearch/tools/clients/postgre/client.py:1-130`). Tool calls fetch artifacts by ID to hydrate helper objects.
- **Milvus** hosts two collections:
  - `image_milvus` for frame embeddings (visual + caption dense/sparse dimensions).
  - `segment_milvus` for captioned events.
  Async clients in `videodeepsearch/tools/clients/milvus/client.py` wrap `pymilvus` hybrid search APIs and enforce schema expectations.
- **Consul-discovered microservices** provide Autoshot, ASR, LLM, and embedding workers, all orchestrated per-run by Prefect (see `AppState` wiring in `video_processing_flow`).
- **Prefect** orchestrates concurrency, handles retries, emits Markdown artifacts, and exposes progress via `HTTPProgressTracker`.

## 5. Tool Layer Exposed to Agents
The agent runtime treats every stored artifact as an addressable “instance” by binding database/storage/vector clients into callable `FunctionTool`s.

### 5.1 Registry & Factory
- `ToolRegistry` centralizes metadata, categories, tags, and dependency declarations (`videodeepsearch/tools/type/registry.py:68-265`). It also auto-documents signatures (`_generate_doc`) so planners know capabilities.
- `ToolFactory` injects runtime dependencies (Milvus, Postgres, MinIO, external encoders, LLM) and wraps registered functions into `FunctionTool`s with consistent output formatting (`tools/type/factory.py:147-254`).
- `ToolOutputFormatter` maps returned Pydantic objects to LlamaIndex `ContentBlock`s so agents receive structured summaries rather than raw dicts (`tools/type/factory.py:38-144`).

### 5.2 Search Tools (`tools/type/search.py`)
- `get_images_from_visual_query` (`:18-83`) – dense visual embedding search against Milvus using text-described visual intent.
- `get_images_from_caption_query` (`:87-167`) – caption embedding search for Vietnamese textual queries.
- `get_images_from_multimodal_query` (`:207-287`) – hybrid reranking combining visual, caption dense, and caption sparse scores.
- `get_segments_from_event_query` (`:173-205`) – semantic event retrieval returning timeline-aligned segment objects.
- `find_similar_images_from_image` (`:289-373`) – encode a reference frame (fetched from MinIO) and locate visually similar frames.

### 5.3 Interaction / Navigation Tools (`tools/type/scan.py`)
- Video context: `get_video_from_segment` and `get_video_from_image` hydrate `VideoInterface` with FPS metadata (`:18-65`).
- Segment traversal: `get_all_segment_info_from_video_interface`, `get_segments` let workers hop forward/backwards through shot boundaries (`:78-205`).
- Image browsing: `get_images` returns neighboring frames sorted by timestamp (`:220-309`).
- Frame extraction: `extract_frames_by_time_window` and `extract_frame_time` temporarily download video blobs, sample frames, and push newly generated images back to MinIO for downstream reasoning (`:311-438`).
- ASR access: `get_asr_from_video` exposes the entire transcript, while `get_related_asr_from_segment`/`get_related_asr_from_image` provide focused snippets around a segment or frame (`tools/type/util.py:130-228`).

### 5.4 Utility & Prompt Tools
- Timecode helpers (`frame_to_timecode`, `from_time_to_index`, etc.) convert between frames and HH:MM:SS strings so plans stay synchronized with fps (`tools/type/util.py:21-118`).
- `read_image`/`read_segment` download binary assets from MinIO for visualization or further LLM prompting (`tools/type/util.py:120-170`).
- Prompt engineering helpers (`enhance_visual_query`, `enhance_textual_query`) and cross-modal captioning (`caption_new_image`) harness an LLM-as-tool to expand or restate queries and to describe ad-hoc frames (`tools/type/llm/llm.py:20-102`).

### 5.5 Dependency Injection Guarantees
- Every registered tool declares dependency parameter names; `ToolFactory` binds them, ensuring agents never handle raw clients directly.
- At runtime, agents supply user-scoped fields (`list_video_id`, `user_id`), while the factory supplies Milvus/Postgres/MinIO instances and the external encoding service clients.

## 6. Agent Workflow & Planning
- `VideoAgentWorkFlow` coordinates a multi-agent plan with greeting, planning, orchestration, worker execution, and consolidation stages (`videodeepsearch/agent/workflow.py:33-205`).
  1. **Greeting agent** triages the request and decides whether planning is needed (`workflow.py:73-123`).
  2. **Planner agent** assembles a `WorkersPlan` blueprint referencing available tools and desired outputs (`workflow.py:125-176`, `agent/orc_schemas.py:5-38`).
  3. **Orchestrator & workers** execute tool calls according to the plan, streaming intermediate results (`workflow.py:178-365`).
  4. **Consolidation agent** synthesizes final answers after all worker events complete.
- Events (`agent/orc_events.py`) provide structured checkpoints for UI streaming, human-in-the-loop approvals, or debugging.
- The sandboxed worker executor (`videodeepsearch/worker/executor.py`) safely runs code generated by planning agents when bespoke logic is required to post-process tool outputs.

## 7. From Indexed Artifacts to Agent Actions
1. **Ingest once**: a video run populates MinIO, Postgres, and Milvus with aligned artifacts covering frames, transcripts, captions, and embeddings.
2. **Tool materialization**: when the agent service starts, `ToolFactory` instantiates tool functions bound to these stores, exposing typed interfaces (`ImageObjectInterface`, `SegmentObjectInterface`, `VideoInterface`) from `agentic_ai.tools.schema.artifact`.
3. **User query**: planner reads registry metadata to decide which tools should execute (visual search, segment scan, ASR context, etc.).
4. **Execution**: tool functions query Milvus/Postgres/MinIO, returning lightweight Pydantic objects without downloading entire videos unless explicitly requested.
5. **Reasoning**: workers chain tool outputs (e.g., retrieve segment → fetch ASR snippet → sample new frames) to craft grounded responses.
6. **Response**: consolidation agent weaves retrieved evidence into the final chat answer, optionally providing direct media payloads through formatted content blocks.

## 8. Operational Considerations & Recommendations
- **Service readiness**: all external microservices (Autoshot, ASR, LLM, embedding) must be healthy before launching Prefect flows; Consul registration is handled by the Prefect agents.
- **Vector hygiene**: Milvus insertion tasks skip already-indexed IDs, so reprocessing a video is idempotent as long as artifact IDs remain stable (`task/milvus_persist_task/main.py:28-187`).
- **Scaling**: GPU-tagged tasks (`GPU_TASK_TAG`) can be mapped to dedicated workers; Prefect tags allow selective routing.
- **Extensibility**: adding a new tool only requires decorating a function with `@tool_registry.register`, providing type hints, and letting `ToolFactory` bind dependencies—no planner changes needed.
- **Observability**: Prefect logs combined with `aggregate_results_task` Markdown outputs give per-run summaries that can be surfaced in dashboards.

This document should help new contributors understand how video ingestion artifacts become agent-usable tools, where the data lives, and how to extend the system without breaking existing retrieval behaviors.
