# Video Deep Search Agent Framework - Technical Documentation

**Video Deep Search** is an advanced, agentic AI system designed for semantic video understanding, retrieval, and evidence collection. It employs a **hierarchical multi-agent architecture** (Orchestrator-Worker pattern) to decompose complex natural language queries into executable search plans, utilizing a suite of specialized tools interacting with vector databases (Milvus), metadata stores (PostgreSQL), and object storage (MinIO).

---

## 📑 Table of Contents

1.  [System Architecture](#-system-architecture)
    *   [High-Level Design](#high-level-design)
    *   [Agent Hierarchy](#agent-hierarchy)
    *   [State Management](#state-management)
2.  [Core Modules](#-core-modules)
    *   [Agents (`/agent`)](#agents-agent)
    *   [Tooling Ecosystem (`/tools`)](#tooling-ecosystem-tools)
    *   [Core Infrastructure (`/core`)](#core-infrastructure-core)
    *   [API & Streaming (`/api`)](#api--streaming-api)
3.  [Data Structures](#-data-structures)
    *   [Context Models](#context-models)
    *   [Middleware & Data Handles](#middleware--data-handles)
4.  [Workflow Lifecycle](#-workflow-lifecycle)
5.  [Configuration & Setup](#-configuration--setup)
6.  [Development Guide](#-development-guide)

---

## 🌟 System Architecture

### High-Level Design

The system follows an **Index-Then-Act** paradigm. Video content is pre-processed (ingested) into:
*   **Milvus**: Vector embeddings for frames (CLIP) and segment captions (Text/Dense+Sparse).
*   **PostgreSQL**: Metadata, lineage, and structural relationships (Video -> Segment -> Frame).
*   **MinIO**: Raw assets (images, video clips) and intermediate JSON data.

The **Agentic Workflow** sits on top of this index, allowing LLMs to query it using tools.

### Agent Hierarchy

The system uses a strict hierarchical delegation model:

1.  **Greeting Agent (`GREETER_AGENT`)**
    *   **Implementation**: `StreamingFunctionAgent`
    *   **Responsibility**: User Interface layer. It maintains the conversation history, handles clarification questions, and decides when to escalate a request to the Orchestrator.
    *   **Tools**: `running_orchestrator_agent_as_tools`.

2.  **Orchestrator Agent (`ORCHESTRATOR_AGENT`)**
    *   **Implementation**: `StreamingFunctionAgent`
    *   **Responsibility**: Session manager and "Project Manager".
        *   **Step 1**: Spawns the **Planner** to generate a roadmap.
        *   **Step 2**: Spawns **Workers** to execute specific steps of the plan.
        *   **Step 3**: Reviews Worker outputs (Evidence).
        *   **Step 4**: Synthesizes a final report or loops back to Step 1/2 if results are insufficient.
    *   **State**: Holds the `OrchestratorContext` (global session state).

3.  **Planning Agent (`PLANNER_AGENT`)**
    *   **Implementation**: `StreamingFunctionAgent`
    *   **Responsibility**: Pure reasoning. It analyzes the user request and available tools to output a structured **Execution Plan** (JSON blueprint). It does *not* call search tools directly.

4.  **Worker Agent (`WORKER_AGENT`)**
    *   **Implementation**: `WorkerAgent` (Custom implementation)
    *   **Responsibility**: Stateless executors. They receive a specific `task_objective` and a subset of tools.
    *   **Process**: Search -> Inspect (View) -> Verify (Context) -> Persist Evidence -> Submit.

### State Management

State is persisted to the local filesystem (simulating a database) to allow long-running async workflows and recovery.

*   **`FileSystemContextStore`**: Manages serialization/deserialization of agent contexts.
*   **Session Isolation**: Each user session has a unique ID, isolating their chat history and discovered evidence.

---

## 📦 Core Modules

### Agents (`/agent`)

*   **`orc_service.py`**: The entry point `ignite_workflow`. It initializes the `GREETER_AGENT`, sets up the WebSocket event stream, and manages the run loop.
*   **`definition.py`**: Contains `@register_agent` decorators that define the configuration (LLM model, system prompt, temperature) for each agent type.
*   **`agent_as_tool.py`**: Critical module that wraps an entire Agent Workflow into a `FunctionTool`. This allows the Orchestrator to "call" a Worker as if it were a python function.
*   **`custom.py`**: Contains `WorkerAgent` and `StreamingFunctionAgent` classes which override standard LlamaIndex agents to support custom event streaming and thinking blocks.

### Tooling Ecosystem (`/tools`)

The project uses a sophisticated **Decorator-based Registry** system to manage tools.

#### 1. The Registry (`tools/base/registry.py`)
Tools are registered using `@tool_registry.register`. This decorator handles:
*   **Grouping**: Assigns tools to functional groups (e.g., `SEARCH_GROUP`, `UTILITY`).
*   **Bundling**: Assigns tools to strategic roles (e.g., `VIDEO_EVIDENCE_WORKER_BUNDLE`).
*   **Permissions**: Restricts tools to specific agents (e.g., only Orchestrator can use `update_video_context`).

#### 2. Middleware (`tools/base/middleware/`)
To avoid passing massive JSON objects (like 50 search results) directly to the LLM context window, the system uses **Data Handles**.

*   **`DataHandle`**: A lightweight reference object (contains ID + Summary) returned by search tools.
*   **`ResultStore`**: A temporary in-memory store (persisted alongside context) that maps `handle_id` -> `Raw Data`.
*   **Input/Output Middleware**:
    *   **Output**: Intercepts a tool's raw list of results -> saves to Store -> returns `DataHandle`.
    *   **Input**: Intercepts a tool call (e.g., `worker_view_results`) -> looks up `handle_id` -> injects raw data into the function.

#### 3. Implementations (`tools/implementation/`)
*   **`search/`**:
    *   `get_images_from_visual_query`: CLIP-based frame search.
    *   `get_segments_from_event_query`: Semantic segment search (text-to-text).
    *   `get_images_from_multimodal_query`: Hybrid fusion of visual + caption signals.
*   **`scan/`**:
    *   `get_segments`: Temporal navigation (next/prev segments).
    *   `extract_frame_time`: Grabs a specific frame as bytes.
*   **`util/`**:
    *   `get_related_asr_from_video_id`: Fetches transcripts for a time window.
*   **`persist/`**:
    *   `worker_persist_evidence`: Saves high-confidence findings to the shared context.
*   **`view/`**:
    *   `worker_view_results`: Formats `DataHandle` content for the LLM to read.

### Core Infrastructure (`/core`)

*   **`app_state.py`**: Singleton pattern (`Appstate`) holding active connections to Milvus, Postgres, MinIO, and LLM instances. Ensures one connection pool per application lifecycle.
*   **`lifespan.py`**: Manages the startup sequence (connect DBs, load embeddings models) and shutdown cleanup.
*   **`config/`**: Pydantic `BaseSettings` classes for strict environment variable validation.

### API & Streaming (`/api`)

*   **`stream.py`**: Implements the WebSocket endpoint `/ws/start_workflow`.
    *   Handles incoming JSON (user query).
    *   Iterates over the `ignite_workflow` generator.
    *   Serializes internal LlamaIndex events (`AgentStream`, `ToolCall`, `AgentOutput`) into JSON for the frontend.

---

## 🏗️ Data Structures

### Context Models
Defined in `agent/context/`:

*   **`OrchestratorContext`**:
    ```python
    class OrchestratorContext(BaseModel):
        session_id: str
        video_context: dict[str, VideoContext]  # Accumulated facts per video
        history_worker_results: list[list[WorkerResult]] # History of all worker outputs
        chat_history: list[ChatMessage]
    ```

*   **`WorkerResult`** (Output of a worker):
    ```python
    class WorkerResult(BaseModel):
        worker_name: str
        task_objective: str
        evidences: list[EvidenceItem] # Key findings
        result_summary: str # Narrative report
    ```

*   **`EvidenceItem`**:
    ```python
    class EvidenceItem(BaseModel):
        artifacts: Sequence[ImageInterface | SegmentInterface]
        confidence_score: int
        claims: str # Why is this evidence relevant?
    ```

### Middleware & Data Handles
Used to keep context windows small.

*   **`DataHandle`**:
    ```python
    class DataHandle(BaseModel):
        handle_id: str # UUID reference
        summary: str   # "Found 50 images, top score 0.85..."
        related_video_ids: list[str]
    ```

---

## 🔄 Workflow Lifecycle

1.  **Ignition**: User connects to WebSocket -> `start_workflow_ws`.
2.  **Greeter Analysis**: `GREETER_AGENT` receives text. If it's a search task -> calls `running_orchestrator_agent_as_tools`.
3.  **Orchestrator Planning**:
    *   `ORCHESTRATOR` calls `run_planning_agent_as_tool`.
    *   `PLANNER` (using `get_registry_tools`) looks at available capabilities and returns a JSON plan (e.g., "Spawn 'VisualWorker' to find red cars").
4.  **Worker Execution**:
    *   `ORCHESTRATOR` calls `running_worker_agent_as_tools` for each task.
    *   **Worker Loop**:
        1.  **Search**: Calls `get_images_from_visual_query`. Middleware stores results, returns `DataHandle`.
        2.  **View**: Worker calls `worker_view_results(handle_id)`. Middleware retrieves data, formats top 10 matches as text.
        3.  **Verify**: Worker calls `get_related_asr_from_video_id` to check dialogue.
        4.  **Persist**: Worker calls `worker_persist_evidence` for good matches.
        5.  **Finish**: Returns summary string.
5.  **Synthesis**:
    *   `ORCHESTRATOR` receives `WorkerResult`.
    *   Updates `video_context` with new findings.
    *   Generates final response to `GREETER`, which streams it to User.

---

## 🔧 Configuration & Setup

### Environment Variables (`.env`)

**LLM Configuration (Gemini):**
```env
GREETING_LLM_MODEL_NAME="gemini-2.5-flash"
GREETING_LLM_THINKING="true"
GREETING_LLM_THINKING_BUDGET="1024"
PLANNER_LLM_MODEL_NAME="gemini-2.5-flash"
SUB_ORCHESTRATOR_LLM_MODEL_NAME="gemini-2.5-flash"
SUB_WORKER_LLM_MODEL_NAME="gemini-2.5-flash"
OUTPUT_RESP_MODEL_NAME="gemini-2.5-flash"
```

**Databases & Storage:**
```env
# Milvus
IMAGE_MILVUS_URI="http://localhost:19530"
IMAGE_MILVUS_COLLECTION_NAME="image_collection"
SEGMENT_MILVUS_CAPTION_URI="http://localhost:19530"
SEGMENT_MILVUS_COLLECTION_NAME="segment_collection"

# Postgres
POSTGRES_CLIENT_DATABASE_URL="postgresql+asyncpg://user:pass@localhost:5432/db"

# MinIO
MINIO_STORAGE_CLIENT_HOST="localhost"
MINIO_STORAGE_CLIENT_PORT="9000"
MINIO_STORAGE_CLIENT_ACCESS_KEY="minioadmin"
MINIO_STORAGE_CLIENT_SECRET_KEY="minioadmin"
MINIO_STORAGE_CLIENT_SECURE="False"
```

**External Embeddings:**
```env
EXTERNAL_IMAGE_EMBEDDING_CLIENT_BASE_URL="http://localhost:8000"
EXTERNAL_TEXT_EMBEDDING_CLIENT_BASE_URL="http://localhost:8001"
```

### Running

1.  **Docker Infrastructure**:
    ```bash
    docker-compose up -d # Starts Postgres, Phoenix (Tracing)
    # Ensure Milvus and MinIO are running separately or add to compose
    ```

2.  **Application**:
    ```bash
    # Install dependencies
    pip install -e .
    
    # Run Server
    python main.py
    ```

---

## 👨‍💻 Development Guide

### Adding a New Tool

1.  **Define the function** in `tools/implementation/`.
2.  **Register it** using the registry.
    ```python
    @tool_registry.register(
        group_doc_name=GroupName.SEARCH_GROUP,
        bundle_spec=VIDEO_EVIDENCE_WORKER_BUNDLE,
        bundle_role_key=BundleRoles.SEMANTIC_SEARCHER,
        output_middleware=output_image_results, # Optional middleware
        belong_to_agents=[WORKER_AGENT]
    )
    async def my_new_tool(query: str) -> List[Result]:
        ...
    ```
3.  **Middleware**: If your tool returns a complex object, implement an output middleware in `tools/base/middleware/output.py` to wrap it in a `DataHandle`.

### Adding a New Agent

1.  **Define Config**: Add a new config class in `core/config/llm_config.py`.
2.  **Register**: Use `@register_agent` in `agent/definition.py`.
3.  **Integrate**: Add logic in `agent/orc_service.py` or `agent_as_tool.py` to spawn it.

### Testing
Use the `test/` directory.
*   `test_agent_wf/`: Scripts to run agents in isolation.
*   `test_client/`: CLI client to interact with the running WebSocket API.
    ```bash
    python test/test_client/main.py
    ```