# Video Deep Search

## Overview

Video Deep Search is an intelligent multi-agent video understanding system that enables semantic search and retrieval across indexed video content. The system uses a hierarchical agent architecture where agents are orchestrated as tools, enabling complex multi-modal reasoning tasks through collaborative problem-solving.

### Core Capabilities

- **Multi-modal Video Search**: Visual similarity (CLIP), semantic caption search (Vietnamese), and hybrid retrieval
- **Multi-agent Reasoning**: Hierarchical agent system with Greeter → Orchestrator → Planner → Workers pattern
- **Dynamic Tool Discovery**: Self-describing tools with hierarchical documentation for LLM-driven tool selection
- **Context Isolation**: Each agent maintains isolated context while sharing evidence through governed protocols
- **Streaming Workflows**: Real-time WebSocket-based workflow streaming with thinking traces
- **Evidence Persistence**: Structured evidence collection with confidence scoring and claim justification

## Architecture

```mermaid
graph TB
    subgraph Client
        User[User Query]
        Frontend[WebSocket Client]
    end

    subgraph API[FastAPI Layer]
        WS[WebSocket Handler<br/>/ws/start_workflow]
        Health[Health Check<br/>/health/*]
    end

    subgraph Workflow[Multi-Agent Workflow]
        Greeter[Greeter Agent<br/>Routing & Presentation]
        Orchestrator[Orchestrator Agent<br/>Planning & Coordination]
        Planner[Planner Agent<br/>Strategy Generation]
        Workers[Worker Agents<br/>Specialized Execution]
    end

    subgraph Tools[Tool Layer]
        subgraph SearchTools
            VisualSearch[Visual Query Search]
            CaptionSearch[Caption Search]
            EventSearch[Event Search]
            MultiModal[Multi-modal Fusion]
        end
        subgraph ViewTools
            ViewResults[View Results]
            ViewStatistics[Statistics]
            ViewEvidence[Evidence Inspector]
        end
        subgraph PersistTools
            PersistEvidence[Persist Evidence]
            UpdateContext[Update Video Context]
        end
        subgraph UtilityTools
            ASRRetrieve[ASR Transcript]
            VideoNav[Video Navigator]
            FrameExtract[Frame Extraction]
        end
        subgraph DiscoveryTools
            GroupDocs[Group Documentation]
            BundleSpecs[Bundle Specifications]
            ToolMetadata[Tool Catalog]
        end
    end

    subgraph Storage[Data Layer]
        MilvusImage[Milvus Image Vector DB]
        MilvusSegment[Milvus Segment Vector DB]
        PostgreSQL[Metadata & Artifacts]
        MinIO[Video & Frame Storage]
        ContextFS[Context File System]
    end

    subgraph External[External Services]
        EmbeddingSvc[Embedding Service<br/>CLIP + Text]
        LLMProvider[Google GenAI]
        Phoenix[Arize Phoenix<br/>Observability]
    end

    User --> Frontend
    Frontend --> WS
    WS --> Greeter

    Greeter -->|Route video search| Orchestrator
    Greeter -->|Direct response| User

    Orchestrator -->|Request plan| Planner
    Planner -->|Return strategy| Orchestrator
    Orchestrator -->|Spawn with task| Workers

    Workers --> DiscoveryTools
    Workers --> SearchTools
    Workers --> ViewTools
    Workers --> PersistTools
    Workers --> UtilityTools

    SearchTools --> MilvusImage
    SearchTools --> MilvusSegment
    ViewTools --> ContextFS
    PersistTools --> ContextFS
    UtilityTools --> PostgreSQL
    UtilityTools --> MinIO

    SearchTools --> EmbeddingSvc
    Greeter --> LLMProvider
    Orchestrator --> LLMProvider
    Planner --> LLMProvider
    Workers --> LLMProvider

    Orchestrator --> Phoenix
    Workers --> Phoenix

    Frontend -.->|Real-time events| WS
```

## Key Design Principles

### 1. Agent as Tools Pattern

**Intention**: Enable hierarchical agent orchestration where higher-level agents (Orchestrator, Greeter) can treat lower-level agents as composable tools, providing dynamic task delegation without hardcoded workflows.

**Implementation**:
- Agents are wrapped as `FunctionTool` instances with partial parameter injection
- Orchestrator spawns worker agents with specific tasks, descriptions, and execution plans
- Tool calls from agents are streamed through the workflow event system
- Worker results are collected as `WorkerResult` objects containing evidence and reasoning

**Benefits**:
- Dynamic agent creation based on planning output
- Parallel execution of multiple workers for complex tasks
- Transparent reasoning chain through streamed tool call events
- Reusable agent patterns across different workflows

### 2. Context Isolation with Governed Sharing

**Intention**: Each agent operates in isolation to prevent state pollution, while enabling controlled evidence sharing through well-defined protocols.

**Implementation**:
- `SmallWorkerContext`: Isolated context per worker agent containing:
  - Task objective and execution history
  - Local `ResultStore` for tool outputs (DataHandle objects)
  - Private evidence collection with confidence scoring
- `OrchestratorContext`: Shared orchestration context containing:
  - Session-level video context (cumulative findings per video)
  - History of all worker results across iterations
  - Chat history for continuity
- Governance through tools:
  - `worker_persist_evidence`: Workers commit evidence with justification
  - `update_video_context_orc_agent`: Orchestrator updates shared video knowledge
  - Context retrieval tools allow inspection but not mutation

**Benefits**:
- No accidental state mutations between agents
- Clear ownership of data (worker evidence vs. orchestrator synthesis)
- Audit trail of all decisions through evidence claims
- Context persistence across conversation turns

### 3. Dynamic Tool Discovery Mechanism

**Intention**: Enable LLMs to discover and select appropriate tools through a hierarchical documentation system, reducing hallucinations and improving tool selection accuracy.

**Implementation**:
- Three-level discovery hierarchy:
  1. **Group Level**: Functional categories (SEARCH_GROUP, CONTEXT_RETRIEVE_GROUP, VIEW_RESULT, PERSIST_RESULT, UTILITY)
  2. **Bundle Level**: Role-based tool groupings for specific workflows (e.g., VIDEO_EVIDENCE_WORKER_BUNDLE groups SemanticSearcher, VideoNavigator, TranscriptAnalyzer, EvidenceManager roles)
  3. **Tool Level**: Complete documentation with parameter schemas

- Discovery tools provided to Planning Agent:
  - `get_all_available_group_tools()`: High-level category overview
  - `get_all_available_bundle_spec()`: Workflow strategies and role assignments
  - (Available but not exposed by default): `list_all_tool_name()`, `get_group_documentation()`, `get_bundle_documentation()`, `all_tool_metadata()`, `get_specific_tool_documentation()`

**Benefits**:
- Planning agent discovers tools progressively without being overwhelmed
- Bundle specifications guide multi-tool workflows
- Reduces LLM hallucination of non-existent tools
- Enables self-documenting tool ecosystem

### 4. Evidence-Driven Reasoning

**Intention**: Base all reasoning and responses on structured evidence with explicit confidence scoring, avoiding hallucinations and enabling auditability.

**Implementation**:
- `EvidenceItem` structure:
  - Source worker and tool call metadata
  - Artifacts (ImageInterface or SegmentInterface) from search results
  - Confidence score (1-10 scale)
  - Claims: Textual justification of why items constitute evidence

- Evidence collection workflow:
  1. Worker executes search → receives `DataHandle` with results
  2. Worker inspects results with `worker_view_results` or `worker_view_statistics`
  3. Worker commits evidence with `worker_persist_evidence(handle_id, confidence_score, claims, slicing)`
  4. Worker finalizes with summary (currently simplified to just persisting evidence)

- Orchestrator access:
  - Retrieves worker evidence through context tools
  - Synthesizes across multiple workers
  - Updates shared video context with cross-worker findings

**Benefits**:
- Every claim is traceable to specific search results
- Confidence thresholds filter weak matches
- Enables evidence review and auditing
- Prevents hallucination by requiring source artifacts

## Technology Stack

### Core Framework
- **FastAPI**: Async web framework with WebSocket support
- **LlamaIndex Workflows**: Agent orchestration, tool calling, streaming
- **Pydantic**: Data validation, configuration management

### AI/ML
- **Google GenAI**: LLM provider for all agents (supports thinking traces)
- **External Embedding Service**: CLIP (visual + text), Vietnamese text embeddings

### Vector Search
- **Milvus**: Hybrid vector search (dense + sparse)
  - ImageMilvusClient: Visual embeddings, caption dense/sparse
  - SegmentCaptionImageMilvusClient: Segment caption embeddings
- **Hybrid Search**: Weighted ranker fusing multiple signals

### Storage
- **PostgreSQL**: Metadata, artifact lineage, ASR transcripts
- **MinIO**: S3-compatible object storage for videos, frames, JSON artifacts
- **File System**: Context persistence (`./local` directory)

### Observability
- **Arize Phoenix**: LLM tracing, tool call visualization
- **Structured Logging**: Loguru for application logs

## Project Structure

```
videodeepsearch/
├── agent/                      # Multi-agent system
│   ├── base.py                 # Agent registry, factory, configuration
│   ├── definition.py            # Pre-defined agent configurations
│   ├── prompt.py               # System prompts for each agent type
│   ├── custom.py               # Custom agent implementations
│   ├── orc_service.py          # Orchestrator workflow service
│   ├── agent_as_tool.py        # Agent-as-tools pattern implementation
│   ├── context/                # Context management
│   │   ├── management.py       # File system context persistence
│   │   ├── orc_context.py      # Orchestrator context models
│   │   └── worker_context.py   # Worker context models
│   └── README.md
│
├── api/                        # FastAPI endpoints
│   ├── stream.py              # WebSocket workflow streaming
│   ├── health.py             # Health check endpoints
│   └── README.md
│
├── core/                       # Core application logic
│   ├── app_state.py          # Singleton application state
│   ├── lifespan.py           # Application lifecycle management
│   ├── config/
│   │   ├── client_config.py  # Client configuration (Milvus, MinIO, etc.)
│   │   └── llm_config.py    # LLM configuration
│   └── README.md
│
├── tools/                      # Tool ecosystem
│   ├── base/                 # Tool infrastructure
│   │   ├── registry.py       # Tool registry and discovery
│   │   ├── schema.py         # Data schemas (ImageInterface, SegmentInterface)
│   │   ├── factory.py        # Tool factory helpers
│   │   ├── middleware/       # Input/output processing
│   │   │   ├── data_handle.py      # Result handle system
│   │   │   ├── input.py            # Input middleware
│   │   │   ├── output.py           # Output middleware
│   │   │   └── arg_doc.py         # Argument documentation
│   │   └── doc_template/    # Tool documentation templates
│   │       ├── bundle_template.py   # Bundle specifications
│   │       └── group_doc.py        # Group documentation
│   ├── clients/              # External service clients
│   │   ├── milvus/          # Milvus vector DB clients
│   │   ├── minio/           # MinIO storage client
│   │   ├── postgre/         # PostgreSQL client
│   │   └── external/        # Embedding service client
│   ├── implementation/        # Tool implementations
│   │   ├── search/          # Search tools
│   │   ├── view/            # Result viewing tools
│   │   ├── persist/         # Evidence persistence tools
│   │   ├── scan/            # Video navigation tools
│   │   ├── util/            # Utility tools (ASR, etc.)
│   │   └── llm/            # LLM-based tools
│   ├── helpers.py            # Utility functions
│   └── README.md
│
├── test/                       # Tests and notebooks
│   ├── test_agent_wf/        # Agent workflow tests
│   ├── test_client/          # Client integration tests
│   ├── test_notebook/        # Jupyter notebooks
│   ├── test_notebook2/       # Additional notebooks
│   └── README.md
│
├── main.py                     # Application entry point
├── pyproject.toml             # Python dependencies
├── docker-compose.yml           # Infrastructure services (Phoenix, PostgreSQL)
└── README.md                  # This file
```

## Quick Start

### Prerequisites
- Python 3.11+
- Docker (for PostgreSQL and Phoenix)
- Access to embedding service (configured via environment variables)

### Configuration

Required environment variables (create `.env` file):

```bash
# Milvus Image Vector DB
IMAGE_MILVUS_URI=http://localhost:19530
IMAGE_MILVUS_COLLECTION_NAME=image_collection
IMAGE_MILVUS_VISUAL_PARAM={"metric_type":"IP","params":{"nprobe":10}}
IMAGE_MILVUS_CAPTION_PARAM={"metric_type":"IP","params":{"nprobe":10}}
IMAGE_MILVUS_SPARSE_PARAM={"metric_type":"IP","params":{"drop_ratio_build":0.0}}

# Milvus Segment Vector DB
SEGMENT_MILVUS_CAPTION_URI=http://localhost:19530
SEGMENT_MILVUS_COLLECTION_NAME=segment_collection
SEGMENT_MILVUS_DENSE_PARAM={"metric_type":"IP","params":{"nprobe":10}}
SEGMENT_MILVUS_SPARSE_PARAM={"metric_type":"IP","params":{"drop_ratio_build":0.0}}

# PostgreSQL
POSTGRES_CLIENT_DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/postgres

# MinIO
MINIO_STORAGE_CLIENT_HOST=localhost
MINIO_STORAGE_CLIENT_PORT=9000
MINIO_STORAGE_CLIENT_ACCESS_KEY=minioadmin
MINIO_STORAGE_CLIENT_SECRET_KEY=minioadmin
MINIO_STORAGE_CLIENT_SECURE=false

# External Embedding Service
EXTERNAL_IMAGE_EMBEDDING_CLIENT_BASE_URL=http://localhost:8000
EXTERNAL_IMAGE_EMBEDDING_CLIENT_MODEL_NAME=clip-vit-base-patch32
EXTERNAL_IMAGE_EMBEDDING_CLIENT_DEVICE=cuda
EXTERNAL_IMAGE_EMBEDDING_CLIENT_BATCH_SIZE=32

EXTERNAL_TEXT_EMBEDDING_CLIENT_BASE_URL=http://localhost:8000
EXTERNAL_TEXT_EMBEDDING_CLIENT_MODEL_NAME=vietnamese-bge-m3
EXTERNAL_TEXT_EMBEDDING_CLIENT_DEVICE=cuda
EXTERNAL_TEXT_EMBEDDING_CLIENT_BATCH_SIZE=32

# Google GenAI (LLM)
GOOGLE_API_KEY=your_api_key_here
```

### Running the Application

1. Start infrastructure services:
```bash
docker-compose up -d
```

2. Install dependencies:
```bash
pip install -e .
```

3. Start the API server:
```bash
python main.py
```

The API will be available at `http://0.0.0.0:8050`

### Testing the Workflow

Connect via WebSocket to `ws://localhost:8050/ws/start_workflow` with the following JSON message:

```json
{
  "user_id": "test_user",
  "video_ids": ["video_123"],
  "user_demand": "Find scenes where people are having a conversation in Vietnamese",
  "chat_history": [],
  "session_id": "session_abc"
}
```

You will receive a stream of workflow events including:
- `AgentStream`: Streaming text and thinking traces
- `ToolCallResult`: Tool invocation results
- `AgentOutput`: Final agent response

## API Endpoints

### WebSocket Endpoints

#### POST `/ws/start_workflow`
Stream workflow execution events.

**Request Format**:
```json
{
  "user_id": "string",
  "video_ids": ["string"],
  "user_demand": "string",
  "chat_history": [
    {"role": "user|assistant", "content": "string"}
  ],
  "session_id": "string"
}
```

**Event Stream**:
- `AgentStream`: Delta text, thinking blocks, tool calls
- `ToolCallResult`: Tool execution results
- `AgentOutput`: Final response
- `complete`: Workflow completion signal
- `error`: Error details

### REST Endpoints

#### GET `/`
Health check returning service status.

#### GET `/health/check_tools`
List all registered search tools.

**Response**:
```json
{
  "total": 15,
  "tools": [
    "get_images_from_visual_query",
    "get_images_from_caption_query",
    "get_segments_from_event_query",
    ...
  ]
}
```

#### GET `/health/check_agents`
List all registered agents.

#### GET `/health/app_state`
Check connection status of all services.

#### GET `/health/ready`
Overall readiness check.

## Development Guide

### Adding a New Tool

1. Define the function in `tools/implementation/` (appropriate subdirectory)
2. Register using the `@tool_registry.register()` decorator
3. Specify group, bundle, and role assignments
4. Add input/output middleware if needed
5. The tool will automatically be available to appropriate agents

Example:
```python
from videodeepsearch.tools.base.registry import tool_registry
from videodeepsearch.tools.base.doc_template.group_doc import GroupName
from videodeepsearch.tools.base.doc_template.bundle_template import VIDEO_EVIDENCE_WORKER_BUNDLE
from videodeepsearch.tools.base.types import BundleRoles
from videodeepsearch.agent.definition import WORKER_AGENT

@tool_registry.register(
    group_doc_name=GroupName.SEARCH_GROUP,
    bundle_spec=VIDEO_EVIDENCE_WORKER_BUNDLE,
    bundle_role_key=BundleRoles.SEMANTIC_SEARCHER,
    output_middleware=output_image_results,
    input_middleware=None,
    belong_to_agents=[WORKER_AGENT],
    ignore_params=['list_video_id', 'user_id']
)
async def my_new_search_tool(
    query: str,
    top_k: int,
    # Auto-injected parameters
    list_video_id: list[str],
    user_id: str,
) -> list[ImageInterface]:
    """
    Search tool description for LLM.

    **When to use:**
    - Describe when this tool is appropriate

    **When NOT to use:**
    - Describe when to avoid this tool
    """
    # Implementation
    return results
```

### Adding a New Agent

1. Define agent configuration in `agent/definition.py`:
```python
@register_agent('MY_NEW_AGENT')
def create_my_agent(
    llm: FunctionCallingLLM,
    tools: list[FunctionTool]
) -> AgentConfig:
    config = AgentConfig(
        name='MY_NEW_AGENT',
        description='Agent description',
        system_prompt='System prompt',
        llm=llm,
        tools=tools,
        type_of_agent=FunctionAgent,
        streaming=True
    )
    return config
```

2. Use the agent via the registry:
```python
registry = get_global_agent_registry()
agent = registry.spawn(
    name='MY_NEW_AGENT',
    llm=get_llm_instance('MY_NEW_AGENT'),
    tools=my_tools
)
```

### Tool Discovery Debugging

When developing tools, you can expose additional discovery functions by modifying `get_registry_tools()` in `tools/base/registry.py`:

```python
def get_registry_tools() -> list[FunctionTool]:
    # Available for Planning Agent
    registry_discovery_fns = [
        tool_registry.get_all_available_group_tools,
        tool_registry.get_all_available_bundle_spec,
        # Uncomment for debugging:
        # tool_registry.list_all_tool_name,
        # tool_registry.all_tool_metadata,
        # tool_registry.get_specific_tool_documentation,
    ]
    return [FunctionTool.from_defaults(fn=f) for f in registry_discovery_fns]
```

### Context Persistence

Context is automatically persisted to `./local/<session_id>.json` after each orchestrator run. To manually inspect or modify:

```python
from videodeepsearch.agent.context.management import FileSystemContextStore
from llama_index.core.workflow import Context, JsonPickleSerializer

# Load context
store = FileSystemContextStore(storage_dir='./local')
context_dict = store.load_context(session_id='session_abc')

# Restore into Context object
ctx = Context.from_dict(workflow=agent, data=context_dict)

# Save modified context
store.save_context(session_id='session_abc', context_model=ctx)
```

## Observability

### Phoenix Dashboard

Access Phoenix at `http://localhost:6006` to view:
- LLM traces with thinking blocks
- Tool call hierarchies
- Agent execution timelines
- Error tracking

### Structured Logging

Logs are captured by Phoenix and written to console. Key log events:
- Context creation/loading
- Tool registration
- Agent spawning
- Workflow events

## Troubleshooting

### Common Issues

**Issue**: Tools not available to agent
- **Solution**: Verify tool is registered with correct `belong_to_agents` and bundle role

**Issue**: Context not persisting
- **Solution**: Ensure `./local` directory exists and is writable

**Issue**: Milvus connection timeout
- **Solution**: Check `IMAGE_MILVUS_URI` and `SEGMENT_MILVUS_CAPTION_URI` environment variables

**Issue**: LLM API errors
- **Solution**: Verify `GOOGLE_API_KEY` is set and valid

## License

[Your License Here]

## Contributing

[Contributing Guidelines]
