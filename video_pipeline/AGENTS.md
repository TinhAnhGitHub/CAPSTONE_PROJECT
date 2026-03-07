# AGENTS.md

Guidelines for agentic coding agents working in the video_pipeline repository.

## Build / Lint / Test Commands

```bash
# Install (editable mode)
pip install -e .
pip install -e ".[dev]"          # with dev dependencies
uv pip install -e ".[dev]"       # if using uv

# Install worker dependencies (ML, video processing)
pip install -e ".[dev,worker]"

# Lint & format
ruff check src/
ruff format src/

# Run all tests
pytest

# Run single test file
pytest tests/path/to/test_file.py

# Run single test function
pytest tests/path/to/test_file.py::test_function_name

# Run with coverage
pytest --cov=video_pipeline

# Run API server
video-pipeline-api                                    # via pyproject.toml script
uvicorn video_pipeline.api.app:app --reload --port 8050   # dev mode

# Infrastructure (from project root)
docker compose up -d          # start postgres, redis, prefect, minio, qdrant
docker compose down

# Prefect deployment
prefect deploy                # uses prefect.yaml
prefect worker start --pool poc-pool
```

## Environment Variables

Required in `.env` at project root:
- `APP_ENV` — environment name (dev, staging, prod)
- `PREFECT_API_URL` — Prefect server URL
- `OPENROUTER_API_KEY` — API key for VLM captioning

## Architecture Overview

This is a **Prefect 3 video-processing pipeline** that ingests a single video and fans it out through parallel ML analysis stages. Every result is persisted as a typed artifact to MinIO (binary) and PostgreSQL (metadata).

### Directory Structure

```
src/video_pipeline/
├── api/                    # FastAPI trigger layer (fire-and-forget)
│   ├── app.py              # FastAPI instance, middleware, entry point
│   ├── lifespan.py         # startup: verify deployment exists
│   └── routers/
│       ├── upload.py       # POST /api/uploads → run_deployment(..., timeout=0)
│       └── health.py       # GET /api/health, /api/health/prefect
├── flow/
│   ├── main.py             # full pipeline: video → autoshot → ASR + images → caption/embed/OCR → Qdrant
│   ├── submain.py          # POC variant: stops after ASR + image extraction
│   └── subtask.py          # preprocess_video_task (audio + frame index extraction)
├── task/
│   ├── base/base_task.py   # BaseTask[InputT, OutputT] template method
│   ├── video/              # VideoRegistryTask: extract FPS/duration/extension
│   ├── autoshot/           # AutoshotTask: ML scene-boundary detection (Triton)
│   ├── asr/                # ASRTask: Qwen3-ASR speech transcription
│   ├── image_extraction/   # ImageExtractionTask: frame extraction to WebP
│   ├── image_caption/      # ImageCaptionTask: VLM captioning (OpenRouter)
│   ├── image_embedding/    # ImageEmbeddingTask: visual embeddings (QwenVL)
│   ├── image_ocr/          # ImageOCRTask: OCR (LightON)
│   ├── image_caption_embedding/        # text embeddings (mmBERT)
│   ├── image_caption_multimodal_embedding/  # multimodal embeddings (QwenVL)
│   └── qdrant_indexing/    # upsert embeddings to Qdrant
├── core/
│   ├── artifact/artifact.py    # BaseArtifact + all artifact types
│   ├── storage/pg_tracker.py   # ArtifactPersistentVisitor (MinIO + Postgres)
│   ├── client/
│   │   ├── storage/           # MinioStorageClient, PostgresClient, QdrantClient
│   │   ├── inference/         # AutoShotClient, ASRClient, OCRClient, etc.
│   │   ├── llm_provider/      # OpenRouterClient, GeminiClient, MoondreamClient
│   │   └── progress/          # HTTPProgressTracker, StageRegistry
│   └── state.py               # shared state utilities
└── config/
    ├── settings.py            # Pydantic Settings singleton (get_settings())
    ├── tasks.yaml             # task configurations (retries, timeouts, model names)
    └── environments/          # {dev,staging,prod}.yaml
```

### Pipeline Stages

```
VideoRegistryTask          → extracts FPS / duration / extension
AutoshotTask               → ML scene-boundary detection (Triton)
preprocess_video_task      → slices audio chunks + frame indices per segment
    ├─ asr_chunk_task      → Qwen3-ASR speech transcription
    └─ image_chunk_task    → frame extraction to WebP
        ├─ image_caption_chunk_task          → VLM captioning (OpenRouter)
        ├─ image_embedding_chunk_task        → visual embeddings (QwenVL)
        └─ image_ocr_chunk_task              → OCR (LightON)
            ├─ image_caption_embedding_chunk_task       → text embeddings
            └─ image_caption_multimodal_embedding_chunk_task → multimodal embeddings
                └─ qdrant_indexing_task → upsert to vector DB
```

## Code Style Guidelines

### Imports

```python
# Standard library first
from __future__ import annotations
from pathlib import Path
from typing import Any, TypeVar

# Third-party next
from prefect import get_run_logger, task
from pydantic import BaseModel, Field

# Local imports last (absolute)
from video_pipeline.config import get_settings
from video_pipeline.core.artifact import ImageArtifact
from video_pipeline.task.base.base_task import BaseTask, TaskConfig
```

### Type Annotations

- Use Python 3.12+ syntax: `list[str]`, `dict[str, Any]`, `int | None`
- Generic classes: `class ImageCaptionTask(BaseTask[list[ImageArtifact], list[ImageCaptionArtifact]]):`
- Use `TYPE_CHECKING` for forward references to avoid circular imports

### Naming Conventions

- **Tasks**: `{Name}Task` class + `{name}_task` or `{name}_chunk_task` Prefect task function
- **Artifacts**: `{Name}Artifact` Pydantic model (e.g., `ImageCaptionArtifact`)
- **Configs**: `{Name}Config` Pydantic model (e.g., `OpenRouterConfig`)
- **Clients**: `{Name}Client` class (e.g., `MinioStorageClient`)
- **Task config YAML keys**: `snake_case` (e.g., `image_caption`)

### BaseTask Template Method Pattern

Every task inherits from `BaseTask[InputT, OutputT]`:

```python
@StageRegistry.register  # registers for progress tracking
class MyTask(BaseTask[InputType, OutputType]):
    config = TaskConfig.from_yaml("my_task")  # from tasks.yaml

    async def preprocess(self, input_data: InputType) -> Any:
        """Validate, load data, split into batches."""
        ...

    async def execute(self, preprocessed: Any, client: Any) -> Any:
        """Core processing logic. Use the client for external services."""
        ...

    async def postprocess(self, result: Any) -> OutputType:
        """Create artifacts, persist to storage."""
        ...

    @staticmethod
    async def summary_artifact(final_result: OutputType) -> None:
        """Create Prefect markdown + table artifacts for UI."""
        await acreate_markdown_artifact(...)
        await acreate_table_artifact(...)
```

The `execute_template()` method orchestrates: `preprocess → execute → postprocess`.

### Task Configuration

Load from `tasks.yaml`:

```python
MY_TASK_CONFIG = TaskConfig.from_yaml("task_key")
_base_kwargs = MY_TASK_CONFIG.to_task_kwargs()

@task(**_base_kwargs)
async def my_task(...) -> ...:
    model = MY_TASK_CONFIG.additional_kwargs["model"]
    batch_size = MY_TASK_CONFIG.additional_kwargs.get("batch_size", 10)
```

### Artifact Persistence

Use `ArtifactPersistentVisitor` to persist artifacts:

```python
# In task postprocess or module-level task function:
await self.artifact_visitor.visit_artifact(
    artifact,
    upload_to_minio=io.BytesIO(json_bytes)  # optional: upload binary
)
```

### Settings Singleton

```python
from video_pipeline.config import get_settings

settings = get_settings()
minio_endpoint = settings.minio.endpoint
postgres_url = settings.postgres.connection_string
```

### Error Handling

- Let exceptions propagate for critical failures (Prefect handles retries)
- Use `logger.exception()` for logging with stack trace
- Use `assert` for internal invariants (e.g., `assert artifact.object_name is not None`)

### Prefect Task Functions

```python
@task(**CONFIG.to_task_kwargs())  # type: ignore
async def my_chunk_task(items: list[InputArtifact]) -> list[OutputArtifact]:
    logger = get_run_logger()
    settings = get_settings()

    # Initialize clients
    minio_client = MinioStorageClient(...)
    postgres_client = PostgresClient(config=PgConfig(...))

    task_impl = MyTask(
        artifact_visitor=ArtifactPersistentVisitor(minio_client, postgres_client),
        minio_client=minio_client,
    )

    async with ServiceClient(...) as client:
        artifacts = await task_impl.execute_template(items, client)

    await MyTask.summary_artifact(artifacts)
    return artifacts
```

### Flow Pattern

```python
@flow(task_runner=DaskTaskRunner(...))
async def my_flow(...):
    # Sequential with .submit() + .result()
    video_fut = video_reg_task.submit(video_input)
    video_artifact = video_fut.result()

    # Fan-out with .map()
    batch_futures = some_task.map(batches, wait_for=[upstream_fut])
    results = batch_futures.result()

    # Flatten results
    all_artifacts = [a for batch in results for a in batch]
```

### Ruff Configuration

From `pyproject.toml`:
- Line length: 100
- Target: Python 3.12
- Enabled rules: E, F, W (correctness), I (imports), UP (pyupgrade), B (bugbear), SIM (simplifications), C4 (comprehensions), ANN (type annotations)

## Key Patterns

1. **API layer never imports flow/task code** — uses `prefect.deployments.run_deployment` with `timeout=0`
2. **Progress tracking** — decorate tasks with `@StageRegistry.register`, use `HTTPProgressTracker` in flows
3. **Artifact lineage** — each artifact tracks `lineage_parents` via `_build_lineage_parents()`
4. **Client context managers** — use `async with Client(...) as client:` for resource cleanup
5. **Batch processing** — tasks receive batches, use `.map()` for parallelism, control concurrency via Prefect work-pool limits