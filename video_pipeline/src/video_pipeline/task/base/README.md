# Base Task Infrastructure

## Purpose

The `base` package is not a processing stage. It defines the shared task lifecycle, YAML-backed Prefect task configuration, and cache-key functions used by the concrete tasks under `video_pipeline.task`.

## Main Components

| File | Purpose |
|------|---------|
| `base_task.py` | Defines `TaskConfig` and `BaseTask`. |
| `cache_keys.py` | Defines deterministic Prefect cache key functions for pipeline artifacts and batches. |

## `TaskConfig`

`TaskConfig.from_yaml(task_name)` loads a task entry from `src/video_pipeline/config/tasks.yaml`. The fields include task name, description, stage, tags, retries, timeout, cache settings, and task-specific `additional_kwargs`.

`to_task_kwargs()` converts the config into arguments for `@prefect.task`, including retry settings, timeout, tags, and cache policy. If `cache_enabled` is true and `cache_key_fn` names a function from `CACHE_KEY_FUNCTIONS`, that function is used as the Prefect cache key.

## `BaseTask`

Every concrete task implements the same lifecycle:

```python
preprocess(input_data) -> prepared input
execute(preprocessed, client) -> raw result
postprocess(result) -> final artifact/result
summary_artifact(final_result) -> Prefect summary
```

The helper `execute_template()` runs the lifecycle in order, logs critical failures, calls `on_task_failed()` on error, and returns the final artifact.

## Inputs And Outputs

`BaseTask` is generic over input and output types. Concrete tasks define the real artifact types, for example `BaseTask[VideoInput, VideoArtifact]` or `BaseTask[list[ImageCaptionArtifact], list[ImageEmbeddingArtifact]]`.

## Dask Parallelization

`BaseTask` itself does not create Dask workers. Dask parallelization happens at the Prefect flow layer through `DaskTaskRunner` and task calls such as `.submit()` and `.map()`. The concrete Prefect wrapper functions instantiate a `BaseTask` subclass inside the Dask worker process and then call `execute_template()`.

## Algorithm Details

The base abstraction keeps route/orchestration code separate from processing logic:

1. Prefect wrapper function creates external clients such as MinIO, PostgreSQL, Qdrant, OpenRouter, or model clients.
2. Wrapper constructs the concrete `BaseTask` subclass with shared clients and extra context.
3. `execute_template()` runs the standardized lifecycle.
4. Concrete `postprocess()` methods use `ArtifactPersistentVisitor` to persist metadata and optional binary payloads.

This structure makes task behavior consistent while allowing each stage to define its own algorithm.
