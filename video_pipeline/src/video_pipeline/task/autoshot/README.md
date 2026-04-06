# Autoshot Task Documentation

## Overview

The Autoshot task detects scene boundaries (shots) in video content using a deep learning model served via Triton Inference Server.

## Function

Detects shot boundaries in videos by:
1. Extracting frames from video (resized to 48x27 for efficiency)
2. Processing frames in overlapping batches (100 frames, stride 50)
3. Predicting shot boundaries using a neural network
4. Converting predictions to scene segments (start frame, end frame)

## Client

**Client:** `AutoShotClient` (`video_pipeline.core.client.inference.autoshot_client`)

**Configuration:** `AutoshotConfig`

**Protocol:** Triton Inference Server (gRPC/HTTP)

**Input Format:**
- Batch of 100 frames, shape: `(1, 3, 100, 27, 48)` (NCHW format)

**Output Format:**
- Binary predictions per frame (shot boundary: 0 or 1)

## Prefect Integration

### Task Definition

```python
@task(
    name="Autoshot Segment Detection",
    description="Detect scene boundaries (shots) in video",
    tags=['autoshot-tags', 'gpus-limit'],
    retries=2,
    retry_delay_seconds=5,
    timeout_seconds=30,
    cache_policy=INPUTS,
    task_runner=DaskTaskRunner(
        cluster_kwargs={
            "n_workers": 1,
            "threads_per_worker": 4,
            "processes": True
        }
    )
)
async def autoshot_task(
    video_artifact: VideoArtifact,
    context: TaskExecutionContext
) -> AsyncGenerator[AutoshotArtifact, None]
```

### Task Runner

Uses `DaskTaskRunner` with:
- 1 worker process
- 4 threads per worker
- Process-based execution

### Prefect Variables Required

| Variable | Description |
|----------|-------------|
| `task_init_kwargs` | Initialization kwargs (artifact_visitor, minio_client) |
| `autoshot_client_kwargs` | AutoShotClient configuration |

## Input/Output Artifacts

### Input

**Type:** `VideoArtifact`

| Field | Description |
|-------|-------------|
| `video_id` | Unique video identifier |
| `video_minio_url` | MinIO URL to video file |
| `fps` | Video frames per second |
| `user_id` | Owner user ID |

### Output

**Type:** `AutoshotArtifact`

| Field | Description |
|-------|-------------|
| `related_video_id` | Source video ID |
| `related_video_fps` | Source video FPS |
| `metadata.segments` | List of `[start_frame, end_frame]` pairs |

## Task Lifecycle

```
preprocess()          → Extract frames from video via ffmpeg
    ↓
execute_single()      → Batch frames → AutoShotClient.infer() → predictions
    ↓
postprocess()         → Convert predictions to scene segments
    ↓
yield AutoshotArtifact
```

## Helper Functions

| Function | Description |
|----------|-------------|
| `get_frames_fast()` | Extract resized frames using ffmpeg |
| `get_batches()` | Create overlapping batches with padding |
| `predictions_to_scenes()` | Convert binary predictions to segment boundaries |
| `preprocess_input_client()` | Reshape batch for Triton (NHWC → NCHW) |
| `postprocess_output_client()` | Apply sigmoid and extract valid predictions |

## Error Handling

**Exception:** `AutoshotClientError`

Raised when:
- Triton server returns `None` (input mismatch or server error)

## Example Usage

```python
from video_pipeline.task.autoshot import autoshot_task
from video_pipeline.core.artifact import VideoArtifact

video = VideoArtifact(
    video_id="abc123",
    video_minio_url="minio://videos/abc123.mp4",
    fps=30.0,
    user_id="user1"
)

async for artifact in autoshot_task(video, context):
    print(f"Detected {len(artifact.metadata['segments'])} scenes")
```
