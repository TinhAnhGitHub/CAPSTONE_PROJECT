# Autoshot Task

## Purpose

`autoshot_task` detects shot boundaries in the registered video. It converts the video into frames, sends frame windows to the AutoShot model, converts model predictions into `[start_frame, end_frame]` segments, and persists an `AutoshotArtifact`.

## How It Works

The task implementation is `AutoshotTask` in `main.py`.

1. `preprocess()` streams the video from MinIO to a local temporary file.
2. `preprocess()` extracts decoded frames with `get_frames_fast()`.
3. `execute()` splits the frame array into model batches with `get_batches()`.
4. `execute()` calls `AutoShotClient.ainfer()` for each batch.
5. `execute()` concatenates batch predictions and trims them to the original frame count.
6. `postprocess()` converts binary boundary predictions into scenes with `predictions_to_scenes()`.
7. `postprocess()` splits outlier scenes, enforces a minimum segment count when configured, and persists the artifact.
8. The wrapper removes the temporary video file in `finally`.

## Input

Type: `VideoArtifact`

| Field | Meaning |
|-------|---------|
| `video_id` | Source video identity. |
| `video_minio_url` | Video object location in MinIO. |
| `video_extension` | Used for the temporary file suffix. |
| `fps` | Needed by downstream timestamp conversion. |
| `user_id` | Artifact owner and bucket namespace. |

## Output

Type: `AutoshotArtifact`

| Field | Meaning |
|-------|---------|
| `related_video_id` | Source video ID. |
| `related_video_minio_url` | Source video URL. |
| `related_video_fps` | FPS copied from `VideoArtifact`. |
| `metadata.segments` | List of shot intervals as `[start_frame, end_frame]`. |

## Dask Parallelization

The flow submits this task once with `autoshot_task.submit()`, so Dask schedules the whole stage on a Dask worker but does not map it across multiple videos or segments. Inside the task, model calls are parallelized with `asyncio.gather()` and bounded by `MAX_CONCURRENT_BATCHES = 4`, so up to four AutoShot inference requests run concurrently for one video.

## Algorithm Details

The model operates on fixed-length frame windows. `get_batches()` creates those windows and pads/overlaps frames as needed. Each batch is converted by `preprocess_input_client()` into the tensor layout expected by Triton. The model returns boundary logits or probabilities, and `postprocess_output_client()` converts them into per-frame predictions. `predictions_to_scenes()` then scans boundary positions and emits contiguous frame intervals. `split_outlier_scenes()` prevents very long intervals from dominating later stages, and `enforce_min_scene_segments()` can split coarse results so later ASR/image/KG stages have enough temporal granularity.

## Downstream Consumers

`preprocess_video_task` uses `metadata.segments` to create ASR audio chunks and representative frame indices for image extraction.
