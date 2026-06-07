# Video Registration Task

## Purpose

`video_reg_task` registers one source video before any downstream processing runs. It normalizes the source S3/MinIO URL, extracts basic media metadata, creates a `VideoArtifact`, and persists that artifact to PostgreSQL through `ArtifactPersistentVisitor`.

## How It Works

The task implementation is `VideoRegistryTask` in `main.py`.

1. `preprocess()` receives a `VideoInput` and returns it unchanged.
2. `execute()` parses the video S3 URL into bucket and object name.
3. `execute()` downloads the video to a temporary local file with `MinioStorageClient.fetch_object_from_s3()`.
4. `execute()` reads FPS with `get_video_fps()` and duration with `get_video_duration_ffprobe()`.
5. `execute()` builds a `VideoArtifact` containing video ID, user ID, MinIO URL, FPS, extension, object name, and metadata.
6. `postprocess()` persists the artifact metadata.
7. `summary_artifact()` writes Prefect markdown and table summaries.

## Input

Type: `VideoInput`

| Field | Meaning |
|-------|---------|
| `user_id` | Owner namespace and storage bucket identity. |
| `video_s3_url` | Source video URL such as `s3://bucket/path/video.mp4`. |
| `video_id` | Stable ID used by downstream artifacts. |

## Output

Type: `VideoArtifact`

| Field | Meaning |
|-------|---------|
| `artifact_id` | Uses the provided `video_id`. |
| `video_minio_url` | Normalized `s3://bucket/object` URL. |
| `video_extension` | File extension extracted from the source URL. |
| `fps` | Video frames per second. |
| `object_name` | Object path inside the MinIO bucket. |
| `metadata.duration` | Duration in seconds from ffprobe. |

## Dask Parallelization

This is a single-video setup task, so it is submitted once with `video_reg_task.submit()` from the flow. It still runs on the flow's `DaskTaskRunner`, but it is not mapped across batches because all downstream fan-out depends on this single artifact.

## Algorithm Details

The task is intentionally deterministic and metadata-only. No model inference is used. The main algorithm is URL parsing plus media probing:

1. Parse the URL into storage location.
2. Download the object to a temporary file with the correct suffix.
3. Run video probing helpers against the local file.
4. Convert the metadata into a typed artifact for lineage tracking.

## Downstream Consumers

`AutoshotTask` consumes `VideoArtifact` to download the same video, extract frames, and detect shot boundaries.
