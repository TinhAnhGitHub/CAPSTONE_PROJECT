# Segment Embedding Task

## Purpose

`segment_embedding_chunk_task` creates multimodal embeddings for captioned video segments. Each embedding combines sampled frames from the segment and the segment summary caption.

## How It Works

The task implementation is `SegmentEmbeddingTask` in `main.py`.

1. `preprocess()` receives `SegmentCaptionArtifact` objects and returns them unchanged.
2. `execute()` streams the source video once for the batch.
3. For each segment, it samples `n_frames` frame indices between `start_frame` and `end_frame`.
4. It reads frame bytes, converts them to resized JPEG bytes, and sends them with `summary_caption` to `QwenVLEmbeddingClient.ainfer_video()`.
5. It creates a `SegmentEmbeddingArtifact` with embedding metadata and frame indices.
6. `postprocess()` writes the vector as `.npy` bytes to MinIO and persists the artifact.

## Input

Type: `list[SegmentCaptionArtifact]`

| Field | Meaning |
|-------|---------|
| `summary_caption` | Text paired with segment frames. |
| `start_frame` and `end_frame` | Frame range for sampling. |
| `related_video_minio_url` | Source video. |
| `related_video_fps` | Timing metadata. |

## Output

Type: `list[SegmentEmbeddingArtifact]`

| Field | Meaning |
|-------|---------|
| `caption_text` | Summary caption used during embedding. |
| `frame_indices` | Frames sampled for embedding. |
| `embedding_dim` | Dense vector dimension. |
| `object_name` | MinIO path under `embedding/segment/{video_id}/...npy`. |
| `metadata.embedding` | Raw embedding before postprocess serialization. |

## Dask Parallelization

The flow maps this task with `segment_embedding_chunk_task.map(segment_caption_batches)`. Dask parallelizes across batches. Inside each task, segment embeddings are processed sequentially because each segment needs multiple frame reads and one video embedding request.

## Algorithm Details

The task builds a video-level representation for each temporal segment:

1. Use the captioned segment frame bounds, not the raw shot bounds.
2. Sample a fixed number of frames to represent the temporal interval.
3. Normalize all frames to JPEG/RGB input for the embedding model.
4. Send sampled frames plus summary caption so the vector captures both visual and language context.
5. Persist the vector separately from metadata to keep artifact records lightweight.

## Downstream Consumers

`segment_qdrant_indexing_chunk_task` indexes these dense segment vectors into Qdrant.
