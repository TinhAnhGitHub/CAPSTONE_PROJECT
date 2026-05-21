# Image Extraction Task

## Purpose

`image_chunk_task` extracts representative video frames selected during preprocessing and persists them as `ImageArtifact` records with WebP bytes stored in MinIO.

## How It Works

The task implementation is `ImageExtractionTask` in `main.py`.

1. `preprocess()` receives a batch of `(AutoshotArtifact, frame_index)` items.
2. `preprocess()` downloads the source video once for the batch with `fetch_object_streaming()`.
3. `preprocess()` uses `FastFrameReader` to decode each requested frame into bytes.
4. `execute()` creates one `ImageArtifact` per frame.
5. `postprocess()` uploads frame bytes to MinIO and persists the artifact.
6. `summary_artifact()` writes extracted frame metadata to Prefect.

## Input

Type: `list[ImageItem]`

`ImageItem` is `tuple[AutoshotArtifact, frame_index]`.

| Tuple item | Meaning |
|------------|---------|
| `AutoshotArtifact` | Provides video URL, FPS, extension, video ID, and user ID. |
| `frame_index` | Exact frame number to extract. |

## Output

Type: `list[ImageArtifact]`

| Field | Meaning |
|-------|---------|
| `frame_index` | Extracted frame number. |
| `timestamp` | Human-readable timestamp derived from FPS. |
| `timestamp_sec` | Numeric timestamp in seconds. |
| `object_name` | MinIO path under `images/{video_id}/...webp`. |
| `minio_url_path` | Stored image URL. |
| `autoshot_artifact_id` | Source shot artifact ID. |

## Dask Parallelization

The flow maps image extraction batches with `image_chunk_task.map(image_batches, wait_for=[preprocess_fut])`. Each Dask task downloads the video once for its assigned frame batch, then extracts frames sequentially inside that batch. Batch size is configured by `video_preprocess.image_batch_size`.

## Algorithm Details

Representative frame indices are computed before this task by `get_segment_frame_indices()`. This task focuses on efficient decoding:

1. Group many frame requests into one task to avoid one video download per frame.
2. Open the video once with `FastFrameReader`.
3. Seek/read requested frames by frame index.
4. Store image bytes as WebP and track both frame index and timestamp for later retrieval alignment.

## Downstream Consumers

`image_caption_ocr_chunk_task` consumes `ImageArtifact` objects to generate captions and OCR text.
