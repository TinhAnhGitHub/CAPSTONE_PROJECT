# Image Caption OCR Task

## Purpose

`image_caption_ocr_chunk_task` generates both a visual caption and OCR text for each extracted image. It uses one structured VLM call per image and emits both `ImageCaptionArtifact` and `ImageOCRArtifact` records.

## How It Works

The task implementation is `ImageCaptionOCRTask` in `main.py`.

1. `preprocess()` downloads image bytes from MinIO for every `ImageArtifact` in the batch.
2. `execute()` creates a structured OpenRouter LLM client for the `ImageCaptionOCR` schema.
3. `execute()` encodes each image as a data URL and sends it with `CAPTION_OCR_PROMPT`.
4. `execute()` runs image calls concurrently using `asyncio.gather()` bounded by `max_concurrent`.
5. For each result, it builds an `ImageCaptionArtifact` and `ImageOCRArtifact` plus JSON payload bytes.
6. `postprocess()` uploads both JSON payloads to MinIO and persists both artifacts.
7. `summary_artifact()` records frames processed, OCR coverage, token usage, and cost.

## Input

Type: `list[ImageArtifact]`

| Field | Meaning |
|-------|---------|
| `object_name` | MinIO object for image bytes. |
| `frame_index` | Frame number. |
| `timestamp` and `timestamp_sec` | Time alignment. |
| `related_video_id` | Source video. |
| `image_minio_url` | Source image URL. |

## Output

Type: `tuple[list[ImageCaptionArtifact], list[ImageOCRArtifact], CostTracker]`

| Output | Meaning |
|--------|---------|
| `ImageCaptionArtifact` list | JSON caption artifacts under `caption/image/{video_id}/...json`. |
| `ImageOCRArtifact` list | JSON OCR artifacts under `ocr/image/{video_id}/...json`. |
| `CostTracker` | LLM usage and cost for the batch. |

## Dask Parallelization

The flow maps this task with `image_caption_ocr_chunk_task.map(analysis_batches, wait_for=[image_batch_futures])`. Dask parallelizes across batches. Inside each batch, the task also parallelizes VLM calls with `asyncio.gather()` and a semaphore using `max_concurrent` from `tasks.yaml`.

## Algorithm Details

The task uses a combined caption + OCR prompt to avoid two separate image model calls. The structured response contains a caption string and a list of OCR text spans. OCR spans are joined into a single `ocr_text` for indexing while preserving the original list in metadata. If a single image call fails, the task emits empty caption/OCR content for that frame instead of failing the whole batch.

## Downstream Consumers

Image captions feed image embedding and image-caption text embedding. OCR artifacts feed Elasticsearch OCR indexing.
