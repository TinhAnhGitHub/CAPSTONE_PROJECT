# Image Embedding Task

## Purpose

`image_embedding_chunk_task` creates multimodal image embeddings from image bytes and generated captions. The result is an `ImageEmbeddingArtifact` with a `.npy` vector stored in MinIO.

## How It Works

The task implementation is `ImageEmbeddingTask` in `main.py`.

1. `preprocess()` receives `ImageCaptionArtifact` objects.
2. `preprocess()` uses each caption artifact's `image_minio_url` to download the original image bytes.
3. `execute()` converts each image to resized JPEG bytes.
4. `execute()` sends image bytes and caption text to `QwenVLEmbeddingClient._infer_single_image()`.
5. `execute()` builds an `ImageEmbeddingArtifact` and serializes the vector as `.npy` bytes.
6. `postprocess()` uploads `.npy` bytes to MinIO and persists metadata.

## Input

Type: `list[ImageCaptionArtifact]`

| Field | Meaning |
|-------|---------|
| `metadata.caption` | Caption text paired with the image. |
| `image_minio_url` | Source image bytes. |
| `frame_index` | Frame number. |
| `timestamp_sec` | Time alignment for retrieval filters. |

## Output

Type: `list[ImageEmbeddingArtifact]`

| Field | Meaning |
|-------|---------|
| `caption_text` | Caption embedded with the image. |
| `object_name` | MinIO path under `embedding/image/{video_id}/...npy`. |
| `metadata.embedding_dim` | Vector dimensionality. |
| `image_minio_url` | Source image URL. |
| `timestamp_sec` | Retrieval timestamp. |

## Dask Parallelization

The flow maps this task with `image_embedding_chunk_task.map(caption_batches, wait_for=[caption_ocr_batch_futures])`. Dask parallelizes across caption batches. Inside a batch, embeddings are requested concurrently with `asyncio.gather()`.

## Algorithm Details

The algorithm creates a joint image-text representation rather than image-only vectors:

1. Load the original frame bytes.
2. Convert to RGB JPEG with fixed size to normalize model input.
3. Read the generated caption from artifact metadata.
4. Send `(image, caption)` to the QwenVL embedding service.
5. Store the dense vector as `float32` `.npy` bytes for later Qdrant ingestion.

## Downstream Consumers

`image_qdrant_indexing_chunk_task` indexes these vectors into the image Qdrant collection.
