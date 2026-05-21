# Image Caption Embedding Task

## Purpose

`image_caption_embedding_chunk_task` embeds generated image caption text with mmBERT. These text-only embeddings support caption semantic search and hybrid dense/sparse Qdrant indexing.

## How It Works

The task implementation is `ImageCaptionEmbeddingTask` in `main.py`.

1. `preprocess()` extracts `metadata.caption` from each `ImageCaptionArtifact`.
2. `execute()` sends all caption strings in the batch to `MMBertClient.ainfer()` in one request.
3. `execute()` creates one `TextCaptionEmbeddingArtifact` per caption and serializes each vector to `.npy` bytes.
4. `postprocess()` uploads `.npy` files to MinIO and persists artifacts.
5. `summary_artifact()` records the number of embeddings and average dimension.

## Input

Type: `list[ImageCaptionArtifact]`

| Field | Meaning |
|-------|---------|
| `metadata.caption` | Caption text to embed. |
| `frame_index` | Source frame. |
| `image_id` | Source `ImageArtifact` ID. |
| `image_minio_url` | Source image URL. |

## Output

Type: `list[TextCaptionEmbeddingArtifact]`

| Field | Meaning |
|-------|---------|
| `caption_id` | Source caption artifact ID. |
| `object_name` | MinIO path under `embedding/image_caption/{video_id}/...npy`. |
| `metadata.embedding_dim` | Dense vector dimension. |
| `metadata.caption` | Embedded text. |

## Dask Parallelization

The flow maps this task with `image_caption_embedding_chunk_task.map(caption_embedding_batches, wait_for=[caption_batch_futures])`. Dask parallelizes across batches. Each mapped task uses one batched mmBERT call for its batch.

## Algorithm Details

The task is a dense text embedding stage. It does not generate sparse vectors itself. Sparse SPLADE vectors are generated later during Qdrant indexing so the raw text can be indexed in both dense and sparse forms.

## Downstream Consumers

`image_caption_qdrant_indexing_chunk_task` indexes these caption embeddings into a hybrid dense+sparse Qdrant collection.
