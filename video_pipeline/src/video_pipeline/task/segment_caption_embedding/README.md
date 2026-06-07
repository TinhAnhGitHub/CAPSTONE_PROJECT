# Segment Caption Embedding Task

## Purpose

`segment_caption_embedding_chunk_task` embeds segment summary captions with mmBERT. These text-only vectors support semantic retrieval over segment descriptions and hybrid Qdrant search.

## How It Works

The task implementation is `SegmentCaptionEmbeddingTask` in `main.py`.

1. `preprocess()` extracts `summary_caption` from each `SegmentCaptionArtifact`.
2. `execute()` delegates to `execute_single()`.
3. `execute_single()` sends all captions in the batch to `MMBertClient.ainfer()`.
4. It creates one `TextCapSegmentEmbedArtifact` per caption.
5. It serializes each dense vector as `.npy` bytes.
6. `postprocess()` uploads embeddings to MinIO and persists artifacts.

## Input

Type: `list[SegmentCaptionArtifact]`

| Field | Meaning |
|-------|---------|
| `summary_caption` | Text to embed. |
| `start_frame` and `end_frame` | Segment frame range. |
| `start_sec` and `end_sec` | Segment time range. |
| `artifact_id` | Source segment caption ID. |

## Output

Type: `list[TextCapSegmentEmbedArtifact]`

| Field | Meaning |
|-------|---------|
| `segment_cap_id` | Source `SegmentCaptionArtifact` ID. |
| `related_segment_caption_url` | Source caption artifact MinIO URL. |
| `object_name` | MinIO path under `embedding/caption_segment/{video_id}/...npy`. |
| `metadata.embedding_dim` | Dense vector dimension. |
| `metadata.summary_caption` | Embedded text. |

## Dask Parallelization

The flow maps this task with `segment_caption_embedding_chunk_task.map(segment_caption_batches, wait_for=segment_caption_futures)`. Dask parallelizes across caption batches. Each mapped task uses one batched mmBERT call.

## Algorithm Details

This is the text-only complement to multimodal segment embedding. It embeds only `summary_caption`, leaving sparse SPLADE encoding to the later Qdrant indexing task. This lets retrieval use dense semantic similarity, sparse lexical matching, or both.

## Downstream Consumers

`segment_caption_qdrant_indexing_chunk_task` indexes these artifacts into a hybrid segment-caption Qdrant collection.
