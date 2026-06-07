# Qdrant Indexing Tasks

## Purpose

The `qdrant_indexing` package writes dense and hybrid retrieval vectors into Qdrant. It has separate collections for image embeddings, segment embeddings, audio transcript text, image caption text, and segment caption text.

## Tasks In This Package

| Task | Input | Collection | Vector fields |
|------|-------|------------|---------------|
| `image_qdrant_indexing_chunk_task` | `list[ImageEmbeddingArtifact]` | `{collection_base}_image` | Dense image vector. |
| `segment_qdrant_indexing_chunk_task` | `list[SegmentEmbeddingArtifact]` | `{collection_base}_segment` | Dense segment vector. |
| `audio_transcript_qdrant_indexing_chunk_task` | `list[AudioTranscriptEmbedArtifact]` | `{collection_base}_audio_transcript` | Dense mmBERT + sparse SPLADE. |
| `image_caption_qdrant_indexing_chunk_task` | `list[TextCaptionEmbeddingArtifact]` | `{collection_base}_image_caption` | Dense mmBERT + sparse SPLADE. |
| `segment_caption_qdrant_indexing_chunk_task` | `list[TextCapSegmentEmbedArtifact]` | `{collection_base}_segment_caption` | Dense mmBERT + sparse SPLADE. |

## Shared Flow

All indexing tasks follow the same lifecycle:

1. `preprocess()` downloads `.npy` dense vectors from MinIO with `load_npy_from_minio()`.
2. Text indexing tasks also call `encode_sparse_vectors()` to produce SPLADE sparse vectors.
3. `execute()` creates the Qdrant collection if it does not exist.
4. `execute()` converts artifacts into Qdrant point dictionaries with vector fields and payload metadata.
5. `execute()` calls `insert_vectors()` and returns inserted point IDs.
6. `postprocess()` returns those IDs unchanged.
7. `summary_artifact()` writes collection name and point counts.

## Input And Output

Output for every chunk task is `list[str]`, where each string is an inserted Qdrant point ID.

### Image Dense Index

Input: `ImageEmbeddingArtifact`

| Payload field | Meaning |
|---------------|---------|
| `frame_index` | Source frame number. |
| `timestamp_sec` | Numeric timestamp for filters. |
| `related_video_id` | Source video. |
| `image_minio_url` | Source image. |
| `caption_text` | Caption paired with image embedding. |

### Segment Dense Index

Input: `SegmentEmbeddingArtifact`

| Payload field | Meaning |
|---------------|---------|
| `start_sec`, `end_sec` | Segment time range. |
| `frame_indices` | Frames used for embedding. |
| `caption_text` | Segment summary caption. |
| `related_audio_segment_artifact_id` | Source audio segment. |

### Audio Transcript Hybrid Index

Input: `AudioTranscriptEmbedArtifact`

| Payload field | Meaning |
|---------------|---------|
| `audio_text` | Raw transcript text. |
| `segment_index` | Audio segment order. |
| `start_sec`, `end_sec` | Segment time range. |
| `embedding_dim` | Dense vector dimension. |

### Image Caption Hybrid Index

Input: `TextCaptionEmbeddingArtifact`

| Payload field | Meaning |
|---------------|---------|
| `caption_text` | Generated image caption. |
| `frame_index` | Source frame. |
| `image_id` | Source image artifact ID. |
| `image_minio_url` | Source image URL. |

### Segment Caption Hybrid Index

Input: `TextCapSegmentEmbedArtifact`

| Payload field | Meaning |
|---------------|---------|
| `caption_text` | Segment summary caption. |
| `segment_cap_id` | Source segment caption artifact ID. |
| `start_sec`, `end_sec` | Segment time range. |
| `start_frame`, `end_frame` | Segment frame range. |

## Dask Parallelization

Every Qdrant indexing task is mapped over batches in the flow with Prefect's `.map(...)`, and the flow uses `DaskTaskRunner`. That means Dask parallelizes indexing across batches. Each mapped task owns its own MinIO, PostgreSQL, and Qdrant clients and closes them after the batch finishes.

Examples in the flow:

```python
image_qdrant_indexing_chunk_task.map(image_index_batches, wait_for=embedding_batch_futures)
segment_qdrant_indexing_chunk_task.map(segment_index_batches, wait_for=segment_embedding_futures)
audio_transcript_qdrant_indexing_chunk_task.map(audio_transcript_index_batches, wait_for=audio_transcript_embedding_futures)
```

## Algorithm Details

Dense-only collections use cosine distance with a configured vector size. Hybrid collections create two vector configs: one dense vector field and one sparse vector field. Dense vectors are produced earlier by mmBERT or QwenVL and stored as `.npy`; sparse vectors are generated during indexing with SPLADE through Triton.

Collection names are built from `get_settings().qdrant.collection_base` plus a suffix. `make_qdrant_client()` enables gRPC/prefer-gRPC for faster bulk insertion.

## Retrieval Meaning

Dense image and segment collections support multimodal semantic search. Hybrid text collections support semantic search through dense embeddings and lexical matching through sparse SPLADE vectors. Payload fields preserve video ID, user ID, timestamps, frames, and source MinIO URLs for filtering and result reconstruction.
