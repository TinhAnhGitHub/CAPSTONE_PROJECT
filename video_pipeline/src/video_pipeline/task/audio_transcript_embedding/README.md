# Audio Transcript Embedding Task

## Purpose

`audio_transcript_embedding_chunk_task` embeds raw spoken transcript text from `AudioSegmentArtifact.audio_text`. This supports semantic search over what was said, independent of generated segment captions.

## How It Works

The task implementation is `AudioTranscriptEmbeddingTask` in `main.py`.

1. `preprocess()` extracts and strips `audio_text` from each audio segment.
2. Empty transcript segments are skipped.
3. `execute()` sends all non-empty texts to `MMBertClient.ainfer()` as one batch.
4. `execute()` creates `AudioTranscriptEmbedArtifact` objects and serializes vectors as `.npy` bytes.
5. `postprocess()` uploads vector files to MinIO and persists artifact metadata.
6. `summary_artifact()` records coverage, dimensions, and transcript previews.

## Input

Type: `list[AudioSegmentArtifact]`

| Field | Meaning |
|-------|---------|
| `audio_text` | Transcript to embed. |
| `segment_index` | Segment order. |
| `start_frame` and `end_frame` | Frame range. |
| `start_sec` and `end_sec` | Time range. |

## Output

Type: `list[AudioTranscriptEmbedArtifact]`

| Field | Meaning |
|-------|---------|
| `related_audio_segment_artifact_id` | Source audio segment ID. |
| `audio_text` | Embedded transcript text. |
| `embedding_dim` | Dense vector dimension. |
| `object_name` | MinIO path under `embedding/audio_transcript/{video_id}/...npy`. |
| `metadata.audio_text_preview` | Short text preview for inspection. |

## Dask Parallelization

The flow maps this task with `audio_transcript_embedding_chunk_task.map(segment_batches)`. Dask parallelizes across audio segment batches. Each mapped task performs one batched mmBERT request for all non-empty transcripts in that batch.

## Algorithm Details

The algorithm preserves ASR semantics directly:

1. Use the merged audio segment transcript, not the visual caption.
2. Drop empty strings to avoid meaningless vectors.
3. Embed all remaining texts in one model call to reduce request overhead.
4. Store dense vectors as `.npy` files so Qdrant indexing can load them later.

## Downstream Consumers

`audio_transcript_qdrant_indexing_chunk_task` indexes the dense transcript vectors and adds sparse SPLADE vectors for hybrid search.
