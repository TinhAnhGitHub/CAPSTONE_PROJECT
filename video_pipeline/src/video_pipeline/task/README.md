# Video Pipeline Task Directory

## Purpose

This directory contains the processing stages used by `single_video_processing_flow`. Each stage wraps a concrete `BaseTask` implementation with a Prefect task function, so it can run under Prefect orchestration and Dask scheduling.

## Common Task Lifecycle

Most task classes follow this pattern:

1. `preprocess()` prepares artifacts, downloads required inputs, or extracts text.
2. `execute()` performs the core model call, indexing call, graph algorithm, or metadata extraction.
3. `postprocess()` stores binary outputs in MinIO and artifact metadata in PostgreSQL.
4. `summary_artifact()` creates a human-readable Prefect artifact.

## Dask Model

The flow uses `DaskTaskRunner(cluster_kwargs=get_settings().dask.to_cluster_kwargs())`. Parallelism comes from Prefect task submission:

| Pattern | Meaning |
|---------|---------|
| `.submit()` | Runs one task on the Dask task runner. Used for video-level stages such as registration, autoshot, KG finalization, and ArangoDB indexing. |
| `.map(batches)` | Runs one Dask task per batch. Used for ASR, image extraction, caption/OCR, embeddings, Qdrant indexing, OCR indexing, and KG extraction. |
| `asyncio.gather()` inside a task | Adds intra-task concurrency for remote model calls, often bounded by a semaphore. |

Some stages cannot be mapped even though Dask is available because they need full-video context. Examples are audio segmentation, entity resolution, KG finalization, and ArangoDB indexing.

## Stage Summary

| Directory | Main output | Parallelization |
|-----------|-------------|-----------------|
| `video` | `VideoArtifact` | Single `.submit()`. |
| `autoshot` | `AutoshotArtifact` | Single `.submit()` plus internal async model batching. |
| `asr` | `ASRArtifact` list | Dask `.map()` over ASR batches. |
| `audio_segment` | `AudioSegmentArtifact` list | Single `.submit()` because it needs all ASR segments. |
| `image_extraction` | `ImageArtifact` list | Dask `.map()` over frame batches. |
| `image_caption_ocr` | `ImageCaptionArtifact` and `ImageOCRArtifact` lists | Dask `.map()` plus internal async VLM calls. |
| `image_embedding` | `ImageEmbeddingArtifact` list | Dask `.map()` plus internal async embedding calls. |
| `image_caption_embedding` | `TextCaptionEmbeddingArtifact` list | Dask `.map()` over caption batches. |
| `segment_caption` | `SegmentCaptionArtifact` list | Dask `.map()` plus internal async VLM calls. |
| `segment_embedding` | `SegmentEmbeddingArtifact` list | Dask `.map()` over segment batches. |
| `segment_caption_embedding` | `TextCapSegmentEmbedArtifact` list | Dask `.map()` over segment-caption batches. |
| `audio_transcript_embedding` | `AudioTranscriptEmbedArtifact` list | Dask `.map()` over audio segment batches. |
| `qdrant_indexing` | Qdrant point IDs | Dask `.map()` over indexing batches. |
| `ocr_indexing` | Elasticsearch document IDs | Dask `.map()` over OCR batches. |
| `kg_graph` | `KGGraphArtifact` | Dask `.map()` for extraction, single tasks for global resolution/finalization. |
| `arango_indexing` | `ArangoIndexingArtifact` | Single `.submit()` for the full graph. |

## Shared Utilities

`video_utils.py` contains shared helpers used by video-processing tasks. Currently it exposes `frames_to_timestamp(frame, fps)`, which converts a frame number into an `HH:MM:SS.mmm` timestamp. Most task-specific helper logic lives inside each task directory's `helper.py` or `util.py`.

## Knowledge Graph Path

The graph path starts after segment captions are available:

1. `segment_caption` creates summaries and atomic event captions.
2. `kg_graph` extracts raw per-segment entities and relationships.
3. `kg_graph` resolves duplicate entities into canonical global entities with dense+sparse clustering and LLM verification.
4. `kg_graph` links segment events and micro-events using temporal order, semantic similarity, shared entities, and LLM confirmation.
5. `arango_indexing` stores the final graph in ArangoDB vertex and edge collections.
