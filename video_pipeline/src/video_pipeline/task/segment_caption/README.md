# Segment Caption Task

## Purpose

`segment_caption_chunk_task` generates structured multimodal captions for audio segments. It combines sampled video frames and the segment transcript, then produces a segment-level summary plus atomic event captions.

## How It Works

The task implementation is `SegmentCaptionTask` in `main.py`.

1. `preprocess()` receives a batch of `AudioSegmentArtifact` objects and returns it unchanged.
2. `execute()` streams the source video from MinIO once for the batch.
3. For each segment, it computes frame indices with `get_segment_frame_indices()`.
4. For each selected frame, it reads WebP bytes with `FastFrameReader` and encodes them as image data URLs.
5. It sends image blocks and the audio transcript to OpenRouter with the `OutputCaptionSegment` structured schema.
6. It runs segment caption calls concurrently, bounded by `max_concurrent` from `tasks.yaml`.
7. It creates `SegmentCaptionArtifact` objects with `summary_caption` and `event_captions`.
8. `postprocess()` persists each artifact.

## Input

Type: `list[AudioSegmentArtifact]`

| Field | Meaning |
|-------|---------|
| `start_sec` and `end_sec` | Time range for frame sampling. |
| `audio_text` | Transcript sent as text context. |
| `related_video_minio_url` | Source video for frame extraction. |
| `related_video_fps` | Used to convert time to frame indices. |

## Output

Type: `tuple[list[SegmentCaptionArtifact], CostTracker]`

| Output | Meaning |
|--------|---------|
| `SegmentCaptionArtifact` list | Captioned video segments with timing and text. |
| `CostTracker` | LLM usage and cost. |

Important artifact fields are `summary_caption`, `event_captions`, `audio_text`, `start_frame`, `end_frame`, `start_sec`, `end_sec`, and `related_audio_segment_artifact_id`.

## Dask Parallelization

The flow splits audio segments into `segment_batches` and calls `segment_caption_chunk_task.map(segment_batches)`. Dask parallelizes across batches. Within each batch, the task uses `asyncio.gather()` with a semaphore, so several segment-level VLM requests can run concurrently on one Dask worker.

## Algorithm Details

This task creates the textual foundation for the KG pipeline:

1. Sample a fixed number of visual frames from the segment interval.
2. Combine sampled visual evidence with ASR transcript text.
3. Ask the VLM for one coherent segment summary, capped by prompt instructions.
4. Ask for multiple short event captions, each representing one atomic observation.
5. Preserve both outputs. The summary is used for event-level KG nodes and embeddings. The event captions are later promoted to micro-events in `kg_graph`.

## Downstream Consumers

`segment_embedding_task`, `segment_caption_embedding_task`, and `kg_graph` consume `SegmentCaptionArtifact` objects.
