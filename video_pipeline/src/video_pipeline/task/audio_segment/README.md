# Audio Segment Task

## Purpose

`audio_segment_task` converts raw ASR shot transcripts into `AudioSegmentArtifact` objects. It keeps short inputs unchanged and uses an LLM to conservatively merge neighboring ASR segments when there are many raw segments.

## How It Works

The task implementation is `AudioSegmentTask` in `main.py`.

1. `preprocess()` receives all `ASRArtifact` objects for one video and returns them unchanged.
2. `execute()` creates a `CostTracker` for the configured OpenRouter model.
3. If there are no ASR artifacts, it returns an empty result.
4. If the number of raw segments is `<= MAX_RAW_SEGMENTS_WITHOUT_MERGE`, it creates one audio segment per ASR artifact.
5. If all transcripts are empty, it also preserves raw segments.
6. Otherwise, it formats all ASR segments and asks the LLM for conservative neighboring merge rules.
7. `build_segments_from_merge_rules()` applies valid merge ranges while preserving untouched segments.
8. If the LLM merge is too aggressive and produces fewer than `MIN_SEGMENTS_AFTER_MERGE`, the task falls back to raw passthrough segments.
9. `postprocess()` persists every audio segment.

## Input

Type: `list[ASRArtifact]`

| Field | Meaning |
|-------|---------|
| `related_video_id` | Source video ID. |
| `related_video_fps` | Used to preserve frame/time alignment. |
| `metadata.text` | Raw transcript text. |
| `metadata.timestamp` | ASR segment start/end timestamps. |
| `metadata.frame_num` | ASR segment frame interval. |

## Output

Type: `tuple[list[AudioSegmentArtifact], CostTracker]`

| Output | Meaning |
|--------|---------|
| `AudioSegmentArtifact` list | Merged or passthrough transcript segments with timing and text. |
| `CostTracker` | LLM token usage, call count, and cost. |

Typical `AudioSegmentArtifact` fields include `segment_index`, `start_frame`, `end_frame`, `start_timestamp`, `end_timestamp`, `start_sec`, `end_sec`, `audio_text`, `related_video_id`, and `user_id`.

## Dask Parallelization

This task is submitted once with `audio_segment_task.submit(asr_results)` because it needs the full ordered transcript list to decide whether neighboring segments should be merged. It runs on the flow's Dask worker pool, but it is intentionally not mapped across batches.

## Algorithm Details

The algorithm is conservative segmentation rather than aggressive summarization:

1. Small transcript lists stay unchanged to avoid unnecessary LLM cost and accidental over-merging.
2. Empty-transcript inputs stay unchanged because the LLM has no semantic evidence.
3. The LLM returns structured `MergeList` rules, not final artifacts.
4. Each merge rule is clamped to valid segment indices.
5. Non-overlapping merge rules are applied in sorted order.
6. Gaps between merge rules are preserved as single raw segments.
7. A minimum segment count guard prevents a whole video from collapsing into a few broad chunks.

## Downstream Consumers

Audio segments feed `segment_caption_task`, `audio_transcript_embedding_task`, and all segment-level retrieval/indexing branches.
