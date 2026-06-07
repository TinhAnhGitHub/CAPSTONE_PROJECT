# OCR Indexing Task

## Purpose

`ocr_indexing_chunk_task` indexes OCR text extracted from image frames into Elasticsearch. It enables keyword and text search over visible text in the video.

## How It Works

The task implementation is `OCRIndexingTask` in `main.py`.

1. `preprocess()` reads `ocr_text` from each `ImageOCRArtifact` metadata object.
2. Empty OCR text is skipped.
3. `execute()` creates the Elasticsearch index if needed.
4. `execute()` calls `batch_index_ocr_documents(..., generate_embeddings=True)`.
5. `postprocess()` returns indexed artifact IDs.
6. `summary_artifact()` records indexed document count.

## Input

Type: `list[ImageOCRArtifact]`

| Field | Meaning |
|-------|---------|
| `metadata.ocr_text` | Visible text extracted from the image. |
| `frame_index` | Source frame. |
| `timestamp_sec` | Time filter field. |
| `related_video_id` | Source video. |
| `image_minio_url` | Source frame URL. |

## Output

Type: `list[str]`

Each string is the artifact ID of an OCR document indexed into Elasticsearch.

## Dask Parallelization

The flow batches OCR artifacts and calls `ocr_indexing_chunk_task.map(ocr_index_batches, wait_for=ocr_batch_futures)`. Dask parallelizes indexing across batches. Each task batch performs one Elasticsearch bulk-style indexing operation.

## Algorithm Details

The task converts artifact metadata into Elasticsearch documents with video, frame, timestamp, and source image fields. It passes an mmBERT client to `ElasticsearchOCRClient`, so the indexer can generate text embeddings in addition to raw text fields when `generate_embeddings=True`.

## Downstream Consumers

This is a terminal indexing task. Retrieval services query Elasticsearch directly for OCR text search.
