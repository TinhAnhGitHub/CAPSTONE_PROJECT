"""Qdrant indexing tasks for video pipeline.

This module provides tasks for indexing various embedding artifacts into Qdrant:

- Image embeddings (dense only)
- Caption embeddings (hybrid: dense + sparse)
- Segment embeddings (dense only)
- Segment caption embeddings (hybrid: dense + sparse)

Usage:
    from video_pipeline.task.qdrant_indexing import (
        image_qdrant_indexing_chunk_task,
        caption_qdrant_indexing_chunk_task,
        segment_qdrant_indexing_chunk_task,
        segment_caption_qdrant_indexing_chunk_task,
    )
"""

from video_pipeline.task.qdrant_indexing.config import (
    QDRANT_INDEXING_CONFIG,
    BASE_KWARGS,
    ADDITIONAL_KWARGS,
    IMAGE_DENSE_FIELD,
    CAPTION_TEXT_DENSE_FIELD,
    CAPTION_MM_DENSE_FIELD,
    CAPTION_SPARSE_FIELD,
    SEGMENT_DENSE_FIELD,
    build_image_collection_name,
    build_caption_collection_name,
    build_segment_collection_name,
)

from video_pipeline.task.qdrant_indexing.image import (
    ImageQdrantIndexingTask,
    image_qdrant_indexing_chunk_task,
)

from video_pipeline.task.qdrant_indexing.caption import (
    CaptionQdrantIndexingTask,
    caption_qdrant_indexing_chunk_task,
)

from video_pipeline.task.qdrant_indexing.segment import (
    SegmentQdrantIndexingTask,
    segment_qdrant_indexing_chunk_task,
)

from video_pipeline.task.qdrant_indexing.segment_caption import (
    SegmentCaptionQdrantIndexingTask,
    segment_caption_qdrant_indexing_chunk_task,
)


__all__ = [
    # Config
    "QDRANT_INDEXING_CONFIG",
    "BASE_KWARGS",
    "ADDITIONAL_KWARGS",
    "IMAGE_DENSE_FIELD",
    "CAPTION_TEXT_DENSE_FIELD",
    "CAPTION_MM_DENSE_FIELD",
    "CAPTION_SPARSE_FIELD",
    "SEGMENT_DENSE_FIELD",
    # Collection builders
    "build_image_collection_name",
    "build_caption_collection_name",
    "build_segment_collection_name",
    # Tasks
    "ImageQdrantIndexingTask",
    "image_qdrant_indexing_chunk_task",
    "CaptionQdrantIndexingTask",
    "caption_qdrant_indexing_chunk_task",
    "SegmentQdrantIndexingTask",
    "segment_qdrant_indexing_chunk_task",
    "SegmentCaptionQdrantIndexingTask",
    "segment_caption_qdrant_indexing_chunk_task",
]