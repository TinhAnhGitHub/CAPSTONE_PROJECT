from video_pipeline.task.qdrant_indexing.config import (
    QDRANT_INDEXING_CONFIG,
    BASE_KWARGS,
    ADDITIONAL_KWARGS,
    IMAGE_DENSE_FIELD,
    SEGMENT_DENSE_FIELD,
    AUDIO_TRANSCRIPT_DENSE_FIELD,
    IMAGE_CAPTION_DENSE_FIELD,
    IMAGE_CAPTION_SPARSE_FIELD,
    SEGMENT_CAPTION_DENSE_FIELD,
    SEGMENT_CAPTION_SPARSE_FIELD,
    build_image_collection_name,
    build_segment_collection_name,
    build_audio_transcript_collection_name,
    build_image_caption_collection_name,
    build_segment_caption_collection_name,
)

from video_pipeline.task.qdrant_indexing.image import (
    ImageQdrantIndexingTask,
    image_qdrant_indexing_chunk_task,
)

from video_pipeline.task.qdrant_indexing.segment import (
    SegmentQdrantIndexingTask,
    segment_qdrant_indexing_chunk_task,
)

from video_pipeline.task.qdrant_indexing.audio_transcript import (
    AudioTranscriptQdrantIndexingTask,
    audio_transcript_qdrant_indexing_chunk_task,
)

from video_pipeline.task.qdrant_indexing.image_caption import (
    ImageCaptionQdrantIndexingTask,
    image_caption_qdrant_indexing_chunk_task,
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
    "SEGMENT_DENSE_FIELD",
    "AUDIO_TRANSCRIPT_DENSE_FIELD",
    "IMAGE_CAPTION_DENSE_FIELD",
    "IMAGE_CAPTION_SPARSE_FIELD",
    "SEGMENT_CAPTION_DENSE_FIELD",
    "SEGMENT_CAPTION_SPARSE_FIELD",

    "build_image_collection_name",
    "build_segment_collection_name",
    "build_audio_transcript_collection_name",
    "build_image_caption_collection_name",
    "build_segment_caption_collection_name",

    "ImageQdrantIndexingTask",
    "image_qdrant_indexing_chunk_task",
    "SegmentQdrantIndexingTask",
    "segment_qdrant_indexing_chunk_task",
    "AudioTranscriptQdrantIndexingTask",
    "audio_transcript_qdrant_indexing_chunk_task",
    "ImageCaptionQdrantIndexingTask",
    "image_caption_qdrant_indexing_chunk_task",
    "SegmentCaptionQdrantIndexingTask",
    "segment_caption_qdrant_indexing_chunk_task",
]