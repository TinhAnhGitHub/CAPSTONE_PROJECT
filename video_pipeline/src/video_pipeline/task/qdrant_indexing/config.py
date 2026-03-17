"""Shared configuration and constants for Qdrant indexing tasks."""

from __future__ import annotations

from typing import Any

from qdrant_client.models import Distance

from video_pipeline.task.base.base_task import TaskConfig
from video_pipeline.core.client.storage.qdrant.config import QdrantIndexConfig
from video_pipeline.config import get_settings

QDRANT_INDEXING_CONFIG = TaskConfig.from_yaml("qdrant_indexing")
BASE_KWARGS = QDRANT_INDEXING_CONFIG.to_task_kwargs()
ADDITIONAL_KWARGS: dict[str, Any] = QDRANT_INDEXING_CONFIG.additional_kwargs

IMAGE_QDRANT_INDEXING_CONFIG = TaskConfig.from_yaml("image_qdrant_indexing")
CAPTION_QDRANT_INDEXING_CONFIG = TaskConfig.from_yaml("caption_qdrant_indexing")
SEGMENT_QDRANT_INDEXING_CONFIG = TaskConfig.from_yaml("segment_qdrant_indexing")
SEGMENT_CAPTION_QDRANT_INDEXING_CONFIG = TaskConfig.from_yaml("segment_caption_qdrant_indexing")
AUDIO_TRANSCRIPT_QDRANT_INDEXING_CONFIG = TaskConfig.from_yaml("audio_transcript_qdrant_indexing")

IMAGE_DENSE_FIELD = ADDITIONAL_KWARGS["image_dense_field"]
CAPTION_TEXT_DENSE_FIELD = ADDITIONAL_KWARGS["caption_text_dense_field"]
CAPTION_MM_DENSE_FIELD = ADDITIONAL_KWARGS["caption_mm_dense_field"]
CAPTION_SPARSE_FIELD = ADDITIONAL_KWARGS["caption_sparse_field"]
SEGMENT_DENSE_FIELD = "segment_dense"
AUDIO_TRANSCRIPT_DENSE_FIELD = "audio_transcript_dense"


def get_collection_base() -> str:
    """Get the collection base name from settings."""
    return get_settings().qdrant.collection_base


def build_image_collection_name() -> str:
    """Build the image collection name."""
    return f"{get_collection_base()}_image"


def build_caption_collection_name() -> str:
    """Build the caption collection name."""
    return f"{get_collection_base()}_caption"


def build_segment_collection_name() -> str:
    """Build the segment collection name."""
    return f"{get_collection_base()}_segment"


def build_audio_transcript_collection_name() -> str:
    """Build the audio transcript collection name."""
    return f"{get_collection_base()}_audio_transcript"


def get_image_index_configs():
    """Get index configuration for image embeddings (dense only)."""
    on_disk = ADDITIONAL_KWARGS.get("on_disk", False)
    image_dim = ADDITIONAL_KWARGS.get("image_dim", 1536)

    cfg = QdrantIndexConfig(
        vector_size=image_dim,
        distance=Distance.COSINE,
        on_disk=on_disk,
    )
    return [cfg], [IMAGE_DENSE_FIELD]


def get_caption_index_configs():
    """Get index configuration for caption embeddings (hybrid: dense + sparse)."""
    on_disk = ADDITIONAL_KWARGS.get("on_disk", False)
    caption_dim = ADDITIONAL_KWARGS.get("caption_dim", 768)
    mm_dim = ADDITIONAL_KWARGS.get("mm_dim", 1536)

    text_dense_cfg = QdrantIndexConfig(
        vector_size=caption_dim,
        distance=Distance.COSINE,
        on_disk=on_disk,
    )
    mm_dense_cfg = QdrantIndexConfig(
        vector_size=mm_dim,
        distance=Distance.COSINE,
        on_disk=on_disk,
    )
    sparse_cfg = QdrantIndexConfig(
        vector_size=0,
        distance=Distance.COSINE,
        on_disk=on_disk,
        is_sparse=True,
    )
    return (
        [text_dense_cfg, mm_dense_cfg, sparse_cfg],
        [CAPTION_TEXT_DENSE_FIELD, CAPTION_MM_DENSE_FIELD, CAPTION_SPARSE_FIELD],
    )


def get_segment_index_configs():
    """Get index configuration for segment embeddings (dense only)."""
    on_disk = ADDITIONAL_KWARGS.get("on_disk", False)
    segment_dim = ADDITIONAL_KWARGS.get("image_dim", 1536)

    cfg = QdrantIndexConfig(
        vector_size=segment_dim,
        distance=Distance.COSINE,
        on_disk=on_disk,
    )
    return [cfg], [SEGMENT_DENSE_FIELD]


def get_audio_transcript_index_configs():
    """Get index configuration for audio transcript embeddings (dense only).

    Uses mmBERT 768-dim embeddings for semantic search over spoken content.
    """
    on_disk = ADDITIONAL_KWARGS.get("on_disk", False)
    # mmBERT produces 768-dimensional embeddings
    audio_transcript_dim = ADDITIONAL_KWARGS.get("caption_dim", 768)

    cfg = QdrantIndexConfig(
        vector_size=audio_transcript_dim,
        distance=Distance.COSINE,
        on_disk=on_disk,
    )
    return [cfg], [AUDIO_TRANSCRIPT_DENSE_FIELD]