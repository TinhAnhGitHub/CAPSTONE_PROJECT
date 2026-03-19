"""Shared configuration and constants for Qdrant indexing tasks."""

from __future__ import annotations

from typing import Any

from qdrant_client.models import Distance

from video_pipeline.task.base.base_task import TaskConfig
from video_pipeline.core.client.storage.qdrant.config import QdrantIndexConfig
from video_pipeline.config import get_settings

QDRANT_INDEXING_CONFIG = TaskConfig.from_yaml("qdrant_indexing_shared")
BASE_KWARGS = QDRANT_INDEXING_CONFIG.to_task_kwargs()
ADDITIONAL_KWARGS: dict[str, Any] = QDRANT_INDEXING_CONFIG.additional_kwargs

IMAGE_QDRANT_INDEXING_CONFIG = TaskConfig.from_yaml("image_qdrant_indexing")
SEGMENT_QDRANT_INDEXING_CONFIG = TaskConfig.from_yaml("segment_qdrant_indexing")
AUDIO_TRANSCRIPT_QDRANT_INDEXING_CONFIG = TaskConfig.from_yaml("audio_transcript_qdrant_indexing")
IMAGE_CAPTION_QDRANT_INDEXING_CONFIG = TaskConfig.from_yaml("image_caption_qdrant_indexing")
SEGMENT_CAPTION_QDRANT_INDEXING_CONFIG = TaskConfig.from_yaml("segment_caption_qdrant_indexing")

IMAGE_DENSE_FIELD = ADDITIONAL_KWARGS["image_dense_field"]
SEGMENT_DENSE_FIELD = ADDITIONAL_KWARGS["segment_dense_field"]

AUDIO_TRANSCRIPT_DENSE_FIELD = ADDITIONAL_KWARGS["audio_transcript_dense_field"]
AUDIO_TRANSCRIPT_SPARSE_FIELD = ADDITIONAL_KWARGS["audio_transcript_sparse_field"]


IMAGE_CAPTION_DENSE_FIELD = ADDITIONAL_KWARGS.get("image_caption_dense_field", "image_caption_dense")
IMAGE_CAPTION_SPARSE_FIELD = ADDITIONAL_KWARGS.get("image_caption_sparse_field", "image_caption_sparse")

SEGMENT_CAPTION_DENSE_FIELD = ADDITIONAL_KWARGS.get("segment_caption_dense_field", "segment_caption_dense")
SEGMENT_CAPTION_SPARSE_FIELD = ADDITIONAL_KWARGS.get("segment_caption_sparse_field", "segment_caption_sparse")


def get_collection_base() -> str:
    return get_settings().qdrant.collection_base


def build_image_collection_name() -> str:
    return f"{get_collection_base()}_image"


def build_segment_collection_name() -> str:
    return f"{get_collection_base()}_segment"


def build_audio_transcript_collection_name() -> str:
    return f"{get_collection_base()}_audio_transcript"


def build_image_caption_collection_name() -> str:
    return f"{get_collection_base()}_image_caption"


def build_segment_caption_collection_name() -> str:
    return f"{get_collection_base()}_segment_caption"


def get_image_index_configs():
    on_disk = ADDITIONAL_KWARGS.get("on_disk", False)
    image_dim = ADDITIONAL_KWARGS.get("image_dim")

    cfg = QdrantIndexConfig(
        vector_size=image_dim, #type:ignore
        distance=Distance.COSINE,
        on_disk=on_disk,
    )
    return [cfg], [IMAGE_DENSE_FIELD]


def get_segment_index_configs():
    on_disk = ADDITIONAL_KWARGS.get("on_disk", False)
    segment_dim = ADDITIONAL_KWARGS.get("image_dim")

    cfg = QdrantIndexConfig(
        vector_size=segment_dim, #type:ignore
        distance=Distance.COSINE,
        on_disk=on_disk,
    )
    return [cfg], [SEGMENT_DENSE_FIELD]


def get_audio_transcript_index_configs():
    on_disk = ADDITIONAL_KWARGS.get("on_disk", False)
    audio_transcript_dim = ADDITIONAL_KWARGS.get("caption_dim", 768)

    cfg = QdrantIndexConfig(
        vector_size=audio_transcript_dim,
        distance=Distance.COSINE,
        on_disk=on_disk,
    )
    sparse_cfg = QdrantIndexConfig(
        vector_size=0, 
        distance=Distance.COSINE,
        on_disk=on_disk,
        is_sparse=True,
    )
    return [cfg, sparse_cfg], [AUDIO_TRANSCRIPT_DENSE_FIELD, AUDIO_TRANSCRIPT_SPARSE_FIELD]


def get_image_caption_index_configs():
    on_disk = ADDITIONAL_KWARGS.get("on_disk", False)
    caption_dim = ADDITIONAL_KWARGS.get("caption_dim", 768)

    dense_cfg = QdrantIndexConfig(
        vector_size=caption_dim,
        distance=Distance.COSINE,
        on_disk=on_disk,
        is_sparse=False,
    )
    sparse_cfg = QdrantIndexConfig(
        vector_size=0, 
        distance=Distance.COSINE,
        on_disk=on_disk,
        is_sparse=True,
    )
    return [dense_cfg, sparse_cfg], [IMAGE_CAPTION_DENSE_FIELD, IMAGE_CAPTION_SPARSE_FIELD]


def get_segment_caption_index_configs():
    on_disk = ADDITIONAL_KWARGS.get("on_disk", False)
    caption_dim = ADDITIONAL_KWARGS.get("caption_dim", 768)

    dense_cfg = QdrantIndexConfig(
        vector_size=caption_dim,
        distance=Distance.COSINE,
        on_disk=on_disk,
        is_sparse=False,
    )
    sparse_cfg = QdrantIndexConfig(
        vector_size=0, 
        distance=Distance.COSINE,
        on_disk=on_disk,
        is_sparse=True,
    )
    return [dense_cfg, sparse_cfg], [SEGMENT_CAPTION_DENSE_FIELD, SEGMENT_CAPTION_SPARSE_FIELD]