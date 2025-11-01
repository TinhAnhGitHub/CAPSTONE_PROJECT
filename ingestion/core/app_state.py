from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.storage import StorageClient
    from core.pipeline.tracker import ArtifactTracker
    from core.artifact.persist import ArtifactPersistentVisitor
    from core.management.cleanup import ArtifactDeleter
    from core.clients.milvus_client import ImageMilvusClient, SegmentCaptionEmbeddingMilvusClient
    from core.clients.base import ClientConfig
    from core.management.progress import HTTPProgressTracker
    from task.video_proc.main import VideoIngestionTask
    from task.asr_task.main import ASRProcessingTask
    from task.autoshot_task.main import AutoshotProcessingTask
    from task.image_processing.main import ImageProcessingTask
    from task.llm_segment_caption.main import SegmentCaptionLLMTask
    from task.llm_image_caption.main import ImageCaptionLLMTask
    from task.text_embedding.main import (
        TextImageCaptionEmbeddingTask,
        TextCaptionSegmentEmbeddingTask,
    )
    from task.image_embedding.main import ImageEmbeddingTask
    from task.milvus_persist_task.main import (
        ImageEmbeddingMilvusTask,
        TextSegmentCaptionMilvusTask,
    )


class AppState:
    """Singleton app-level state container for all core dependencies."""
    _instance: AppState | None = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    # Core dependencies
    storage_client: "StorageClient" = None  # type: ignore
    artifact_tracker: "ArtifactTracker" = None  # type: ignore
    artifact_visitor: "ArtifactPersistentVisitor" = None  # type: ignore
    artifact_deleter: "ArtifactDeleter" = None  # type: ignore

    # Task instances
    video_ingestion_task: "VideoIngestionTask" = None  # type: ignore
    autoshot_task: "AutoshotProcessingTask" = None  # type: ignore
    asr_task: "ASRProcessingTask" = None  # type: ignore
    image_processing_task: "ImageProcessingTask" = None  # type: ignore
    segment_caption_llm_task: "SegmentCaptionLLMTask" = None  # type: ignore
    image_caption_llm_task: "ImageCaptionLLMTask" = None  # type: ignore

    image_embedding_task: "ImageEmbeddingTask" = None  # type: ignore
    text_image_caption_embedding_task: "TextImageCaptionEmbeddingTask" = None  # type: ignore
    text_caption_segment_embedding_task: "TextCaptionSegmentEmbeddingTask" = None  # type: ignore

    # Milvus persistence tasks
    image_embedding_milvus_task: "ImageEmbeddingMilvusTask" = None  # type: ignore
    text_segment_caption_milvus_task: "TextSegmentCaptionMilvusTask" = None  # type: ignore

    # Milvus clients
    image_milvus_client: "ImageMilvusClient" = None  # type: ignore
    seg_milvus_client: "SegmentCaptionEmbeddingMilvusClient" = None  # type: ignore

    # Configuration and progress tracking
    base_client_config: "ClientConfig" = None  # type: ignore
    progress_client: "HTTPProgressTracker" = None  # type: ignore