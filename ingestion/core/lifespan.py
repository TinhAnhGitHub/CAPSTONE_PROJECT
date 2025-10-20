from __future__ import annotations
from contextlib import asynccontextmanager
import os
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from core.management.status import VideoStatusManager, VideoStatusInfo


from core.artifact.persist import ArtifactPersistentVisitor
from core.clients.base import ClientConfig, MilvusCollectionConfig
from core.config.logging import configure_logging, logger_config
from core.config.storage import minio_settings, postgre_settings, milvus_settings
from core.pipeline.tracker import ArtifactTracker
from core.storage import StorageClient
from core.management.cleanup import ArtifactDeleter
from core.app_state import AppState

# Task imports
from task.video_proc.main import VideoIngestionTask
from task.video_proc.config import VideoIngestionSettings
from task.asr_task.main import ASRProcessingTask, ASRSettings
from task.autoshot_task.main import AutoshotProcessingTask, AutoshotSettings
from task.image_processing.main import ImageProcessingTask, ImageProcessingSettings
from task.llm_segment_caption.main import SegmentCaptionLLMTask, LLMCaptionSettings
from task.llm_image_caption.main import ImageCaptionLLMTask, ImageCaptionSettings
from task.text_embedding.main import (
    TextImageCaptionEmbeddingTask,
    TextCaptionSegmentEmbeddingTask,
    TextEmbeddingSettings
)
from task.image_embedding.main import ImageEmbeddingTask, ImageEmbeddingSettings
from task.milvus_persist_task.main import (
    ImageEmbeddingMilvusTask,
    TextImageCaptionMilvusTask,
    TextSegmentCaptionMilvusTask,
    MilvusIndexSettings
)

from core.clients.milvus_client import ImageEmbeddingMilvusClient, TextCaptionEmbeddingMilvusClient, SegmentCaptionEmbeddingMilvusClient

from core.management.progress import ProgressTracker

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    state = AppState() 

    logger.info("🚀 Starting Video Processing Orchestration API...")
    configure_logging(logger_config)
    logger.info("✅ Logging configured")

    storage_client = StorageClient(settings=minio_settings)
    logger.info("✅ Storage client initialized")

    tracker = ArtifactTracker(database_url=postgre_settings.database_url)
    await tracker.initialize()
    logger.info("✅ Artifact tracker initialized")

    visitor = ArtifactPersistentVisitor(
        minio_client=storage_client,
        tracker=tracker
    )
    logger.info("✅ Artifact visitor initialized")
    


    consul_host = os.getenv("CONSUL_HOST", "localhost")
    consul_port = int(os.getenv("CONSUL_PORT", "8500"))
    logger.info(f"{consul_host=}:{consul_port=}")
    base_client_config = ClientConfig(
        timeout_seconds=300.0,
        max_retries=3,
        retry_min_wait=1.0,
        retry_max_wait=10.0,
        consul_host=consul_host,
        consul_port=consul_port,
    )

    app.state.base_client_config=base_client_config

    # ========================================================================
    # Task Configurations
    # ========================================================================
    
    # Video Ingestion
    video_ingestion_config = VideoIngestionSettings(
        retries=2,
        retry_delay_seconds=5,
        timeout_seconds=300
    )
    video_ingestion_task = VideoIngestionTask(
        artifact_visitor=visitor,
        config=video_ingestion_config
    )
    
    # Autoshot
    autoshot_config = AutoshotSettings(
        model_name="autoshot",
        device="cuda",
    )
    autoshot_task = AutoshotProcessingTask(
        artifact_visitor=visitor,
        config=autoshot_config
    )
    
    # ASR
    asr_config = ASRSettings(
        model_name="chunkformer",
        device="cuda",
    )
    asr_task = ASRProcessingTask(
        artifact_visitor=visitor,
        config=asr_config
    )
    
    # Image Processing
    image_config = ImageProcessingSettings(
        num_img_per_segment=3,
    )
    image_processing_task = ImageProcessingTask(
        artifact_visitor=visitor,
        config=image_config
    )
    
    # LLM Segment Caption
    segment_caption_config = LLMCaptionSettings(
        model_name="openrouter_api",
        device="cuda",
        image_per_segments=5
    )
    segment_caption_task = SegmentCaptionLLMTask(
        artifact_visitor=visitor,
        config=segment_caption_config
    )
    
    # LLM Image Caption
    image_caption_config = ImageCaptionSettings(
        model_name="openrouter_api",
        device="cuda"
    )
    image_caption_task = ImageCaptionLLMTask(
        artifact_visitor=visitor,
        config=image_caption_config
    )
    
    # Image Embedding
    image_embedding_config = ImageEmbeddingSettings(
        model_name="open_clip",
        device="cuda",
        batch_size=32
    )   
    image_embedding_task = ImageEmbeddingTask(
        artifact_visitor=visitor,
        config=image_embedding_config
    )
    
    # Text Embedding
    text_embedding_config = TextEmbeddingSettings(
        model_name="mmbert",
        device="cuda",
        batch_size=16
    )
    text_image_caption_task = TextImageCaptionEmbeddingTask(
        artifact_visitor=visitor,
        config=text_embedding_config
    )
    text_segment_caption_task = TextCaptionSegmentEmbeddingTask(
        artifact_visitor=visitor,
        config=text_embedding_config
    )

    milvus_config = MilvusIndexSettings(
        host=milvus_settings.host,
        port=milvus_settings.port,
        user=milvus_settings.user,
        password=milvus_settings.password,
        db_name=milvus_settings.db_name, #!IMPORTANT
        time_out=30.0,
        ingest_batch_size=100
    )

    image_embed_milvus_collection = MilvusCollectionConfig(
        collection_name="image_embeddings",
        dimension=512,
        metric_type="COSINE",
        index_type="HNSW",
        description="Image embeddings for video frames"
    )
    image_embed_milvus_task = ImageEmbeddingMilvusTask(
        artifact_visitor=visitor,
        config_client=milvus_config,
    )

    text_caption_milvus_collection = MilvusCollectionConfig(
        collection_name="text_caption_embeddings",
        dimension=768,
        metric_type="COSINE",
        index_type="HNSW",
        description="Text embeddings for image captions"
    )
    text_caption_milvus_task = TextImageCaptionMilvusTask(
        artifact_visitor=visitor,
        config_client=milvus_config,
    )

    segment_caption_milvus_collection = MilvusCollectionConfig(
        collection_name="segment_caption_embeddings",
        dimension=768,
        metric_type="COSINE",
        index_type="HNSW",
        description="Text embeddings for segment captions"
    )
    segment_caption_milvus_task = TextSegmentCaptionMilvusTask(
        artifact_visitor=visitor,
        config_client=milvus_config,
    )
    image_client = ImageEmbeddingMilvusClient(
        config_collection=image_embed_milvus_collection,
        host=milvus_config.host,
        port=milvus_config.port,
        user=milvus_config.user,#type:ignore
        password=milvus_config.password,#type:ignore
        db_name=milvus_config.db_name,
        timeout=milvus_config.time_out
    )
    text_client = TextCaptionEmbeddingMilvusClient(
        config_collection=text_caption_milvus_collection,
        host=milvus_config.host,
        port=milvus_config.port,
        user=milvus_config.user,#type:ignore
        password=milvus_config.password,#type:ignore
        db_name=milvus_config.db_name,
        timeout=milvus_config.time_out
    )
    seg_client = SegmentCaptionEmbeddingMilvusClient(
        config_collection=segment_caption_milvus_collection,
        host=milvus_config.host,
        port=milvus_config.port,
        user=milvus_config.user,#type:ignore
        password=milvus_config.password,#type:ignore
        db_name=milvus_config.db_name,
        timeout=milvus_config.time_out
    )

    deleter = ArtifactDeleter(tracker=tracker, storage=storage_client, image_client=image_client, text_cap_client=text_client, text_seg_client=seg_client)
    logger.info("✅ Artifact deleter initialized")
    

    video_status = VideoStatusManager(
        storage=storage_client,
        tracker=tracker,
        image_client=image_client, text_cap_client=text_client, text_seg_client=seg_client
    )

    app.state.storage_client = storage_client
    app.state.artifact_tracker = tracker
    app.state.artifact_visitor = visitor
    app.state.artifact_deleter = deleter
    app.state.video_status = video_status

    state.base_client_config = base_client_config
    state.video_ingestion_task = video_ingestion_task
    state.autoshot_task = autoshot_task
    state.asr_task = asr_task
    state.image_processing_task = image_processing_task
    state.segment_caption_llm_task = segment_caption_task
    state.image_caption_llm_task = image_caption_task
    state.image_embedding_task = image_embedding_task
    state.text_image_caption_embedding_task = text_image_caption_task
    state.text_caption_segment_embedding_task = text_segment_caption_task

    state.image_embedding_milvus_task = image_embed_milvus_task
    state.text_image_caption_milvus_task = text_caption_milvus_task
    state.text_segment_caption_milvus_task = segment_caption_milvus_task

    # Assign milvus configs
    state.image_embedding_milvus_config = image_embed_milvus_collection
    state.text_image_caption_milvus_config = text_caption_milvus_collection
    state.text_segment_caption_milvus_config = segment_caption_milvus_collection
    
    progress_tracker = ProgressTracker()
    state.progress_tracker = progress_tracker


    logger.info("✅ All components initialized and stored in app state")
    logger.info("🎉 Application startup complete!")


    yield
    logger.info("🛑 Shutting down application...")
    await tracker.close()
    logger.info("✅ Tracker closed")
    logger.info("👋 Application shutdown complete")

    
