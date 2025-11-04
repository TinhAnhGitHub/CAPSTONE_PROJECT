from __future__ import annotations
import subprocess
from contextlib import asynccontextmanager
import os
from typing import AsyncIterator, cast
from fastapi import FastAPI
from core.management.status import VideoStatusManager
from core.clients.base import ClientConfig
from core.config.storage import minio_settings, postgre_settings, milvus_settings
from core.config.task_config import consule_conf
from core.config.milvus_index_config import image_caption_dense_conf, image_visual_dense_conf, image_caption_sparse_conf, segment_caption_dense_conf, segment_caption_sparse_conf
from core.app_state import AppState
from core.config.logging import run_logger
import asyncio
from prefect.client.orchestration import get_client
from prefect.exceptions import ObjectNotFound


DEPLOYMENT_NAME = os.getenv('PREFECT_DEPLOYMENT_NAME')
FLOW_NAME = os.getenv('PREFECT_FLOW_NAME')
DEPLOY_IDENTIFIER = f"{FLOW_NAME}/{DEPLOYMENT_NAME}"
PREFECT_FILE_PATH = os.getenv('PREFECT_FILE_PATH')

async def ensure_prefect_deployment_exists() -> None:
    try:
        async with get_client() as client:
            await client.read_deployment_by_name(DEPLOY_IDENTIFIER)
            run_logger.info(f"Deployment {DEPLOY_IDENTIFIER} exists")
            return
    except ObjectNotFound:
        run_logger.info(f"Prefect deployment missing: {DEPLOY_IDENTIFIER}")
    
    cmd = [
        "uv",
        "run",
        "prefect",
        "--no-prompt",
        "deploy",
        "--name",
        DEPLOYMENT_NAME,
        "--prefect-file",
        PREFECT_FILE_PATH
    ]

    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
    except FileNotFoundError as e:
        raise RuntimeError("uv command is not available to create Prefect deployment") from e

    stdout, stderr = await process.communicate()
    if process.returncode != 0:
        err_output = stderr.decode().strip() or stdout.decode().strip()
        raise RuntimeError(
            f"Prefec deployment failed: {err_output}"
        )
    
    run_logger.info(
        f"Prefect deployment ensurted for : {DEPLOY_IDENTIFIER}: {stdout.decode().strip()}"
    )
    





@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    await ensure_prefect_deployment_exists()

    state = AppState()
    from core.storage import StorageClient
    from core.pipeline.tracker import ArtifactTracker
    from core.artifact.persist import ArtifactPersistentVisitor
    from core.management.cleanup import ArtifactDeleter
    from core.clients.milvus_client import ImageMilvusClient, SegmentCaptionEmbeddingMilvusClient

    storage_client = StorageClient(settings=minio_settings)
    tracker = ArtifactTracker(database_url=postgre_settings.database_url)
    await tracker.initialize()
    visitor = ArtifactPersistentVisitor(
        minio_client=storage_client,
        tracker=tracker
    )    

    base_client_config = ClientConfig(
        timeout_seconds=consule_conf.timeout_seconds,
        max_retries=consule_conf.max_retries,
        retry_min_wait=consule_conf.retry_min_wait,
        retry_max_wait=consule_conf.retry_max_wait,
        consul_host=consule_conf.host,
        consul_port=consule_conf.port,
    )

    

    # ========================================================================
    # Task Configurations
    # ========================================================================
    # video_ingestion_task = VideoIngestionTask(artifact_visitor=visitor)
    # autoshot_task = AutoshotProcessingTask(artifact_visitor=visitor, model_name=tautoshot_conf.model_name, device=tautoshot_conf.device)

    # asr_task = ASRProcessingTask(artifact_visitor=visitor, model_name=tasr_conf.model_name, device=tasr_conf.device)

    # image_processing_task = ImageProcessingTask(artifact_visitor=visitor,num_img_per_segment=timage_processing_conf.num_img_per_segment)

    # segment_caption_task = SegmentCaptionLLMTask(artifact_visitor=visitor, image_per_segments=tllm_conf.image_per_segments, model_name=tllm_conf.model_name, device=tllm_conf.device)

    # image_caption_task = ImageCaptionLLMTask(artifact_visitor=visitor, model_name=tllm_conf.model_name, device=tllm_conf.device)

    # image_embedding_task = ImageEmbeddingTask(artifact_visitor=visitor, batch_size=t_i_embed_conf.batch_size, model_name=t_i_embed_conf.model_name, device=t_i_embed_conf.device)

    # text_image_caption_task = TextImageCaptionEmbeddingTask(artifact_visitor=visitor, batch_size=t_t_embed_conf.batch_size, model_name=t_i_embed_conf.model_name, device=t_i_embed_conf.device)
    # text_segment_caption_task = TextCaptionSegmentEmbeddingTask(artifact_visitor=visitor)
    # image_embed_milvus_task = ImageEmbeddingMilvusTask(artifact_visitor=visitor, ingest_batch_size=500)
    # segment_caption_milvus_task = TextSegmentCaptionMilvusTask(artifact_visitor=visitor, ingest_batch_size=500)


    image_milvus_client = ImageMilvusClient(
        host=milvus_settings.host,
        port=milvus_settings.port,
        collection_name='image_milvus',
        user=milvus_settings.user, #type:ignore
        password=milvus_settings.password, #type:ignore
        db_name=milvus_settings.db_name,
        timeout=milvus_settings.time_out,
        visual_index_config=image_visual_dense_conf,
        caption_dense_index_config=image_caption_dense_conf,
        caption_sparse_index_config=image_caption_sparse_conf
    )

    seg_milvus_client = SegmentCaptionEmbeddingMilvusClient(
        host=milvus_settings.host,
        port=milvus_settings.port,
        collection_name="segment_milvus",
        user=milvus_settings.user, #type:ignore
        password=milvus_settings.password, #type:ignore
        db_name=milvus_settings.db_name,
        timeout=milvus_settings.time_out,
        visual_index_config=None,
        caption_dense_index_config=segment_caption_dense_conf,
        caption_sparse_index_config=segment_caption_sparse_conf
    )

    deleter = ArtifactDeleter(tracker=tracker, storage=storage_client, image_client=image_milvus_client,text_seg_client=seg_milvus_client)
    

    video_status = VideoStatusManager(
        storage=storage_client,
        tracker=tracker,
        image_client=image_milvus_client,text_seg_client=seg_milvus_client
    )

    app.state.storage_client = storage_client
    app.state.artifact_tracker = tracker
    app.state.artifact_visitor = visitor
    app.state.artifact_deleter = deleter
    app.state.video_status = video_status
    app.state.base_client_config=base_client_config

    state.base_client_config = base_client_config
    # state.video_ingestion_task = video_ingestion_task
    # state.autoshot_task = autoshot_task
    # state.asr_task = asr_task
    # state.image_processing_task = image_processing_task
    # state.segment_caption_llm_task = segment_caption_task
    # state.image_caption_llm_task = image_caption_task
    # state.image_embedding_task = image_embedding_task
    # state.text_image_caption_embedding_task = text_image_caption_task
    # state.text_caption_segment_embedding_task = text_segment_caption_task

    # state.image_embedding_milvus_task = image_embed_milvus_task
    # state.text_segment_caption_milvus_task = segment_caption_milvus_task    


    # state.image_milvus_client = image_milvus_client
    # state.seg_milvus_client = seg_milvus_client

    
    # state.progress_client = HTTPProgressTracker(
    #     base_url=tracker_conf.base_url,
    #     endpoint=tracker_conf.endpoint,
    # )



    yield
    await tracker.close()

    
