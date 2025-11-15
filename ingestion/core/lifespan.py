from __future__ import annotations
import subprocess
from contextlib import asynccontextmanager
import os
from typing import AsyncIterator, cast
from fastapi import FastAPI
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
        print("jojo")
    except FileNotFoundError as e:
        raise RuntimeError("uv command is not available to create Prefect deployment") from e

    stdout, stderr = await process.communicate()
    if process.returncode != 0:
        err_output = stderr.decode().strip() or stdout.decode().strip()
        raise RuntimeError(
            f"Prefec deployment failed: {err_output}"
        )
    
    worker_pool_cmds = [
        ["uv", "run", "prefect", "work-pool", "create", "--type", "process", "local-pool"],
        ["uv", "run", "prefect", "concurrency-limit", "create", "llm-service", "3"],
        ["uv", "run", "prefect", "concurrency-limit", "create", "embedding-service", "3"],
        ["uv", "run", "prefect", "concurrency-limit", "create", "autoshot-task", "1"],
        ["uv", "run", "prefect", "concurrency-limit", "create", "asr-task", "1"],
        ["uv", "run", "prefect", "concurrency-limit", "create", "video-registry", "1"],
    ]
    for cmd in worker_pool_cmds:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            run_logger.warning(f"⚠️ Work pool command failed:\n{stderr.decode().strip() or stdout.decode().strip()}")
        else:
            run_logger.info(f"✅ Ran: {' '.join(cmd)}")
        
    
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
    


    app.state.storage_client = storage_client
    app.state.artifact_tracker = tracker
    app.state.artifact_visitor = visitor
    app.state.artifact_deleter = deleter
    app.state.base_client_config=base_client_config
    state.base_client_config = base_client_config
    yield
    await tracker.close()

    
