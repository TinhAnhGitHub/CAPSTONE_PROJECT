from __future__ import annotations
from typing import Any, cast
import asyncio
from datetime import datetime
from collections import defaultdict
from core.management.progress import ProcessingStage
from prefect import flow, task
from prefect.futures import PrefectFuture
from prefect.artifacts import acreate_markdown_artifact
from prefect.context import get_run_context
from prefect.logging import get_run_logger
from prefect.cache_policies import INPUTS, TASK_SOURCE, NO_CACHE

from core.artifact.schema import (
    VideoArtifact,
    AutoshotArtifact,
    ASRArtifact,
    ImageArtifact,
    SegmentCaptionArtifact,
    ImageCaptionArtifact,
    ImageEmbeddingArtifact,
    TextCapSegmentEmbedArtifact,
    TextCaptionEmbeddingArtifact, 
)

from task.video_proc.main import VideoInput
from task.llm_segment_caption.main import ShotASRInput

from core.config.logging import run_logger
from prefect.cache_policies import NO_CACHE

from core.clients.autoshot_client import AutoshotClient
from core.clients.asr_client import ASRClient
from core.clients.llm_client import LLMClient
from core.clients.image_embed_client import ImageEmbeddingClient
from core.clients.text_embed_client import TextEmbeddingClient
from core.app_state import AppState

#type stating
from task.milvus_persist_task.main import (
    ImageEmbeddingMilvusTask,
)
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



VIDEO_REGISTRY = 'video_registry'
AUTOSHOT_TASK = 'autoshot-task'
ASR_TASK = 'asr-task'
LLM_TASK_TAG = 'llm-service'
EMBED_TASK_TAG = 'embedding-service'
PERSIST_TASK_TAG = "milvus-persist"



@task(
    name='Video registry',
    description="This task will take uploaded videos, and persist into the tracker + minio S3",
    tags=[VIDEO_REGISTRY],
    cache_policy=NO_CACHE
)
async def entry_video_ingestion(
    video_uploads: VideoInput,
) -> list[VideoArtifact]:
    logger = get_run_logger()
    task_instance = AppState().video_ingestion_task
    progress_client = AppState().progress_client

    preprocessed_input = await task_instance.preprocess(input_data=video_uploads)
    video_artifacts = []
    total_uploads = len(video_uploads.files)
    logger.info("Starting video ingestion for %d uploads", total_uploads)

    for upload in video_uploads.files:
        logger.info("Starting ingestion for video %s", upload[0])
        await progress_client.start_video(
            video_id=upload[0]
        )
    async for result in task_instance.execute(preprocessed_input, None):
        processed = await task_instance.postprocess(result)
        video_artifacts.append(processed)        
        await progress_client.update_stage_progress(
            video_id=processed.artifact_id, 
            stage=ProcessingStage.VIDEO_INGEST,
            total_items=1,
            completed_items=1
        )
        
    return video_artifacts


@task(
    name="Autoshot Processing",
    description="Given a list of videos, detect the shot  segment boundary",
    tags=[AUTOSHOT_TASK],
    cache_policy=NO_CACHE
)
async def autoshot_task(
    videos: list[VideoArtifact],
):
    
    logger = get_run_logger()
    logger.info("Starting autoshot processing for %d videos", len(videos))
    task_instance = AppState().autoshot_task
    client_config =  AppState().base_client_config
    progress_client = AppState().progress_client


    async with AutoshotClient(
        config=client_config
    ) as client:

        await client.load_model(
            model_name=task_instance.kwargs.get('model_name'), #type:ignore
            device=task_instance.kwargs.get('device'), #type:ignore
        )
        preprocessed = await task_instance.preprocess(videos)
        logger.debug("Prepared %d segments for autoshot inference", len(preprocessed))

        video_id2total_items = defaultdict(int)
        for item in preprocessed:
            video_id2total_items[item.related_video_id] += 1
            
        video_id2current_items = defaultdict(int)
        results = []
        async for result in task_instance.execute(preprocessed, client):
            processed = await task_instance.postprocess(result)
            results.append(processed)
            video_id2current_items[processed.related_video_id] += 1
            video_id = processed.related_video_id

            await progress_client.update_stage_progress(
                video_id=video_id,
                stage=ProcessingStage.AUTOSHOT_SEGMENTATION,
                total_items=video_id2total_items[video_id],
                completed_items=video_id2current_items[video_id],
                details={}
            )
        
        await client.unload_model()
    logger.info("Completed autoshot processing; produced %d artifacts", len(results))
    return results
    


@task(
    name="ASR Processing",
    description="Given a list of videos, extracting the corresponding ASR",
    tags=[ASR_TASK],
    log_prints=True,
    cache_policy=NO_CACHE
    
)
async def asr_task(
    videos: list[VideoArtifact],
):
    logger = get_run_logger()
    logger.info("Starting ASR processing for %d videos", len(videos))
    task_instance = AppState().asr_task
    client_config = AppState().base_client_config
    progress_client = AppState().progress_client
    
    async with ASRClient(config=client_config) as client:
        await client.load_model(
            model_name=task_instance.kwargs.get('model_name'), #type:ignore
            device=task_instance.kwargs.get('device'), #type:ignore
        )
        preprocessed = await task_instance.preprocess(videos)
        logger.debug("Prepared %d audio chunks for ASR inference", len(preprocessed))
        video_id2total_items = defaultdict(int)
        for item in preprocessed:
            video_id2total_items[item.related_video_id] += 1
            
        video_id2current_items = defaultdict(int)
        results = []
        async for result in task_instance.execute(preprocessed, client):
            processed = await task_instance.postprocess(result)
            results.append(processed)
            video_id2current_items[processed.related_video_id] += 1
            video_id = processed.related_video_id

            await progress_client.update_stage_progress(
                video_id=video_id,
                stage=ProcessingStage.ASR_TRANSCRIPTION,
                total_items=video_id2total_items[video_id],
                completed_items=video_id2current_items[video_id],
                details={}
            )
        await client.unload_model()
    logger.info("Completed ASR processing; generated %d artifacts", len(results))
    return results



@task(
    name='Image Extraction',
    description="Extract frames from video segments",
    log_prints=True,
    cache_policy=NO_CACHE
)
async def image_processing_task(
    autoshots: list[AutoshotArtifact],
)-> list[ImageArtifact]:
    logger = get_run_logger()
    logger.info("Starting image extraction task for %d autoshot collections", len(autoshots))
    task_instance = AppState().image_processing_task
    preprocessed = await task_instance.preprocess(autoshots)
    progress_client = AppState().progress_client

    video_id2total_items = {
        v[0].related_video_id: len(v)
        for v in preprocessed.values() if v
    }
    total_frames = sum(video_id2total_items.values())
    logger.debug("Prepared %d frames across %d videos", total_frames, len(video_id2total_items))

    results = []
    video_id2current_items = defaultdict(int)
    async for result in task_instance.execute(preprocessed, None):
        processed = await task_instance.postprocess(result)
        results.append(processed)
        video_id2current_items[processed.related_video_id] += 1
        video_id = processed.related_video_id

        await progress_client.update_stage_progress(
            video_id=video_id,
            stage=ProcessingStage.IMAGE_EXTRACTION,
            total_items=video_id2total_items[video_id],
            completed_items=video_id2current_items[video_id],
            details={}
        )
    
    logger.info("Completed image extraction; produced %d image artifacts", len(results))
    return results


@task(
    name='Segmentation Caption Generation',
    description="Generate captions for video segments using LLM",
    tags=[LLM_TASK_TAG],
    cache_policy=NO_CACHE
)
async def segment_caption_task(
    autoshots: list[AutoshotArtifact],
    asrs: list[ASRArtifact],
) -> list[SegmentCaptionArtifact]:
    logger = get_run_logger()
    logger.info(
        "Starting segment caption generation for %d autoshot artifacts and %d transcripts",
        len(autoshots),
        len(asrs),
    )
     
    task_instance =  AppState().segment_caption_llm_task
    client_config =  AppState().base_client_config
    progress_client = AppState().progress_client

    input_data = ShotASRInput(
        list_asrs=asrs,
        lists_autoshots=autoshots
    )
    async with LLMClient(config=client_config)  as client:
        await client.load_model(
            model_name=task_instance.kwargs.get('model_name'), #type:ignore
            device=task_instance.kwargs.get('device'), #type:ignore
        )
        preprocessed = await task_instance.preprocess(input_data)
        logger.debug("Prepared %d prompt payloads for LLM captioning", len(preprocessed))
        video_id2total_items = defaultdict(int)
        for item in preprocessed:
            video_id2total_items[item.related_video_id] += 1

        video_id2current_items = defaultdict(int)
        results = []
        async for result in task_instance.execute(preprocessed, client):
            processed = await task_instance.postprocess(result)
            results.append(processed)
            video_id2current_items[processed.related_video_id] += 1
            video_id = processed.related_video_id

            await progress_client.update_stage_progress(
                video_id=video_id,
                stage=ProcessingStage.SEGMENT_CAPTIONING,
                total_items=video_id2total_items[video_id],
                completed_items=video_id2current_items[video_id],
                details={}
            )
        
        # await client.unload_model()
    logger.info("Completed segment caption generation with %d artifacts", len(results))
    return results


@task(
    name='Image Caption Generation',
    description='Generate captions for individual images using LLM',
    tags=[LLM_TASK_TAG],
    cache_policy=NO_CACHE
)
async def image_caption_task(
    images: list[ImageArtifact] | PrefectFuture,
) -> list[ImageCaptionArtifact]:
    logger = get_run_logger()
    task_instance =  AppState().image_caption_llm_task
    client_config =  AppState().base_client_config
    progress_client = AppState().progress_client
    async with LLMClient(config=client_config) as client:
        await client.load_model(
           model_name=task_instance.kwargs.get('model_name'), #type:ignore
            device=task_instance.kwargs.get('device'), #type:ignore
        )
        preprocessed = await task_instance.preprocess(cast(list[ImageArtifact],images))
        logger.info("Starting image captioning for %d images", len(preprocessed))
        video_id2total_items = defaultdict(int)
        for item in preprocessed:
            video_id2total_items[item.related_video_id] += 1
        video_id2current_items = defaultdict(int)
        results = []
        async for result in task_instance.execute(preprocessed, client):
            processed = await task_instance.postprocess(result)
            results.append(processed)
            video_id2current_items[processed.related_video_id] += 1
            video_id = processed.related_video_id

            await progress_client.update_stage_progress(
                video_id=video_id,
                stage=ProcessingStage.IMAGE_CAPTIONING,
                total_items=video_id2total_items[video_id],
                completed_items=video_id2current_items[video_id],
                details={}
            )
        # await client.unload_model()
    logger.info("Completed image captioning with %d artifacts", len(results))
    return results  


@task(
    name='Image Embedding generation',
    description='Generate embeddings for images',
    tags=[EMBED_TASK_TAG],
    cache_policy=NO_CACHE
)
async def image_embedding_task(
    images: list[ImageArtifact] | PrefectFuture,
) -> list[ImageEmbeddingArtifact]:
    logger = get_run_logger()
    task_instance =  AppState().image_embedding_task
    client_config    =  AppState().base_client_config
    progress_client = AppState().progress_client

    async with ImageEmbeddingClient(config=client_config) as client:
        await client.load_model(
            model_name=task_instance.kwargs.get('model_name'), #type:ignore
            device=task_instance.kwargs.get('device'), #type:ignore
        )
        preprocessed = await task_instance.preprocess(cast(list[ImageArtifact],images))
        logger.info("Starting image embedding generation for %d images", len(preprocessed))
        video_id2total_items = defaultdict(int)
        for item in preprocessed:
            video_id2total_items[item.related_video_id] += 1
        video_id2current_items = defaultdict(int)
        results = []
        async for result in task_instance.execute(preprocessed, client):
            processed = await task_instance.postprocess(result)
            results.append(processed)
            video_id2current_items[processed.related_video_id] += 1
            video_id = processed.related_video_id

            await progress_client.update_stage_progress(
                video_id=video_id,
                stage=ProcessingStage.IMAGE_EMBEDDING,
                total_items=video_id2total_items[video_id],
                completed_items=video_id2current_items[video_id],
                details={}
            )
        await client.unload_model()
    logger.info("Completed image embedding generation with %d artifacts", len(results))
    return results 



@task(
    name="segment_caption_embedding",
    description="Generate embeddings for image captions",
    tags=[EMBED_TASK_TAG],
    cache_policy=NO_CACHE
)
async def  segment_text_caption_embedding_task(
    segment_captions: list[SegmentCaptionArtifact] | PrefectFuture,
) ->list[TextCapSegmentEmbedArtifact]:
    logger = get_run_logger()
    task_instance =  AppState().text_caption_segment_embedding_task
    client_config =  AppState().base_client_config
    progress_client = AppState().progress_client

    async with TextEmbeddingClient(config=client_config) as client:
        await client.load_model(
           model_name=task_instance.kwargs.get('model_name'), #type:ignore
            device=task_instance.kwargs.get('device'), #type:ignore
        )
        preprocessed = await task_instance.preprocess(segment_captions) #type:ignore
        logger.info("Starting segment caption embedding for %d captions", len(preprocessed))
        video_id2total_items = defaultdict(int)
        for item in preprocessed:
            video_id2total_items[item.related_video_id] += 1

        video_id2current_items = defaultdict(int)
        results = []
        async for result in task_instance.execute(preprocessed, client):
            processed = await task_instance.postprocess(result)
            results.append(processed)
            video_id2current_items[processed.related_video_id] += 1
            video_id = processed.related_video_id

            await progress_client.update_stage_progress(
                video_id=video_id,
                stage=ProcessingStage.TEXT_CAP_SEGMENT_EMBEDDING,
                total_items=video_id2total_items[video_id],
                completed_items=video_id2current_items[video_id],
                details={}
            )
        await client.unload_model()
    logger.info("Completed segment caption embedding with %d artifacts", len(results))
    return results  



@task(
    name='text caption embedding',
    description="Generate embedding from text",
    tags=[EMBED_TASK_TAG],
    cache_policy=NO_CACHE
)
async def text_image_caption_embedding_task(
    captions: list[ImageCaptionArtifact] |PrefectFuture,
)-> list[TextCaptionEmbeddingArtifact]:
    logger = get_run_logger()
    task_instance =  AppState().text_image_caption_embedding_task
    client_config =  AppState().base_client_config
    progress_client = AppState().progress_client

    async with TextEmbeddingClient(config=client_config) as client:
        await client.load_model(
            model_name=task_instance.kwargs.get('model_name'), #type:ignore
            device=task_instance.kwargs.get('device'), #type:ignore
        )
        preprocessed = await task_instance.preprocess(cast(list[ImageCaptionArtifact], captions))
        logger.info("Starting text caption embedding for %d captions", len(preprocessed))
        video_id2total_items = defaultdict(int)
        for item in preprocessed:
            video_id2total_items[item.related_video_id] += 1

        video_id2current_items = defaultdict(int)
        results = []
        async for result in task_instance.execute(preprocessed, client):
            processed = await task_instance.postprocess(result)
            results.append(processed)
            video_id2current_items[processed.related_video_id] += 1
            video_id = processed.related_video_id

            await progress_client.update_stage_progress(
                video_id=video_id,
                stage=ProcessingStage.TEXT_CAP_IMAGE_EMBEDDING,
                total_items=video_id2total_items[video_id],
                completed_items=video_id2current_items[video_id],
                details={}
            )
        await client.unload_model()
    logger.info("Completed text caption embedding with %d artifacts", len(results))
    return results  #type:ignore



@task(
    name='Image embedding Milvus Persist',
    description='Persist image embedding into milvus',
    tags=[PERSIST_TASK_TAG],
    cache_policy=NO_CACHE
)
async def image_embedding_milvus_persist_task(
    image_caption_embeddings: tuple[list[ImageEmbeddingArtifact], list[TextCaptionEmbeddingArtifact]] | PrefectFuture,
):
    logger = get_run_logger()
    task_instance =   AppState().image_embedding_milvus_task

    image_milvus_client = AppState().image_milvus_client
    progress_client = AppState().progress_client
    await image_milvus_client.connect()

    preprocessed = await task_instance.preprocess(cast(tuple[list[ImageEmbeddingArtifact], list[TextCaptionEmbeddingArtifact]], image_caption_embeddings))
    logger.info("Starting Milvus persistence for %d image embedding batches", len(preprocessed))

    video_id2total_items = defaultdict(int)
    for item in preprocessed:
        video_id2total_items[item[0].related_video_id] += 1

    video_id2current_items = defaultdict(int)
    results = []
    async for result in task_instance.execute(preprocessed, image_milvus_client):
        processed = await task_instance.postprocess(result)
        results.append(processed)
        video_id2current_items[processed[0].related_video_id] += 1
        video_id = processed[0].related_video_id

        await progress_client.update_stage_progress(
            video_id=video_id,
            stage=ProcessingStage.IMAGE_MILVUS,
            total_items=video_id2total_items[video_id],
            completed_items=video_id2current_items[video_id],
            details={}
        )
    await image_milvus_client.close()
    logger.info("Completed Milvus persistence for image embeddings with %d batches", len(results))
    
    return results

@task(
    name='Text Segment Caption Milvus Persist',
    description='Persist segment caption embeddings into Milvus vector database',
    tags=[PERSIST_TASK_TAG],
    cache_policy=NO_CACHE
)
async def text_segment_caption_milvus_persist_task(
    text_segment_embeddings: list[TextCapSegmentEmbedArtifact] | PrefectFuture,
):
    logger = get_run_logger()
    task_instance =  AppState().text_segment_caption_milvus_task
    seg_milvus_client =  AppState().seg_milvus_client
    progress_client = AppState().progress_client
    await  seg_milvus_client.connect()
    
    preprocessed = await task_instance.preprocess(cast(list[TextCapSegmentEmbedArtifact], text_segment_embeddings))
    logger.info("Starting Milvus persistence for %d segment caption embeddings", len(preprocessed))
    results = []
    video_id2total_items = defaultdict(int)
    for item in preprocessed:
        video_id2total_items[item.related_video_id] += 1
    
    video_id2current_items = defaultdict(int)
    results = []
    async for result in task_instance.execute(preprocessed, seg_milvus_client):
        processed = await task_instance.postprocess(result)
        results.append(processed)
        video_id2current_items[processed.related_video_id] += 1
        video_id = processed.related_video_id

        await progress_client.update_stage_progress(
            video_id=video_id,
            stage=ProcessingStage.TEXT_CAP_SEGMENT_MILVUS,
            total_items=video_id2total_items[video_id],
            completed_items=video_id2current_items[video_id],
            details={}
        )
    
    await  seg_milvus_client.close()
    logger.info("Completed Milvus persistence for segment captions with %d records", len(results))
    return results



@task(
    name="aggregate_results",
    description="Aggregate all processing results into final manifest",
    
)
async def aggregate_results_task(
    run_id: str,
    videos: list[VideoArtifact],
    autoshots: list[AutoshotArtifact],
    asrs: list[ASRArtifact],
    images: list[ImageArtifact],
    segment_captions: list[SegmentCaptionArtifact],
    image_captions: list[ImageCaptionArtifact],
    image_embeddings: list[ImageEmbeddingArtifact],
    text_caption_embeddings: list[TextCaptionEmbeddingArtifact],
    text_segment_embeddings: list[TextCapSegmentEmbedArtifact],
):
    logger = get_run_logger()
    

    manifest = {
        "run_id": run_id,
        "completed_at": datetime.now().isoformat(),
        "summary": {
            "videos": len(videos),
            "autoshots": len(autoshots),
            "asrs": len(asrs),
            "images": len(images),
            "segment_captions": len(segment_captions),
            "image_captions": len(image_captions),
            "image_embeddings": len(image_embeddings),
            "text_caption_embeddings": len(text_caption_embeddings),
            "text_segment_embeddings": len(text_segment_embeddings),
        },
    }
    summary_table = "\n".join(
        f"| {key.replace('_', ' ').title()} | {value} |"
        for key, value in manifest["summary"].items()
    )
    summary_markdown = (
        f"# Video Processing Summary\n\n"
        f"* **Run ID:** `{run_id}`\n"
        f"* **Completed at:** {manifest['completed_at']}\n\n"
        "| Artifact | Count |\n| --- | ---: |\n"
        f"{summary_table}\n"
    )

    await acreate_markdown_artifact(
        description="Aggregated metrics for the video processing run",
        markdown=summary_markdown,
    )

    
    logger.info("Aggregated manifest for run %s with summary %s", run_id, manifest["summary"])
    return manifest


@flow(
    name="complete-video-processing-pipeline",
    description="End-to-end video processing with parallel task execution",
    persist_result=True,
    log_prints=True
)
async def video_processing_flow(
    video_files: list[tuple[str,str]],
    user_id: str,
    run_id:str,
)-> dict[str, Any] | None:
    
    async def ensure_state_initialized() -> None:
        state = AppState()
        if state.progress_client is not None:
            return

        from core.config.storage import minio_settings, postgre_settings, milvus_settings
        from core.config.task_config import (
            timage_processing_conf,
            tllm_conf,
            t_i_embed_conf,
            t_t_embed_conf,
            tautoshot_conf,
            tasr_conf,
            consule_conf,
            tracker_conf,
        )
        from core.storage import StorageClient
        from core.pipeline.tracker import ArtifactTracker
        from core.artifact.persist import ArtifactPersistentVisitor
        from core.management.cleanup import ArtifactDeleter
        from core.management.progress import HTTPProgressTracker
        from core.clients.milvus_client import (
            ImageMilvusClient,
            SegmentCaptionEmbeddingMilvusClient,
        )
        from core.config.milvus_index_config import (
            image_caption_dense_conf,
            image_visual_dense_conf,
            image_caption_sparse_conf,
            segment_caption_dense_conf,
            segment_caption_sparse_conf,
        )

        storage_client = StorageClient(settings=minio_settings)
        tracker = ArtifactTracker(database_url=postgre_settings.database_url)
        await tracker.initialize()
        visitor = ArtifactPersistentVisitor(minio_client=storage_client, tracker=tracker)

        from core.clients.base import ClientConfig
        base_client_config = ClientConfig(
            timeout_seconds=consule_conf.timeout_seconds,
            max_retries=consule_conf.max_retries,
            retry_min_wait=consule_conf.retry_min_wait,
            retry_max_wait=consule_conf.retry_max_wait,
            consul_host=consule_conf.host,
            consul_port=consule_conf.port,
        )

        state.video_ingestion_task = VideoIngestionTask(artifact_visitor=visitor)
        state.autoshot_task = AutoshotProcessingTask(
            artifact_visitor=visitor,
            model_name=tautoshot_conf.model_name,
            device=tautoshot_conf.device,
        )
        state.asr_task = ASRProcessingTask(
            artifact_visitor=visitor,
            model_name=tasr_conf.model_name,
            device=tasr_conf.device,
        )
        state.image_processing_task = ImageProcessingTask(
            artifact_visitor=visitor,
            num_img_per_segment=timage_processing_conf.num_img_per_segment,
            upload_concurrency=timage_processing_conf.upload_concurrency,
        )
        state.segment_caption_llm_task = SegmentCaptionLLMTask(
            artifact_visitor=visitor,
            image_per_segments=tllm_conf.image_per_segments,
            model_name=tllm_conf.model_name,
            device=tllm_conf.device,
            batch_size=tllm_conf.batch_size
        )
        state.image_caption_llm_task = ImageCaptionLLMTask(
            artifact_visitor=visitor,
            model_name=tllm_conf.model_name,
            device=tllm_conf.device,
            batch_size=tllm_conf.batch_size
        )
        state.image_embedding_task = ImageEmbeddingTask(
            artifact_visitor=visitor,
            batch_size=t_i_embed_conf.batch_size,
            model_name=t_i_embed_conf.model_name,
            device=t_i_embed_conf.device,
        )
        state.text_image_caption_embedding_task = TextImageCaptionEmbeddingTask(
            artifact_visitor=visitor,
            batch_size=t_t_embed_conf.batch_size,
            model_name=t_t_embed_conf.model_name,
            device=t_t_embed_conf.device,
        )
        state.text_caption_segment_embedding_task = TextCaptionSegmentEmbeddingTask(
            artifact_visitor=visitor,
            batch_size=t_t_embed_conf.batch_size,
            model_name=t_t_embed_conf.model_name,
            device=t_t_embed_conf.device,
        )

        state.image_embedding_milvus_task = ImageEmbeddingMilvusTask(
            artifact_visitor=visitor,
            ingest_batch_size=500,
        )
        state.text_segment_caption_milvus_task = TextSegmentCaptionMilvusTask(
            artifact_visitor=visitor,
            ingest_batch_size=500,
        )

        state.image_milvus_client = ImageMilvusClient(
            host=milvus_settings.host,
            port=milvus_settings.port,
            collection_name="image_milvus",
            user=milvus_settings.user,  # type: ignore
            password=milvus_settings.password,  # type: ignore
            db_name=milvus_settings.db_name,
            timeout=milvus_settings.time_out,
            visual_index_config=image_visual_dense_conf,
            caption_dense_index_config=image_caption_dense_conf,
            caption_sparse_index_config=image_caption_sparse_conf,
        )
        state.seg_milvus_client = SegmentCaptionEmbeddingMilvusClient(
            host=milvus_settings.host,
            port=milvus_settings.port,
            collection_name="segment_milvus",
            user=milvus_settings.user,  # type: ignore
            password=milvus_settings.password,  # type: ignore
            db_name=milvus_settings.db_name,
            timeout=milvus_settings.time_out,
            visual_index_config=None,
            caption_dense_index_config=segment_caption_dense_conf,
            caption_sparse_index_config=segment_caption_sparse_conf,
        )

        # Deleter and final wiring
        state.base_client_config = base_client_config
        state.progress_client = HTTPProgressTracker(
            base_url=tracker_conf.base_url,
            endpoint=tracker_conf.endpoint,
        )
        _ = ArtifactDeleter(
            tracker=tracker,
            storage=storage_client,
            image_client=state.image_milvus_client,
            text_seg_client=state.seg_milvus_client,
        )

    await ensure_state_initialized()

    run_logger = get_run_logger()
    context = get_run_context()
    flow_run_id = context.flow_run.id if context else "unknown" #type:ignore
    run_logger.info(f"Starting video processing flow for run_id={run_id}")

    try:
        video_input = VideoInput(files=video_files, user_id=user_id)
        videos = entry_video_ingestion.submit(video_uploads=video_input)
        

        
        run_logger.info("Stage 2: Parallel Autoshot and Processing")
        run_logger.info("Stage 2: Parallel Autoshot and Processing")
        autoshot_future = autoshot_task.submit(
            videos=videos  #type:ignore
        )   
        run_logger.info("Stage 2: Parallel Autoshot and Processing")
        asr_future = asr_task.submit(
            videos=videos  #type:ignore
        )
        run_logger.info("Stage 2: Parallel Autoshot and Processing")

        autoshot_artifacts =  autoshot_future.result()
        asr_artifacts  =  cast(list[ASRArtifact], asr_future.result())

        run_logger.info(
            f"Completed parallel processing: "
            f"{len(autoshot_artifacts)} autoshots, {len(asr_artifacts)} transcripts" #type:ignore
        )

        run_logger.info(
            "Stage 3.1: Running LLM Segmentation Caption + Segmentation Caption Embedding"
        )

        segmentation_captions= segment_caption_task.submit(
            autoshots=autoshot_artifacts, #type:ignore
            asrs=asr_artifacts,
        )
        

        text_segmentation_embeddings = segment_text_caption_embedding_task.submit(
            segment_captions=segmentation_captions
        )

        run_logger.info(f"Stage 3.2: Running Image caption -> Image embedding + Image caption")

        images_artifact_future = image_processing_task.submit(
            autoshots=autoshot_artifacts #type:ignore
        )
        
        image_captions_future = image_caption_task.submit(
            images=images_artifact_future 
        )
        image_embeddings_future = image_embedding_task.submit(
            images=images_artifact_future
        )

        text_caption_embedding_future = text_image_caption_embedding_task.submit(
            captions=image_captions_future
        )

        segment_captions =  cast(list[SegmentCaptionArtifact], segmentation_captions.result())
        text_segment_embeddings =  cast(list[TextCapSegmentEmbedArtifact], text_segmentation_embeddings.result())

        images =  cast(list[ImageArtifact], images_artifact_future.result())
        image_captions =  cast(list[ImageCaptionArtifact], image_captions_future.result())
        image_embeddings =  cast(list[ImageEmbeddingArtifact], image_embeddings_future.result())
        text_caption_embeddings =  cast(list[TextCaptionEmbeddingArtifact], text_caption_embedding_future.result())

        run_logger.info(
            f"Completed Stage 3 branches: "
            f"{len(segment_captions)} segment captions → {len(text_segment_embeddings)} segment embeddings, "
            f"{len(images)} images → ({len(image_captions)} image captions + {len(image_embeddings)} image embeddings) → {len(text_caption_embeddings)} text caption embeddings"
        )
        

        image_embed_milvus_future = image_embedding_milvus_persist_task.submit(
            image_caption_embeddings=(image_embeddings, text_caption_embeddings)
        )
        image_embed_milvus_result = image_embed_milvus_future.result()

        segment_caption_milvus_future = text_segment_caption_milvus_persist_task.submit(
            text_segment_embeddings=text_segment_embeddings
        )
        segment_caption_milvus_result = segment_caption_milvus_future.result()

        progress_client = AppState().progress_client
        for video_id, _ in video_files:
            await progress_client.trigger_http_not_throttle(video_id=video_id) 

        run_logger.info("Stage 5: Aggregating results")
        manifest_future = aggregate_results_task.submit(
            run_id=str(flow_run_id),
            videos=videos,  #type:ignore
            autoshots=autoshot_artifacts, #type:ignore
            asrs=asr_artifacts,
            images=images,
            segment_captions=segment_captions,
            image_captions=image_captions,
            image_embeddings=image_embeddings,
            text_caption_embeddings=text_caption_embeddings,
            text_segment_embeddings=text_segment_embeddings,
        )
        manifest = cast(dict, manifest_future.result())

        run_logger.info(f"Pipeline completed successfully for flow_run_id={flow_run_id} | run_id={run_id}")
        return manifest
    except KeyboardInterrupt:
        pc = AppState().progress_client
        if pc is not None:
            pc.clear_video_progress_cache()

    except Exception as e:
        pc = AppState().progress_client
        if pc is not None:
            pc.clear_video_progress_cache()
        run_logger.exception("Pipeline failed: %s", e)
        raise e
    
     
