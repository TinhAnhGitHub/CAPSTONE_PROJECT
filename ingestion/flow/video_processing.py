from __future__ import annotations
from typing import Any, cast
from pathlib import Path
from fastapi import UploadFile

from datetime import datetime
from core.management.progress import ProgressTracker, ProcessingStage
from prefect import flow, task, get_run_logger
from prefect.futures import PrefectFuture
from prefect.cache_policies import INPUTS, TASK_SOURCE, NO_CACHE
from prefect.concurrency.asyncio import concurrency
from prefect.artifacts import create_table_artifact, create_markdown_artifact
from datetime import timedelta

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



from core.config.storage import minio_settings, postgre_settings
from core.config.logging import run_logger


from core.clients.autoshot_client import AutoshotClient
from core.clients.asr_client import ASRClient
from core.clients.llm_client import LLMClient
from core.clients.image_embed_client import ImageEmbeddingClient
from core.clients.text_embed_client import TextEmbeddingClient

from core.clients.milvus_client import ImageEmbeddingMilvusClient, TextCaptionEmbeddingMilvusClient, SegmentCaptionEmbeddingMilvusClient
from core.lifespan import AppState



@task(
    name='Video registry',
    description="This task will take uploaded videos, and persist into the tracker + minio S3",
    cache_policy=NO_CACHE,
    
)
async def entry_video_ingestion(
    video_uploads: VideoInput,
) -> list[VideoArtifact]:
    task_instance = AppState().video_ingestion_task
    preprocessed_input = await task_instance.preprocess(input_data=video_uploads)
    video_artifacts = []
    async for result in task_instance.execute(preprocessed_input, None):
        processed = await task_instance.postprocess(result)
        video_artifacts.append(processed)        
    return video_artifacts


@task(
    name="Autoshot Processing",
    description="Given a list of videos, detect the shot  segment boundary",
    cache_policy=INPUTS + TASK_SOURCE,
    cache_expiration=timedelta(days=30),
    persist_result=True,
    
)
async def autoshot_task(
    videos: list[VideoArtifact],
):
    task_instance = AppState().autoshot_task
    client_config =  AppState().base_client_config
    async with AutoshotClient(
        config=client_config
    ) as client:
        await client.load_model(
            model_name=task_instance.config.model_name,
            device=task_instance.config.device,
        )
        preprocessed = await task_instance.preprocess(videos)
        run_logger.info(f"Len of preprocessed: {len(preprocessed)}")
        run_logger.info(f"Preprocessed data: {preprocessed[0]}")

        results = []
        async for result in task_instance.execute(preprocessed, client):
            processed = await task_instance.postprocess(result)
            results.append(processed)
        
        await client.unload_model()
    return results


@task(
    name="ASR Processing",
    description="Given a list of videos, extracting the corresponding ASR",
    persist_result=True,
    
)
async def asr_task(
    videos: list[VideoArtifact],
):
    task_instance = AppState().asr_task
    client_config = AppState().base_client_config
    async with ASRClient(config=client_config) as client:
        await client.load_model(
            model_name=task_instance.config.model_name,
            device=task_instance.config.device,
        )
        preprocessed = await task_instance.preprocess(videos)
        results = []
        async for result in task_instance.execute(preprocessed, client):
            postprocessed_result = await task_instance.postprocess(result)
            results.append(postprocessed_result)

        await client.unload_model()
        return results


@task(
    name='Image Extraction',
    description="Extract frames from video segments",
    
)
async def image_processing_task(
    autoshots: list[AutoshotArtifact],
)-> list[ImageArtifact]:
    task_instance = AppState().image_processing_task
    preprocessed = await task_instance.preprocess(autoshots)
    results = []
    async for result in task_instance.execute(preprocessed, None):
        processed = await task_instance.postprocess(result)
        results.append(processed)
    
    return results


@task(
    name='Segmentation Caption Generation',
    description="Generate captions for video segments using LLM",
    
)
async def segment_caption_task(
    autoshots: list[AutoshotArtifact],
    asrs: list[ASRArtifact],
) -> list[SegmentCaptionArtifact]:
     
    task_instance =  AppState().segment_caption_llm_task
    client_config =  AppState().base_client_config

    input_data = ShotASRInput(
        list_asrs=asrs,
        lists_autoshots=autoshots
    )
    async with LLMClient(config=client_config)  as client:
        await client.load_model(
            model_name=task_instance.config.model_name,
            device=task_instance.config.device,
        )
        preprocessed_data = await task_instance.preprocess(input_data)
        results = []
        async for result in task_instance.execute(preprocessed_data, client):
            processed = await task_instance.postprocess(result)
            results.append(processed)
        
        await client.unload_model()
        return results


@task(
    name='Image Caption Generation',
    description='Generate captions for individual images using LLM',
    
)
async def image_caption_task(
    images: list[ImageArtifact] | PrefectFuture,
) -> list[ImageCaptionArtifact]:
    task_instance =  AppState().image_caption_llm_task
    client_config =  AppState().base_client_config
    async with LLMClient(config=client_config) as client:
        await client.load_model(
            model_name=task_instance.config.model_name,
            device=task_instance.config.device,
        )
        preprocessed = await task_instance.preprocess(cast(list[ImageArtifact],images))
        results = []
        async for result in task_instance.execute(preprocessed, client):
            postprocessed = await task_instance.postprocess(result)
            results.append(postprocessed)
        await client.unload_model()
        return results  


@task(
    name='Image Embedding generation',
    description='Generate embeddings for images',
    
)
async def image_embedding_task(
    images: list[ImageArtifact] | PrefectFuture,
) -> list[ImageEmbeddingArtifact]:
    task_instance =  AppState().image_embedding_task
    client_config    =  AppState().base_client_config
    async with ImageEmbeddingClient(config=client_config) as client:
        await client.load_model(
            model_name=task_instance.config.model_name,
            device=task_instance.config.device,
        )
        preprocessed = await task_instance.preprocess(cast(list[ImageArtifact],images))
        results = []
        async for result in task_instance.execute(preprocessed, client):
            postprocessed = await task_instance.postprocess(result)
            results.append(postprocessed)
        await client.unload_model()
        return results #type:ignore



@task(
    name="segment_caption_embedding",
    description="Generate embeddings for image captions",
    
)
async def  segment_text_caption_embedding_task(
    segment_captions: list[SegmentCaptionArtifact] | PrefectFuture,
) ->list[TextCapSegmentEmbedArtifact]:
    task_instance =  AppState().text_caption_segment_embedding_task
    client_config =  AppState().base_client_config

    async with TextEmbeddingClient(config=client_config) as client:
        await client.load_model(
            model_name=task_instance.config.model_name,
            device=task_instance.config.device,
        )
        preprocessed = await task_instance.preprocess(segment_captions) #type:ignore
        results = []
        async for result in task_instance.execute(preprocessed, client):
            processed = await task_instance.postprocess(result)
            results.append(processed)
        await client.unload_model()
        return results  #type:ignore



@task(
    name='text caption embedding',
    description="Generate embedding from text",
    retries=0,
    
)
async def text_image_caption_embedding_task(
    captions: list[ImageCaptionArtifact] |PrefectFuture,
)-> list[TextCaptionEmbeddingArtifact]:
    task_instance =  AppState().text_image_caption_embedding_task
    client_config =  AppState().base_client_config

    async with TextEmbeddingClient(config=client_config) as client:
        await client.load_model(
            model_name=task_instance.config.model_name,
            device=task_instance.config.device,
        )
        preprocessed = await task_instance.preprocess(cast(list[ImageCaptionArtifact], captions))
        results = []
        async for result in task_instance.execute(preprocessed, client):
            processed = await task_instance.postprocess(result)
            results.append(processed)
        await client.unload_model()
        return results  #type:ignore



@task(
    name='Image embedding Milvus Persist',
    description='Persist image embedding into milvus',
    
)
async def image_embedding_milvus_persist_task(
    image_embeddings: list[ImageEmbeddingArtifact] | PrefectFuture,
):
    task_instance =  AppState().image_embedding_milvus_task
    milvus_collection_config = AppState().image_embedding_milvus_config
    
    print(f"{task_instance.config.model_dump(mode='json')}")
    async with ImageEmbeddingMilvusClient(
        config_collection=milvus_collection_config,
        host=task_instance.config.host,
        port=task_instance.config.port,
        user=task_instance.config.user, #type:ignore
        password=task_instance.config.password, #type:ignore
        db_name=task_instance.config.db_name,
        timeout=task_instance.config.time_out
    ) as client:
        print("Before milvus ingestion")
        preprocessed = await task_instance.preprocess(cast(list[ImageEmbeddingArtifact], image_embeddings))
        results = []
        async for result in task_instance.execute(preprocessed, client):
            processed = await task_instance.postprocess(result)
            results.append(processed)
        return results



@task(
    name='Text Image Caption Milvus Persist',
    description='Persist text caption embeddings into Milvus vector database',
    
)
async def text_image_caption_milvus_persist_task(
    text_caption_embeddings: list[TextCaptionEmbeddingArtifact] | PrefectFuture,
):
    task_instance =  AppState().text_image_caption_milvus_task
    milvus_config =  AppState().text_image_caption_milvus_config
    async with TextCaptionEmbeddingMilvusClient(
        config_collection=milvus_config,
        host=task_instance.config.host,
        port=task_instance.config.port,
        user=task_instance.config.user,#type:ignore
        password=task_instance.config.password,#type:ignore
        db_name=task_instance.config.db_name,
        timeout=task_instance.config.time_out
    ) as client:
        preprocessed = await task_instance.preprocess(cast(list[TextCaptionEmbeddingArtifact], text_caption_embeddings))
        results = []
        async for result in task_instance.execute(preprocessed, client):
            processed = await task_instance.postprocess(result)
            results.append(processed)
        return results



@task(
    name='Text Segment Caption Milvus Persist',
    description='Persist segment caption embeddings into Milvus vector database',
    
)
async def text_segment_caption_milvus_persist_task(
    text_segment_embeddings: list[TextCapSegmentEmbedArtifact] | PrefectFuture,
):
    task_instance =  AppState().text_segment_caption_milvus_task
    milvus_config =  AppState().text_segment_caption_milvus_config
    async with SegmentCaptionEmbeddingMilvusClient(
        config_collection=milvus_config,
        host=task_instance.config.host,
        port=task_instance.config.port,
        user=task_instance.config.user,#type:ignore
        password=task_instance.config.password,#type:ignore
        db_name=task_instance.config.db_name,
        timeout=task_instance.config.time_out
    ) as client:

        print("Before milvus")
        preprocessed = await task_instance.preprocess(cast(list[TextCapSegmentEmbedArtifact], text_segment_embeddings))
        results = []
        
        async for result in task_instance.execute(preprocessed, client):
            processed = await task_instance.postprocess(result)
            results.append(processed)
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
        }
    }
    
    run_logger.info(f"Pipeline completed successfully: {manifest['summary']}")
    return manifest


@flow(
    name="complete-video-processing-pipeline",
    description="End-to-end video processing with parallel task execution",
    persist_result=True,
    log_prints=True
)
async def video_processing_flow(
    video_files: list[UploadFile],
    run_id: str,
    user_id: str,
)-> dict[str, Any] | None:
    
    """
    Complete Video Processing pipeline with parallel execution
    Args:
        video_files: List of Uploaded video files
        run_id: Unique Identifier
        metadata: Optional Metadata about the run
    
    Returns:
        Final Processing manifest with all artifacts
    """
    run_logger.info(f"Starting video processing flow for run_id={run_id}\n")    
    
    
    

    try:
        run_logger.info("Stage 1: Video Ingestion - Register the videos")
        video_input = VideoInput(files=video_files, user_id=user_id)
        video_futures = entry_video_ingestion.submit(
            video_uploads=video_input,
        )
        videos = video_futures.result()
        run_logger.info(f"Ingested {len(videos)} videos")  #type:ignore


        
        run_logger.info("Stage 2: Parallel Autoshot and Processing")

        run_logger.info(f"Videos output: {videos}")
        autoshot_future = autoshot_task.submit(
            videos=videos #type:ignore
        )
        asr_future = asr_task.submit(
            videos=videos #type:ignore
        )

        autoshot_artifacts =  cast(list[AutoshotArtifact], autoshot_future.result())
        asr_artifacts  =  cast(list[ASRArtifact], asr_future.result())

        run_logger.info(
            f"Completed parallel processing: "
            f"{len(autoshot_artifacts)} autoshots, {len(asr_artifacts)} transcripts" #type:ignore
        )

        run_logger.info(
            "Stage 3.1: Running LLM Segmentation Caption + Segmentation Caption Embedding"
        )

        segmentation_captions= segment_caption_task.submit(
            autoshots=autoshot_artifacts,
            asrs=asr_artifacts,
        )
        

        text_segmentation_embeddings = segment_text_caption_embedding_task.submit(
            segment_captions=segmentation_captions
        )

        run_logger.info(f"Stage 3.2: Running Image caption -> Image embedding + Image caption")

        images_artifact_future = image_processing_task.submit(
            autoshots=autoshot_artifacts
        )
        run_logger.debug(f'{images_artifact_future=}')
        
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
            image_embeddings=image_embeddings
        )
        image_embed_milvus_result = image_embed_milvus_future.result()

        text_caption_milvus_future = text_image_caption_milvus_persist_task.submit(
            text_caption_embeddings=text_caption_embeddings
        )
        text_caption_milvus_result = text_caption_milvus_future.result()


        segment_caption_milvus_future = text_segment_caption_milvus_persist_task.submit(
            text_segment_embeddings=text_segment_embeddings
        )
        segment_caption_milvus_result = segment_caption_milvus_future.result()

        # run_logger.info("Stage 5: Aggregate Results")
        # final_manifest = await aggregate_results_task(
        #     run_id=run_id,
        #     videos=videos,
        #     autoshots=autoshot_artifacts,
        #     asrs=asr_artifacts,
        #     images=images,
        #     segment_captions=segment_captions,
        #     image_captions=image_captions,
        #     image_embeddings=image_embeddings,
        #     text_caption_embeddings=text_caption_embeddings,
        #     text_segment_embeddings=text_segment_embeddings,
        # )

        # return final_manifest

    except KeyboardInterrupt:
        AppState().progress_tracker.clear_video_progress_cache()

    except Exception as e:
        AppState().progress_tracker.clear_video_progress_cache()
        run_logger.exception(f"Pipeline failed: {str(e)}")
        raise
    
     
