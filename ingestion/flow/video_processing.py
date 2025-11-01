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

from core.config.logging import run_logger


from core.clients.autoshot_client import AutoshotClient
from core.clients.asr_client import ASRClient
from core.clients.llm_client import LLMClient
from core.clients.image_embed_client import ImageEmbeddingClient
from core.clients.text_embed_client import TextEmbeddingClient
from core.lifespan import AppState

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



GPU_TASK_TAG = 'gpu-heavy'
LLM_TASK_TAG = 'llm-service'
EMBED_TASK_TAG = 'embedding-service'
PERSIST_TASK_TAG = "milvus-persist"



@task(
    name='Video registry',
    description="This task will take uploaded videos, and persist into the tracker + minio S3",
    tags=["video-ingest"],
)
async def entry_video_ingestion(
    video_uploads: VideoInput,
) -> list[VideoArtifact]:
    task_instance = AppState().video_ingestion_task
    progress_client = AppState().progress_client

    preprocessed_input = await task_instance.preprocess(input_data=video_uploads)
    video_artifacts = []

    for upload in video_uploads:
        await progress_client.start_video(
            video_id=upload[0]
        )
    async for result in task_instance.execute(preprocessed_input, None):
        processed = await task_instance.postprocess(result)
        video_artifacts.append(processed)        
        response = await progress_client.start_video(video_id=processed.artifact_id)
        print(response)
    return video_artifacts


@task(
    name="Autoshot Processing",
    description="Given a list of videos, detect the shot  segment boundary",
    cache_policy=INPUTS + TASK_SOURCE,
    cache_expiration=timedelta(days=30),
    persist_result=True,
    tags=[GPU_TASK_TAG],
)
async def autoshot_task(
    videos: list[VideoArtifact],
):
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
    


@task(
    name="ASR Processing",
    description="Given a list of videos, extracting the corresponding ASR",
    persist_result=True,
    tags=[GPU_TASK_TAG],
    
)
async def asr_task(
    videos: list[VideoArtifact],
):
    task_instance = AppState().asr_task
    client_config = AppState().base_client_config
    progress_client = AppState().progress_client
    
    async with ASRClient(config=client_config) as client:
        await client.load_model(
            model_name=task_instance.kwargs.get('model_name'), #type:ignore
            device=task_instance.kwargs.get('device'), #type:ignore
        )
        preprocessed = await task_instance.preprocess(videos)
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
    return results



@task(
    name='Image Extraction',
    description="Extract frames from video segments",
    tags=["image-processing"],
)
async def image_processing_task(
    autoshots: list[AutoshotArtifact],
)-> list[ImageArtifact]:
    
    task_instance = AppState().image_processing_task
    preprocessed = await task_instance.preprocess(autoshots)
    progress_client = AppState().progress_client

    video_id2total_items = defaultdict(int)
    unique_video_id = list({x.related_video_id for v in preprocessed.values() for x in v})
    # preprocessed return each video -> list of images. So the total items would be 1 -> no smaller percentage progress
    for video_id in unique_video_id:
        video_id2total_items[video_id] += 1

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
    
    return results


@task(
    name='Segmentation Caption Generation',
    description="Generate captions for video segments using LLM",
    tags=[LLM_TASK_TAG],
)
async def segment_caption_task(
    autoshots: list[AutoshotArtifact],
    asrs: list[ASRArtifact],
) -> list[SegmentCaptionArtifact]:
     
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
        
        await client.unload_model()

    return results


@task(
    name='Image Caption Generation',
    description='Generate captions for individual images using LLM',
    tags=[LLM_TASK_TAG],
)
async def image_caption_task(
    images: list[ImageArtifact] | PrefectFuture,
) -> list[ImageCaptionArtifact]:
    task_instance =  AppState().image_caption_llm_task
    client_config =  AppState().base_client_config
    progress_client = AppState().progress_client
    async with LLMClient(config=client_config) as client:
        await client.load_model(
           model_name=task_instance.kwargs.get('model_name'), #type:ignore
            device=task_instance.kwargs.get('device'), #type:ignore
        )
        preprocessed = await task_instance.preprocess(cast(list[ImageArtifact],images))
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
        await client.unload_model()
    return results  


@task(
    name='Image Embedding generation',
    description='Generate embeddings for images',
    tags=[EMBED_TASK_TAG],
)
async def image_embedding_task(
    images: list[ImageArtifact] | PrefectFuture,
) -> list[ImageEmbeddingArtifact]:
    task_instance =  AppState().image_embedding_task
    client_config    =  AppState().base_client_config
    progress_client = AppState().progress_client

    async with ImageEmbeddingClient(config=client_config) as client:
        await client.load_model(
            model_name=task_instance.kwargs.get('model_name'), #type:ignore
            device=task_instance.kwargs.get('device'), #type:ignore
        )
        preprocessed = await task_instance.preprocess(cast(list[ImageArtifact],images))
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
    return results 



@task(
    name="segment_caption_embedding",
    description="Generate embeddings for image captions",
    tags=[EMBED_TASK_TAG],
)
async def  segment_text_caption_embedding_task(
    segment_captions: list[SegmentCaptionArtifact] | PrefectFuture,
) ->list[TextCapSegmentEmbedArtifact]:
    task_instance =  AppState().text_caption_segment_embedding_task
    client_config =  AppState().base_client_config
    progress_client = AppState().progress_client

    async with TextEmbeddingClient(config=client_config) as client:
        await client.load_model(
           model_name=task_instance.kwargs.get('model_name'), #type:ignore
            device=task_instance.kwargs.get('device'), #type:ignore
        )
        preprocessed = await task_instance.preprocess(segment_captions) #type:ignore
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
    return results  



@task(
    name='text caption embedding',
    description="Generate embedding from text",
    tags=[EMBED_TASK_TAG],
)
async def text_image_caption_embedding_task(
    captions: list[ImageCaptionArtifact] |PrefectFuture,
)-> list[TextCaptionEmbeddingArtifact]:
    task_instance =  AppState().text_image_caption_embedding_task
    client_config =  AppState().base_client_config
    progress_client = AppState().progress_client

    async with TextEmbeddingClient(config=client_config) as client:
        await client.load_model(
            model_name=task_instance.kwargs.get('model_name'), #type:ignore
            device=task_instance.kwargs.get('device'), #type:ignore
        )
        preprocessed = await task_instance.preprocess(cast(list[ImageCaptionArtifact], captions))
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
    return results  #type:ignore



@task(
    name='Image embedding Milvus Persist',
    description='Persist image embedding into milvus',
    tags=[PERSIST_TASK_TAG],
)
async def image_embedding_milvus_persist_task(
    image_caption_embeddings: tuple[list[ImageEmbeddingArtifact], list[TextCaptionEmbeddingArtifact]] | PrefectFuture,
):
    task_instance =   AppState().image_embedding_milvus_task

    image_milvus_client = AppState().image_milvus_client
    progress_client = AppState().progress_client
    await image_milvus_client.connect()

    preprocessed = await task_instance.preprocess(cast(tuple[list[ImageEmbeddingArtifact], list[TextCaptionEmbeddingArtifact]], image_caption_embeddings))

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
    
    return results

@task(
    name='Text Segment Caption Milvus Persist',
    description='Persist segment caption embeddings into Milvus vector database',
    tags=[PERSIST_TASK_TAG],
)
async def text_segment_caption_milvus_persist_task(
    text_segment_embeddings: list[TextCapSegmentEmbedArtifact] | PrefectFuture,
):
    task_instance =  AppState().text_segment_caption_milvus_task
    seg_milvus_client =  AppState().seg_milvus_client
    progress_client = AppState().progress_client
    await  seg_milvus_client.connect()
    
    preprocessed = await task_instance.preprocess(cast(list[TextCapSegmentEmbedArtifact], text_segment_embeddings))
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
    run_logger = get_run_logger()
    

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
        key=f"video-processing/{run_id}",
        description="Aggregated metrics for the video processing run",
        markdown=summary_markdown,
    )

    
    run_logger.info(f"Pipeline completed successfully: {manifest['summary']}")
    return manifest


@flow(
    name="complete-video-processing-pipeline",
    description="End-to-end video processing with parallel task execution",
    persist_result=True,
    log_prints=False
)
async def video_processing_flow(
    video_files: list[tuple[str,str]],
    user_id: str,
    run_id:str,
)-> dict[str, Any] | None:
    
    run_logger = get_run_logger()
    context = get_run_context()
    flow_run_id = context.flow_run.id if context else "unknown" #type:ignore
    run_logger.info(f"Starting video processing flow for run_id={run_id}")

    try:
        video_input = VideoInput(files=video_files, user_id=user_id)
        videos = entry_video_ingestion.submit(video_uploads=video_input)
        run_logger.info(f"Ingested {len(videos)} videos") #type:ignore

        
        run_logger.info("Stage 2: Parallel Autoshot and Processing")

        run_logger.info(f"Videos output: {videos}")
        autoshot_future = autoshot_task.submit(
            videos=videos #type:ignore
        )
        asr_future = asr_task.submit(
            videos=videos #type:ignore
        )

        video_result = cast(list[VideoArtifact], videos.result())

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
            image_caption_embeddings=(image_embeddings, text_caption_embeddings)
        )
        image_embed_milvus_result = image_embed_milvus_future.result()

        segment_caption_milvus_future = text_segment_caption_milvus_persist_task.submit(
            text_segment_embeddings=text_segment_embeddings
        )
        segment_caption_milvus_result = segment_caption_milvus_future.result()
        

        run_logger.info("Stage 5: Aggregating results")
        manifest_future = aggregate_results_task.submit(
            run_id=str(flow_run_id),
            videos=video_result, 
            autoshots=autoshot_artifacts,
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
        AppState().progress_client.clear_video_progress_cache()

    except Exception as e:
        AppState().progress_client.clear_video_progress_cache()
        run_logger.exception(f"Pipeline failed: {str(e)}")
        raise
    
     
