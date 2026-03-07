from typing import cast, Any
from datetime import datetime

from prefect import flow, get_run_logger
from prefect.artifacts import acreate_markdown_artifact
from prefect.context import get_run_context
from prefect_dask import DaskTaskRunner  # type: ignore

from video_pipeline.task.video.main import VideoInput, video_reg_task, VideoRegistryTask, VideoArtifact
from video_pipeline.task.autoshot.main import autoshot_task, AutoshotTask, AutoshotArtifact
from video_pipeline.task.asr.main import asr_chunk_task, ASRTask
from video_pipeline.task.image_extraction.main import image_chunk_task, ImageExtractionTask
# from video_pipeline.task.image_caption.main import image_caption_chunk_task, ImageCaptionTask
# from video_pipeline.task.image_embedding.main import image_embedding_chunk_task, ImageEmbeddingTask
# from video_pipeline.task.image_ocr.main import image_ocr_chunk_task, ImageOCRTask
# from video_pipeline.task.image_caption_embedding.main import image_caption_embedding_chunk_task
# from video_pipeline.task.image_caption_multimodal_embedding.main import image_caption_multimodal_embedding_chunk_task
from video_pipeline.flow.subtask import preprocess_video_task
from video_pipeline.config import get_settings
from video_pipeline.core.client.progress import HTTPProgressTracker, StageRegistry, RunStatus



@flow(
    name="Single Video Processing Flow",
    description="Process a single video through the complete pipeline",
    log_prints=True,
    persist_result=True,
    retries=2,
    retry_delay_seconds=5,
    task_runner=DaskTaskRunner( #type:ignore
        cluster_kwargs=get_settings().dask.to_cluster_kwargs()
    ),
)
async def single_video_processing_flow(
    video_id: str,
    user_id: str,
    video_file_path: str,
    additional_flow_description: str = "",
    tracker_url: str = "http://host.docker.internal:8123",
) -> dict[str, Any]:
    logger = get_run_logger()
    context = get_run_context()
    flow_run_id = (
        str(context.flow_run.id) if context and context.flow_run else "unknown" #type:ignore
    ) 
    flow_run_name = context.flow_run.name if context and context.flow_run else "unknown" #type:ignore

    video_input = VideoInput(
        user_id=user_id,
        video_id=video_id,
        video_s3_url=video_file_path
    )

    tracker: HTTPProgressTracker | None = None
    if tracker_url:
        tracker = HTTPProgressTracker(base_url=tracker_url)
        await tracker.start_video(video_id, StageRegistry.all_stage_names())

    logger.info(
        f"[Flow:{flow_run_name}] Starting pipeline | "
        f"video_id={video_id} user_id={user_id} path={video_file_path}"
    )

    try:
        logger.info(f"[Flow:{flow_run_name}] Submitting video_reg_task to Dask")
        video_fut = video_reg_task.submit(video_input)
        logger.info(f"[Flow:{flow_run_name}] video_reg_task submitted — future pending")
        video_artifact = cast(VideoArtifact, video_fut.result())
        await VideoRegistryTask.summary_artifact(video_artifact)
        if tracker:
            await tracker.complete_stage(video_id, VideoRegistryTask.__name__)

        logger.info(f"[Flow:{flow_run_name}] Submitting autoshot_task (depends on video_reg_task)")
        shots_fut = autoshot_task.submit(
            video_artifact,
        )
        logger.info(f"[Flow:{flow_run_name}] autoshot_task submitted — waiting on video_reg_task future")

        logger.info(f"[Flow:{flow_run_name}] Awaiting video_reg_task result...")
        logger.info(
            f"[Flow:{flow_run_name}] video_reg_task done | "
            f"video_id={video_artifact.video_id} fps={video_artifact.fps} "
            f"ext={video_artifact.video_extension} duration={video_artifact.metadata.get('duration')}s" #type:ignore
        )

        logger.info(f"[Flow:{flow_run_name}] Awaiting autoshot_task result...")
        shot_result = cast(AutoshotArtifact, shots_fut.result())# single AutoshotArtifact
        await AutoshotTask.summary_artifact(shot_result)
        if tracker:
            await tracker.complete_stage(video_id, AutoshotTask.__name__)

        n_shots = len(shot_result.metadata.get("segments", []))  # type: ignore
        logger.info(
            f"[Flow:{flow_run_name}] autoshot_task done | "
            f"{n_shots} shot(s) detected"
        )

        logger.info(f"[Flow:{flow_run_name}] Submitting preprocess_video_task")
        preprocess_fut = preprocess_video_task.submit(shot_result, wait_for=[shots_fut])  # type: ignore
        asr_batches, image_batches = preprocess_fut.result()  # type: ignore
        n_asr = sum(len(b) for b in asr_batches)
        n_img = sum(len(b) for b in image_batches)

        logger.info(
            f"[Flow:{flow_run_name}] preprocess_video_task done | "
            f"{n_asr} audio chunk(s) in {len(asr_batches)} ASR batch(es) | "
            f"{n_img} frame index(es) in {len(image_batches)} image batch(es)"
        )

        logger.info(
            f"[Flow:{flow_run_name}] Fanning out "
            f"{len(asr_batches)} asr_chunk_task(s) and {len(image_batches)} image_chunk_task(s)"
        )

        asr_batch_futures = asr_chunk_task.map( #type:ignore
            asr_batches,  
            wait_for=[preprocess_fut]  
        ) 
        image_batch_futures = image_chunk_task.map( #type:ignore
            image_batches, 
            wait_for=[preprocess_fut]  
        )

        asr_batch_results = asr_batch_futures.result()
        image_batch_results = image_batch_futures.result()

        asr_results = [artifact for batch in asr_batch_results for artifact in batch]
        image_results = [artifact for batch in image_batch_results for artifact in batch]


        await ImageExtractionTask.summary_artifact(image_results)
        if tracker:
            await tracker.complete_stage(video_id, ImageExtractionTask.__name__)
        await ASRTask.summary_artifact(asr_results)
        if tracker:
            await tracker.complete_stage(video_id, ASRTask.__name__)

        logger.info(
            f"[Flow:{flow_run_name}] Extraction done | "
            f"{len(asr_results)} ASR artifact(s) | {len(image_results)} image artifact(s)"
        )

        analysis_batch_size: int = 10
        analysis_batches = [
            image_results[i:i + analysis_batch_size]
            for i in range(0, len(image_results), analysis_batch_size)
        ]

        logger.info(
            f"[Flow:{flow_run_name}] Fanning out "
            f"{len(analysis_batches)} batch(es) each for caption, embedding, and OCR"
        )

        completed_at = datetime.now().isoformat()
        manifest = {
            "run_id": flow_run_id,
            "flow_run_id": flow_run_id,
            "video_id": video_id,
            "completed_at": completed_at,
            "summary": {
                "shots": n_shots,
                "asr_artifacts": len(asr_results),
                "image_artifacts": len(image_results),
                # "caption_embedding_artifacts": len(caption_embedding_results),
                # "caption_multimodal_embedding_artifacts": len(caption_multimodal_embedding_results),
            },
        }
        await acreate_markdown_artifact(
            description=f"Processing summary for video {video_id}",
            markdown=(
                f"# Video Processing Complete\n\n"
                f"**Video ID:** `{video_id}`\n"
                f"**Flow Run:** `{flow_run_id}`\n"
                f"**Completed:** {completed_at}\n\n"
                f"## Summary\n\n"
                f"| Artifact | Count |\n"
                f"|----------|-------|\n"
                f"| Shots | {n_shots} |\n"
                f"| ASR Artifacts | {len(asr_results)} |\n"
                f"| Image Artifacts | {len(image_results)} |\n"
                # f"| Caption Artifacts | {len(caption_results)} |\n"
                # f"| Embedding Artifacts | {len(embedding_results)} |\n"
            ),
        )

        if tracker:
            await tracker.complete_run(video_id)
        logger.info(f"Pipeline complete for video: {video_id}")

        return manifest
    except Exception as e:
        if tracker:
            await tracker.complete_run(video_id, RunStatus.FAILED, str(e))
        logger.exception(f"Pipeline failed for video {video_id}: {e}")
        raise