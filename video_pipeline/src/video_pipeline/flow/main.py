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
from video_pipeline.task.image_caption.main import image_caption_chunk_task, ImageCaptionTask
from video_pipeline.task.image_embedding.main import image_embedding_chunk_task, ImageEmbeddingTask
from video_pipeline.task.image_ocr.main import image_ocr_chunk_task, ImageOCRTask
from video_pipeline.task.image_caption_embedding.main import image_caption_embedding_chunk_task, ImageCaptionEmbeddingTask
from video_pipeline.task.image_caption_multimodal_embedding.main import image_caption_multimodal_embedding_chunk_task, ImageCaptionMultimodalEmbeddingTask
from video_pipeline.task.qdrant_indexing.main import (
    ImageQdrantIndexingTask,
    CaptionQdrantIndexingTask,
    image_qdrant_indexing_chunk_task,
    caption_qdrant_indexing_chunk_task,
)
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
    tracker_url: str = "",
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
            video_result, #type:ignore
        )
        logger.info(f"[Flow:{flow_run_name}] autoshot_task submitted — waiting on video_reg_task future")

        logger.info(f"[Flow:{flow_run_name}] Awaiting video_reg_task result...")
        logger.info(
            f"[Flow:{flow_run_name}] video_reg_task done | "
            f"video_id={video_result.video_id} fps={video_result.fps} " #type:ignore
            f"ext={video_result.video_extension} duration={video_result.metadata.get('duration')}s" #type:ignore
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

        caption_batch_futures = image_caption_chunk_task.map(  # type: ignore
            analysis_batches,
            wait_for=image_batch_futures,
        )
        embedding_batch_futures = image_embedding_chunk_task.map(  # type: ignore
            analysis_batches,
            wait_for=image_batch_futures,
        )
        ocr_batch_futures = image_ocr_chunk_task.map(  # type: ignore
            analysis_batches,
            wait_for=image_batch_futures,
        )

        caption_results = [a for batch in caption_batch_futures.result() for a in batch]
        await ImageCaptionTask.summary_artifact(caption_results)
        if tracker:
            await tracker.complete_stage(video_id, ImageCaptionTask.__name__)

        embedding_results = [a for batch in embedding_batch_futures.result() for a in batch]
        await ImageEmbeddingTask.summary_artifact(embedding_results)
        if tracker:
            await tracker.complete_stage(video_id, ImageEmbeddingTask.__name__)

        ocr_results = [a for batch in ocr_batch_futures.result() for a in batch]
        await ImageOCRTask.summary_artifact(ocr_results)
        if tracker:
            await tracker.complete_stage(video_id, ImageOCRTask.__name__)


        caption_batches = [
            caption_results[i:i + analysis_batch_size]
            for i in range(0, len(caption_results), analysis_batch_size)
        ]

        logger.info(
            f"[Flow:{flow_run_name}] Image analysis done | "
            f"{len(caption_results)} caption | {len(embedding_results)} embedding | "
            f"{len(ocr_results)} OCR — fanning out {len(caption_batches)} caption embedding batch(es)"
        )

        caption_embedding_futures = image_caption_embedding_chunk_task.map(  # type: ignore
            caption_batches,
            wait_for=caption_batch_futures,
        )
        caption_multimodal_embedding_futures = image_caption_multimodal_embedding_chunk_task.map(  # type: ignore
            caption_batches,
            wait_for=caption_batch_futures,
        )

        caption_embedding_results = [
            a for batch in caption_embedding_futures.result() for a in batch
        ]
        await ImageCaptionEmbeddingTask.summary_artifact(caption_embedding_results)
        if tracker:
            await tracker.complete_stage(video_id, ImageCaptionEmbeddingTask.__name__)

        caption_multimodal_embedding_results = [
            a for batch in caption_multimodal_embedding_futures.result() for a in batch
        ]
        await ImageCaptionMultimodalEmbeddingTask.summary_artifact(caption_multimodal_embedding_results)
        if tracker:
            await tracker.complete_stage(video_id, ImageCaptionMultimodalEmbeddingTask.__name__)

        logger.info(
            f"[Flow:{flow_run_name}] Embedding done | "
            f"{len(caption_embedding_results)} cap-text-embed | "
            f"{len(caption_multimodal_embedding_results)} cap-mm-embed — "
            f"fanning out Qdrant indexing"
        )

        # --- Qdrant indexing fan-out ---
        # Image embeddings → dense collection
        image_index_batches = [
            embedding_results[i:i + analysis_batch_size]
            for i in range(0, len(embedding_results), analysis_batch_size)
        ]
        image_index_futures = image_qdrant_indexing_chunk_task.map(  # type: ignore
            image_index_batches,
            wait_for=embedding_batch_futures,
        )

        # Caption embeddings → hybrid collection (text dense + multimodal dense + sparse)
        # Both lists are co-indexed by frame, so we zip them into same-size batches.
        caption_index_text_batches = [
            caption_embedding_results[i:i + analysis_batch_size]
            for i in range(0, len(caption_embedding_results), analysis_batch_size)
        ]
        caption_index_mm_batches = [
            caption_multimodal_embedding_results[i:i + analysis_batch_size]
            for i in range(0, len(caption_multimodal_embedding_results), analysis_batch_size)
        ]
        caption_index_futures = caption_qdrant_indexing_chunk_task.map(  # type: ignore
            caption_index_text_batches,
            caption_index_mm_batches,
            wait_for=[caption_embedding_futures, caption_multimodal_embedding_futures],
        )

        image_index_ids = [uid for batch in image_index_futures.result() for uid in batch]
        await ImageQdrantIndexingTask.summary_artifact(image_index_ids)
        if tracker:
            await tracker.complete_stage(video_id, ImageQdrantIndexingTask.__name__)

        caption_index_ids = [uid for batch in caption_index_futures.result() for uid in batch]
        await CaptionQdrantIndexingTask.summary_artifact(caption_index_ids)
        if tracker:
            await tracker.complete_stage(video_id, CaptionQdrantIndexingTask.__name__)

        logger.info(
            f"[Flow:{flow_run_name}] All tasks done | "
            f"{len(asr_results)} ASR | {len(image_results)} image | "
            f"{len(caption_results)} caption | {len(embedding_results)} embedding | "
            f"{len(ocr_results)} OCR | {len(caption_embedding_results)} cap-text-embed | "
            f"{len(caption_multimodal_embedding_results)} cap-mm-embed | "
            f"{len(image_index_ids)} image-qdrant | {len(caption_index_ids)} caption-qdrant"
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
                "caption_artifacts": len(caption_results),
                "embedding_artifacts": len(embedding_results),
                "ocr_artifacts": len(ocr_results),
                "caption_embedding_artifacts": len(caption_embedding_results),
                "caption_multimodal_embedding_artifacts": len(caption_multimodal_embedding_results),
                "image_qdrant_points": len(image_index_ids),
                "caption_qdrant_points": len(caption_index_ids),
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
                f"| Caption Artifacts | {len(caption_results)} |\n"
                f"| Image Embeddings | {len(embedding_results)} |\n"
                f"| OCR Artifacts | {len(ocr_results)} |\n"
                f"| Caption Text Embeddings | {len(caption_embedding_results)} |\n"
                f"| Caption Multimodal Embeddings | {len(caption_multimodal_embedding_results)} |\n"
                f"| Image Qdrant Points | {len(image_index_ids)} |\n"
                f"| Caption Qdrant Points | {len(caption_index_ids)} |\n"
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