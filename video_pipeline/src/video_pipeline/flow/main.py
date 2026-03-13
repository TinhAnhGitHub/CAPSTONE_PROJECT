import logging

from typing import cast, Any
from datetime import datetime

from prefect import flow, get_run_logger
from prefect.artifacts import acreate_markdown_artifact
from prefect.context import get_run_context
from prefect_dask import DaskTaskRunner  # type: ignore

from video_pipeline.task.video.main import (
    VideoInput,
    video_reg_task,
    VideoRegistryTask,
    VideoArtifact,
)
from video_pipeline.task.autoshot.main import autoshot_task, AutoshotTask, AutoshotArtifact
from video_pipeline.task.asr.main import asr_chunk_task, ASRTask
from video_pipeline.task.audio_segment.main import audio_segment_task, AudioSegmentTask
from video_pipeline.task.segment_embedding.main import (
    segment_embedding_chunk_task,
    SegmentEmbeddingTask,
)
from video_pipeline.task.segment_caption.main import segment_caption_chunk_task, SegmentCaptionTask
from video_pipeline.task.segment_caption_embedding.main import (
    segment_caption_embedding_chunk_task,
    SegmentCaptionEmbeddingTask,
)
from video_pipeline.task.segment_caption_multimodal_embedding.main import (
    segment_caption_multimodal_embedding_chunk_task,
    SegmentCaptionMultimodalEmbeddingTask,
)
from video_pipeline.task.image_extraction.main import image_chunk_task, ImageExtractionTask
from video_pipeline.task.image_caption.main import image_caption_chunk_task, ImageCaptionTask
from video_pipeline.task.image_embedding.main import image_embedding_chunk_task, ImageEmbeddingTask
from video_pipeline.task.image_caption_embedding.main import (
    image_caption_embedding_chunk_task,
    ImageCaptionEmbeddingTask,
)
from video_pipeline.task.image_caption_multimodal_embedding.main import (
    image_caption_multimodal_embedding_chunk_task,
    ImageCaptionMultimodalEmbeddingTask,
)
from video_pipeline.task.qdrant_indexing.main import (
    ImageQdrantIndexingTask,
    CaptionQdrantIndexingTask,
    image_qdrant_indexing_chunk_task,
    caption_qdrant_indexing_chunk_task,
    SegmentQdrantIndexingTask,
    SegmentCaptionQdrantIndexingTask,
    segment_qdrant_indexing_chunk_task,
    segment_caption_qdrant_indexing_chunk_task,
)
from video_pipeline.flow.subtask import preprocess_video_task
from video_pipeline.config import get_settings
from video_pipeline.core.client.progress import HTTPProgressTracker, StageRegistry, RunStatus
from video_pipeline.core.storage.prefect_block import create_minio_result_storage


def _make_batches(items: list, batch_size: int) -> list[list]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


@flow(
    name="Single Video Processing Flow",
    description="Process a single video through the complete pipeline",
    log_prints=True,
    persist_result=True,
    result_storage="s3-bucket/result-storage",
    retries=1,
    task_runner=DaskTaskRunner(  # type:ignore
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
        str(context.flow_run.id) if context and context.flow_run else "unknown"  # type:ignore
    )
    flow_run_name = (
        context.flow_run.name if context and context.flow_run else "unknown"  # type:ignore
    )

    video_input = VideoInput(user_id=user_id, video_id=video_id, video_s3_url=video_file_path)

    tracker: HTTPProgressTracker | None = None
    if tracker_url:
        tracker = HTTPProgressTracker(base_url=tracker_url)
        await tracker.start_video(video_id, StageRegistry.all_stage_names())

    logger.info(
        f"[Flow:{flow_run_name}] Starting pipeline | "
        f"video_id={video_id} user_id={user_id} path={video_file_path}"
    )

    try:
        # ── Stage 1: Video registration ───────────────────────────────────────
        logger.info(f"[Flow:{flow_run_name}] Submitting video_reg_task")
        video_fut = video_reg_task.submit(video_input)
        video_artifact = cast(VideoArtifact, video_fut.result())
        await VideoRegistryTask.summary_artifact(video_artifact)
        if tracker:
            await tracker.complete_stage(video_id, VideoRegistryTask.__name__)

        # ── Stage 2: Shot detection ───────────────────────────────────────────
        logger.info(f"[Flow:{flow_run_name}] Submitting autoshot_task")
        shots_fut = autoshot_task.submit(video_artifact)  # type:ignore
        shot_result = cast(AutoshotArtifact, shots_fut.result())
        await AutoshotTask.summary_artifact(shot_result)
        if tracker:
            await tracker.complete_stage(video_id, AutoshotTask.__name__)

        n_shots = len(shot_result.metadata.get("segments", []))  # type: ignore
        logger.info(f"[Flow:{flow_run_name}] autoshot_task done | {n_shots} shot(s) detected")

        # ── Stage 3: Preprocessing → asr_batches + image_batches ─────────────
        logger.info(f"[Flow:{flow_run_name}] Submitting preprocess_video_task")
        preprocess_fut = preprocess_video_task.submit(shot_result, wait_for=[shots_fut])  # type: ignore
        asr_batches, image_batches = preprocess_fut.result()  # type: ignore
        n_asr = sum(len(b) for b in asr_batches)
        n_img = sum(len(b) for b in image_batches)
        logger.info(
            f"[Flow:{flow_run_name}] preprocess done | "
            f"{n_asr} audio chunk(s) in {len(asr_batches)} ASR batch(es) | "
            f"{n_img} frame(s) in {len(image_batches)} image batch(es)"
        )

        # ── Stage 4: Fan out BOTH branches immediately ────────────────────────
        # Audio branch and image branch are independent — submit together so
        # Dask can schedule them concurrently on available workers.
        logger.info(
            f"[Flow:{flow_run_name}] Fanning out ASR ({len(asr_batches)} batch(es)) "
            f"and image extraction ({len(image_batches)} batch(es)) in parallel"
        )
        asr_batch_futures = asr_chunk_task.map(  # type:ignore
            asr_batches, wait_for=[preprocess_fut]
        )
        image_batch_futures = image_chunk_task.map(  # type:ignore
            image_batches, wait_for=[preprocess_fut]
        )

        # ── Stage 5a (audio): ASR → audio_segment ────────────────────────────
        asr_batch_results = asr_batch_futures.result()
        asr_results = [artifact for batch in asr_batch_results for artifact in batch]
        await ASRTask.summary_artifact(asr_results)
        if tracker:
            await tracker.complete_stage(video_id, ASRTask.__name__)
        logger.info(f"[Flow:{flow_run_name}] ASR done | {len(asr_results)} artifact(s)")

        logger.info(f"[Flow:{flow_run_name}] Submitting audio_segment_task")
        audio_segment_results = await audio_segment_task(asr_results)
        await AudioSegmentTask.summary_artifact(audio_segment_results)
        if tracker:
            await tracker.complete_stage(video_id, AudioSegmentTask.__name__)

        # ── Stage 6a: Segment embedding + caption (parallel with each other) ──
        segment_batch_size: int = 5
        segment_batches = _make_batches(audio_segment_results, segment_batch_size)

        logger.info(
            f"[Flow:{flow_run_name}] Fanning out "
            f"{len(segment_batches)} seg-embedding batch(es) and "
            f"{len(segment_batches)} seg-caption batch(es) in parallel"
        )
        segment_embedding_futures = segment_embedding_chunk_task.map(  # type: ignore
            segment_batches,
            wait_for=[audio_segment_task],
        )
        segment_caption_futures = segment_caption_chunk_task.map(  # type: ignore
            segment_batches,
            wait_for=[audio_segment_task],
        )

        # ── Stage 7a: seg_qdrant — depends only on seg_embedding ─────────────
        # Submit immediately when embeddings are ready; do NOT wait for captions.
        segment_embedding_results = [
            a for batch in segment_embedding_futures.result() for a in batch
        ]
        await SegmentEmbeddingTask.summary_artifact(segment_embedding_results)
        if tracker:
            await tracker.complete_stage(video_id, SegmentEmbeddingTask.__name__)

        segment_index_batches = _make_batches(segment_embedding_results, segment_batch_size)
        segment_index_futures = segment_qdrant_indexing_chunk_task.map(  # type: ignore
            segment_index_batches,
            wait_for=segment_embedding_futures,
        )
        logger.info(
            f"[Flow:{flow_run_name}] seg_qdrant_indexing submitted | "
            f"{len(segment_index_batches)} batch(es) — running concurrently with caption branch"
        )

        # ── Stage 7b: caption embedding + mm_embedding (parallel) ────────────
        segment_caption_results = [
            a for batch in segment_caption_futures.result() for a in batch
        ]
        await SegmentCaptionTask.summary_artifact(segment_caption_results)
        if tracker:
            await tracker.complete_stage(video_id, SegmentCaptionTask.__name__)

        segment_caption_batches = _make_batches(segment_caption_results, segment_batch_size)

        logger.info(
            f"[Flow:{flow_run_name}] Fanning out "
            f"{len(segment_caption_batches)} seg-cap-embed batch(es) and "
            f"{len(segment_caption_batches)} seg-cap-mm-embed batch(es) in parallel"
        )
        segment_caption_embedding_futures = segment_caption_embedding_chunk_task.map(  # type: ignore
            segment_caption_batches,
            wait_for=segment_caption_futures,
        )
        segment_caption_multimodal_embedding_futures = segment_caption_multimodal_embedding_chunk_task.map(  # type: ignore
            segment_caption_batches,
            wait_for=segment_caption_futures,
        )

        # ── Stage 8a (image): image_chunk results → caption + embedding ───────
        # Collect image results (these have been computing on Dask the whole time)
        image_batch_results = image_batch_futures.result()
        image_results = [artifact for batch in image_batch_results for artifact in batch]
        await ImageExtractionTask.summary_artifact(image_results)
        if tracker:
            await tracker.complete_stage(video_id, ImageExtractionTask.__name__)
        logger.info(
            f"[Flow:{flow_run_name}] Image extraction done | {len(image_results)} artifact(s)"
        )

        analysis_batch_size: int = 10
        analysis_batches = _make_batches(image_results, analysis_batch_size)

        # Image caption and embedding are independent of each other — submit together.
        logger.info(
            f"[Flow:{flow_run_name}] Fanning out image caption + embedding "
            f"({len(analysis_batches)} batch(es) each) in parallel"
        )
        caption_batch_futures = image_caption_chunk_task.map(  # type: ignore
            analysis_batches,
            wait_for=[image_batch_futures],
        )
        embedding_batch_futures = image_embedding_chunk_task.map(  # type: ignore
            analysis_batches,
            wait_for=[image_batch_futures],
        )

        # ── Stage 9a (image): image_qdrant — depends only on image_embedding ──
        # Submit as soon as image embeddings are ready; do NOT wait for captions.
        embedding_results = [a for batch in embedding_batch_futures.result() for a in batch]
        await ImageEmbeddingTask.summary_artifact(embedding_results)
        if tracker:
            await tracker.complete_stage(video_id, ImageEmbeddingTask.__name__)

        image_index_batches = _make_batches(embedding_results, analysis_batch_size)
        image_index_futures = image_qdrant_indexing_chunk_task.map(  # type: ignore
            image_index_batches,
            wait_for=embedding_batch_futures,
        )
        logger.info(
            f"[Flow:{flow_run_name}] image_qdrant_indexing submitted | "
            f"{len(image_index_batches)} batch(es) — running concurrently with caption embed"
        )

        # ── Stage 9b (image): caption embedding + mm_embedding ───────────────
        caption_results = [a for batch in caption_batch_futures.result() for a in batch]
        await ImageCaptionTask.summary_artifact(caption_results)
        if tracker:
            await tracker.complete_stage(video_id, ImageCaptionTask.__name__)

        caption_embedding_batch_size: int = 5
        caption_mm_batch_size: int = 2
        caption_embedding_batches = _make_batches(caption_results, caption_embedding_batch_size)
        caption_mm_batches = _make_batches(caption_results, caption_mm_batch_size)

        caption_embedding_futures = image_caption_embedding_chunk_task.map(  # type: ignore
            caption_embedding_batches,
            wait_for=[caption_batch_futures],
        )
        caption_multimodal_embedding_futures = image_caption_multimodal_embedding_chunk_task.map(  # type: ignore
            caption_mm_batches,
            wait_for=[caption_batch_futures],
        )

        # ── Stage 10: Collect all remaining results ───────────────────────────
        # Segment caption embeddings
        segment_caption_embedding_results = [
            a for batch in segment_caption_embedding_futures.result() for a in batch
        ]
        await SegmentCaptionEmbeddingTask.summary_artifact(segment_caption_embedding_results)
        if tracker:
            await tracker.complete_stage(video_id, SegmentCaptionEmbeddingTask.__name__)

        segment_caption_multimodal_embedding_results = [
            a for batch in segment_caption_multimodal_embedding_futures.result() for a in batch
        ]
        await SegmentCaptionMultimodalEmbeddingTask.summary_artifact(
            segment_caption_multimodal_embedding_results
        )
        if tracker:
            await tracker.complete_stage(video_id, SegmentCaptionMultimodalEmbeddingTask.__name__)

        # Image caption embeddings
        caption_embedding_results = [
            a for batch in caption_embedding_futures.result() for a in batch
        ]
        await ImageCaptionEmbeddingTask.summary_artifact(caption_embedding_results)
        if tracker:
            await tracker.complete_stage(video_id, ImageCaptionEmbeddingTask.__name__)

        caption_multimodal_embedding_results = [
            a for batch in caption_multimodal_embedding_futures.result() for a in batch
        ]
        await ImageCaptionMultimodalEmbeddingTask.summary_artifact(
            caption_multimodal_embedding_results
        )
        if tracker:
            await tracker.complete_stage(video_id, ImageCaptionMultimodalEmbeddingTask.__name__)

        logger.info(
            f"[Flow:{flow_run_name}] All embeddings done | "
            f"{len(segment_caption_embedding_results)} seg-cap-text-embed | "
            f"{len(segment_caption_multimodal_embedding_results)} seg-cap-mm-embed | "
            f"{len(caption_embedding_results)} cap-text-embed | "
            f"{len(caption_multimodal_embedding_results)} cap-mm-embed"
        )

        # ── Stage 11: Caption qdrant (depends on both caption embed types) ────
        caption_index_text_batches = _make_batches(caption_embedding_results, analysis_batch_size)
        caption_index_mm_batches = _make_batches(
            caption_multimodal_embedding_results, analysis_batch_size
        )
        caption_index_futures = caption_qdrant_indexing_chunk_task.map(  # type: ignore
            caption_index_text_batches,
            caption_index_mm_batches,
            wait_for=[caption_embedding_futures, caption_multimodal_embedding_futures],
        )

        # ── Stage 11: seg_caption_qdrant (depends on both seg cap embed types) ─
        segment_caption_index_text_batches = _make_batches(
            segment_caption_embedding_results, segment_batch_size
        )
        segment_caption_index_mm_batches = _make_batches(
            segment_caption_multimodal_embedding_results, segment_batch_size
        )
        segment_caption_index_futures = segment_caption_qdrant_indexing_chunk_task.map(  # type: ignore
            segment_caption_index_text_batches,
            segment_caption_index_mm_batches,
            wait_for=[
                segment_caption_embedding_futures,
                segment_caption_multimodal_embedding_futures,
            ],
        )

        # ── Stage 12: Collect all indexing results ────────────────────────────
        segment_index_ids = [uid for batch in segment_index_futures.result() for uid in batch]
        await SegmentQdrantIndexingTask.summary_artifact(segment_index_ids)
        if tracker:
            await tracker.complete_stage(video_id, SegmentQdrantIndexingTask.__name__)

        segment_caption_index_ids = [
            uid for batch in segment_caption_index_futures.result() for uid in batch
        ]
        await SegmentCaptionQdrantIndexingTask.summary_artifact(segment_caption_index_ids)
        if tracker:
            await tracker.complete_stage(video_id, SegmentCaptionQdrantIndexingTask.__name__)

        image_index_ids = [uid for batch in image_index_futures.result() for uid in batch]
        await ImageQdrantIndexingTask.summary_artifact(image_index_ids)
        if tracker:
            await tracker.complete_stage(video_id, ImageQdrantIndexingTask.__name__)

        caption_index_ids = [uid for batch in caption_index_futures.result() for uid in batch]
        await CaptionQdrantIndexingTask.summary_artifact(caption_index_ids)
        if tracker:
            await tracker.complete_stage(video_id, CaptionQdrantIndexingTask.__name__)

        completed_at = datetime.now().isoformat()
        logger.info(
            f"[Flow:{flow_run_name}] Pipeline complete | "
            f"shots={n_shots} | "
            f"asr={len(asr_results)} | audio_seg={len(audio_segment_results)} | "
            f"seg_embed={len(segment_embedding_results)} | seg_cap={len(segment_caption_results)} | "
            f"seg_cap_text_embed={len(segment_caption_embedding_results)} | "
            f"seg_cap_mm_embed={len(segment_caption_multimodal_embedding_results)} | "
            f"images={len(image_results)} | captions={len(caption_results)} | "
            f"img_embed={len(embedding_results)} | "
            f"cap_text_embed={len(caption_embedding_results)} | "
            f"cap_mm_embed={len(caption_multimodal_embedding_results)} | "
            f"seg_qdrant={len(segment_index_ids)} | seg_cap_qdrant={len(segment_caption_index_ids)} | "
            f"img_qdrant={len(image_index_ids)} | cap_qdrant={len(caption_index_ids)}"
        )

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
                f"| Audio Segments | {len(audio_segment_results)} |\n"
                f"| Segment Embeddings | {len(segment_embedding_results)} |\n"
                f"| Segment Captions | {len(segment_caption_results)} |\n"
                f"| Segment Caption Text Embeddings | {len(segment_caption_embedding_results)} |\n"
                f"| Segment Caption MM Embeddings | {len(segment_caption_multimodal_embedding_results)} |\n"
                f"| Image Artifacts | {len(image_results)} |\n"
                f"| Caption Artifacts | {len(caption_results)} |\n"
                f"| Image Embeddings | {len(embedding_results)} |\n"
                f"| Caption Text Embeddings | {len(caption_embedding_results)} |\n"
                f"| Caption Multimodal Embeddings | {len(caption_multimodal_embedding_results)} |\n"
                f"| Image Qdrant Points | {len(image_index_ids)} |\n"
                f"| Caption Qdrant Points | {len(caption_index_ids)} |\n"
                f"| Segment Qdrant Points | {len(segment_index_ids)} |\n"
                f"| Segment Caption Qdrant Points | {len(segment_caption_index_ids)} |\n"
            ),
        )

        if tracker:
            await tracker.complete_run(video_id)

        return {}

    except Exception as e:
        if tracker:
            await tracker.complete_run(video_id, RunStatus.FAILED, str(e))
        logger.exception(f"Pipeline failed for video {video_id}: {e}")
        raise