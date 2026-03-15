import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from typing import Any, Callable, ParamSpec, TypeVar, cast

from prefect import flow, get_run_logger
from prefect.artifacts import acreate_markdown_artifact
from prefect.context import get_run_context
from prefect_dask import DaskTaskRunner  # type: ignore

from video_pipeline.config import get_settings
from video_pipeline.core.client.progress import HTTPProgressTracker, RunStatus, StageRegistry
from video_pipeline.core.storage.prefect_block import create_minio_result_storage
from video_pipeline.flow.batch_helper import make_batches
from video_pipeline.flow.subtask import preprocess_video_task
from video_pipeline.task.audio_segment.main import audio_segment_task, AudioSegmentTask
from video_pipeline.task.asr.main import ASRTask, asr_chunk_task
from video_pipeline.task.autoshot.main import AutoshotTask, autoshot_task
from video_pipeline.task.image_caption.main import ImageCaptionTask, image_caption_chunk_task
from video_pipeline.task.image_caption_embedding.main import (
    ImageCaptionEmbeddingTask,
    image_caption_embedding_chunk_task,
)
from video_pipeline.task.image_caption_multimodal_embedding.main import (
    ImageCaptionMultimodalEmbeddingTask,
    image_caption_multimodal_embedding_chunk_task,
)
from video_pipeline.task.image_embedding.main import ImageEmbeddingTask, image_embedding_chunk_task
from video_pipeline.task.image_extraction.main import ImageExtractionTask, image_chunk_task
from video_pipeline.task.image_ocr.main import ImageOCRTask, image_ocr_chunk_task
from video_pipeline.task.ocr_indexing.main import OCRIndexingTask, ocr_indexing_chunk_task
from video_pipeline.task.qdrant_indexing import (
    CaptionQdrantIndexingTask,
    ImageQdrantIndexingTask,
    SegmentCaptionQdrantIndexingTask,
    SegmentQdrantIndexingTask,
    caption_qdrant_indexing_chunk_task,
    image_qdrant_indexing_chunk_task,
    segment_caption_qdrant_indexing_chunk_task,
    segment_qdrant_indexing_chunk_task,
)
from video_pipeline.task.segment_caption.main import SegmentCaptionTask, segment_caption_chunk_task
from video_pipeline.task.segment_caption_embedding.main import (
    SegmentCaptionEmbeddingTask,
    segment_caption_embedding_chunk_task,
)
from video_pipeline.task.segment_caption_multimodal_embedding.main import (
    SegmentCaptionMultimodalEmbeddingTask,
    segment_caption_multimodal_embedding_chunk_task,
)
from video_pipeline.task.segment_embedding.main import (
    SegmentEmbeddingTask,
    segment_embedding_chunk_task,
)
from video_pipeline.task.video.main import (
    VideoArtifact,
    VideoInput,
    VideoRegistryTask,
    video_reg_task,
)
from video_pipeline.task.kg_graph import KGPipelineTask, kg_pipeline_task

from video_pipeline.task.autoshot.main import AutoshotArtifact

# ─────────────────────────────────────────────────────────────────────────────
# Timing utilities
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class TimingRecord:
    """Records timing information for a single task/stage."""

    name: str
    start_time: float = 0.0
    end_time: float = 0.0
    duration_seconds: float = 0.0

    def start(self) -> "TimingRecord":
        self.start_time = time.perf_counter()
        return self

    def stop(self) -> "TimingRecord":
        self.end_time = time.perf_counter()
        self.duration_seconds = self.end_time - self.start_time
        return self
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()
    
@dataclass
class TimingRegistry:
    """Registry to collect all timing records."""

    records: list[TimingRecord] = field(default_factory=list)

    def record(self, name: str) -> TimingRecord:
        record = TimingRecord(name=name)
        self.records.append(record)
        return record

    def to_markdown_table(self) -> str:
        """Generate a markdown table of all timing records."""
        if not self.records:
            return "No timing records."

        lines = ["| Stage | Duration (s) |", "|-------|-------------|"]
        for r in self.records:
            lines.append(f"| {r.name} | {r.duration_seconds:.2f} |")

        total = sum(r.duration_seconds for r in self.records)
        lines.append(f"| **Total** | **{total:.2f}** |")
        return "\n".join(lines)

async def run_video_registration(
    video_input: VideoInput,
    tracker: HTTPProgressTracker | None,
    video_id: str,
    timing: TimingRegistry,
) -> VideoArtifact:
    """Stage 1: Register video in the system."""
    with timing.record("Video Registration") as t:
        t.start()
        video_fut = video_reg_task.submit(video_input)
        video_artifact = cast(VideoArtifact, video_fut.result())
        await VideoRegistryTask.summary_artifact(video_artifact)
        if tracker:
            await tracker.complete_stage(video_id, VideoRegistryTask.__name__)
        t.stop()
    return video_artifact


async def run_autoshot(
    video_artifact: VideoArtifact,
    tracker: HTTPProgressTracker | None,
    video_id: str,
    timing: TimingRegistry,
) -> tuple[AutoshotArtifact, Any]:
    """Stage 2: Detect shots in video."""
    with timing.record("Shot Detection") as t:
        t.start()
        shots_fut = autoshot_task.submit(video_artifact)  # type:ignore
        shot_result = cast(AutoshotArtifact, shots_fut.result())
        await AutoshotTask.summary_artifact(shot_result)
        if tracker:
            await tracker.complete_stage(video_id, AutoshotTask.__name__)
        t.stop()
    return shot_result, shots_fut


async def run_preprocess(
    shot_result: AutoshotArtifact,
    shots_fut: Any,
    timing: TimingRegistry,
) -> tuple[list, list]:
    """Stage 3: Preprocess video into ASR and image batches."""
    with timing.record("Video Preprocessing") as t:
        t.start()
        preprocess_fut = preprocess_video_task.submit(shot_result, wait_for=[shots_fut])  # type: ignore
        asr_batches, image_batches = preprocess_fut.result()  # type: ignore
        t.stop()
    return asr_batches, image_batches


async def run_asr(
    asr_batches: list,
    preprocess_fut: Any,
    tracker: HTTPProgressTracker | None,
    video_id: str,
    timing: TimingRegistry,
) -> tuple[list, Any]:
    """Stage 5a: Run ASR on audio batches."""
    with timing.record("ASR") as t:
        t.start()
        asr_batch_futures = asr_chunk_task.map(asr_batches, wait_for=[preprocess_fut])  # type:ignore
        asr_batch_results = asr_batch_futures.result()
        asr_results = [artifact for batch in asr_batch_results for artifact in batch]
        await ASRTask.summary_artifact(asr_results)
        if tracker:
            await tracker.complete_stage(video_id, ASRTask.__name__)
        t.stop()
    return asr_results, asr_batch_futures


async def run_audio_segment(
    asr_results: list,
    tracker: HTTPProgressTracker | None,
    video_id: str,
    timing: TimingRegistry,
) -> list:
    """Stage 5a continued: Segment audio from ASR results."""
    with timing.record("Audio Segmentation") as t:
        t.start()
        audio_segment_results = await audio_segment_task(asr_results)
        await AudioSegmentTask.summary_artifact(audio_segment_results)
        if tracker:
            await tracker.complete_stage(video_id, AudioSegmentTask.__name__)
        t.stop()
    return audio_segment_results


async def run_segment_embedding(
    segment_batches: list,
    audio_segment_results: list,
    tracker: HTTPProgressTracker | None,
    video_id: str,
    timing: TimingRegistry,
) -> tuple[list, Any]:
    """Stage 6a: Generate segment embeddings."""
    with timing.record("Segment Embedding") as t:
        t.start()
        segment_embedding_futures = segment_embedding_chunk_task.map(  # type: ignore
            segment_batches,
            wait_for=[audio_segment_task],
        )
        segment_embedding_results = [
            a for batch in segment_embedding_futures.result() for a in batch
        ]
        await SegmentEmbeddingTask.summary_artifact(segment_embedding_results)
        if tracker:
            await tracker.complete_stage(video_id, SegmentEmbeddingTask.__name__)
        t.stop()
    return segment_embedding_results, segment_embedding_futures


async def run_segment_caption(
    segment_batches: list,
    audio_segment_results: list,
    tracker: HTTPProgressTracker | None,
    video_id: str,
    timing: TimingRegistry,
) -> tuple[list, Any]:
    """Stage 6b: Generate segment captions."""
    with timing.record("Segment Caption") as t:
        t.start()
        segment_caption_futures = segment_caption_chunk_task.map(  # type: ignore
            segment_batches,
            wait_for=[audio_segment_task],
        )
        segment_caption_results = [a for batch in segment_caption_futures.result() for a in batch]
        await SegmentCaptionTask.summary_artifact(segment_caption_results)
        if tracker:
            await tracker.complete_stage(video_id, SegmentCaptionTask.__name__)
        t.stop()
    return segment_caption_results, segment_caption_futures


async def run_segment_qdrant_indexing(
    segment_embedding_results: list,
    segment_embedding_futures: Any,
    segment_batch_size: int,
    tracker: HTTPProgressTracker | None,
    video_id: str,
    timing: TimingRegistry,
) -> tuple[list, Any]:
    """Stage 7a: Index segment embeddings in Qdrant."""
    with timing.record("Segment Qdrant Indexing") as t:
        t.start()
        segment_index_batches = make_batches(segment_embedding_results, segment_batch_size)
        segment_index_futures = segment_qdrant_indexing_chunk_task.map(  # type: ignore
            segment_index_batches,
            wait_for=segment_embedding_futures,
        )
        segment_index_ids = [uid for batch in segment_index_futures.result() for uid in batch]
        await SegmentQdrantIndexingTask.summary_artifact(segment_index_ids)
        if tracker:
            await tracker.complete_stage(video_id, SegmentQdrantIndexingTask.__name__)
        t.stop()
    return segment_index_ids, segment_index_futures


async def run_segment_caption_embedding(
    segment_caption_batches: list,
    segment_caption_futures: Any,
    tracker: HTTPProgressTracker | None,
    video_id: str,
    timing: TimingRegistry,
) -> tuple[list, Any]:
    """Stage 7b: Generate segment caption text embeddings."""
    with timing.record("Segment Caption Embedding") as t:
        t.start()
        segment_caption_embedding_futures = segment_caption_embedding_chunk_task.map(  # type: ignore
            segment_caption_batches,
            wait_for=segment_caption_futures,
        )
        segment_caption_embedding_results = [
            a for batch in segment_caption_embedding_futures.result() for a in batch
        ]
        await SegmentCaptionEmbeddingTask.summary_artifact(segment_caption_embedding_results)
        if tracker:
            await tracker.complete_stage(video_id, SegmentCaptionEmbeddingTask.__name__)
        t.stop()
    return segment_caption_embedding_results, segment_caption_embedding_futures


async def run_segment_caption_multimodal_embedding(
    segment_caption_batches: list,
    segment_caption_futures: Any,
    tracker: HTTPProgressTracker | None,
    video_id: str,
    timing: TimingRegistry,
) -> tuple[list, Any]:
    """Stage 7b: Generate segment caption multimodal embeddings."""
    with timing.record("Segment Caption MM Embedding") as t:
        t.start()
        segment_caption_multimodal_embedding_futures = (
            segment_caption_multimodal_embedding_chunk_task.map(  # type: ignore
                segment_caption_batches,
                wait_for=segment_caption_futures,
            )
        )
        segment_caption_multimodal_embedding_results = [
            a for batch in segment_caption_multimodal_embedding_futures.result() for a in batch
        ]
        await SegmentCaptionMultimodalEmbeddingTask.summary_artifact(
            segment_caption_multimodal_embedding_results
        )
        if tracker:
            await tracker.complete_stage(video_id, SegmentCaptionMultimodalEmbeddingTask.__name__)
        t.stop()
    return (
        segment_caption_multimodal_embedding_results,
        segment_caption_multimodal_embedding_futures,
    )


async def run_image_extraction(
    image_batches: list,
    preprocess_fut: Any,
    tracker: HTTPProgressTracker | None,
    video_id: str,
    timing: TimingRegistry,
) -> tuple[list, Any]:
    """Stage 8a: Extract images from video."""
    with timing.record("Image Extraction") as t:
        t.start()
        image_batch_futures = image_chunk_task.map(image_batches, wait_for=[preprocess_fut])  # type:ignore
        image_batch_results = image_batch_futures.result()
        image_results = [artifact for batch in image_batch_results for artifact in batch]
        await ImageExtractionTask.summary_artifact(image_results)
        if tracker:
            await tracker.complete_stage(video_id, ImageExtractionTask.__name__)
        t.stop()
    return image_results, image_batch_futures


async def run_image_caption(
    analysis_batches: list,
    image_batch_futures: Any,
    tracker: HTTPProgressTracker | None,
    video_id: str,
    timing: TimingRegistry,
) -> tuple[list, Any]:
    """Stage 8b: Generate image captions."""
    with timing.record("Image Caption") as t:
        t.start()
        caption_batch_futures = image_caption_chunk_task.map(  # type: ignore
            analysis_batches,
            wait_for=[image_batch_futures],
        )
        caption_results = [a for batch in caption_batch_futures.result() for a in batch]
        await ImageCaptionTask.summary_artifact(caption_results)
        if tracker:
            await tracker.complete_stage(video_id, ImageCaptionTask.__name__)
        t.stop()
    return caption_results, caption_batch_futures


async def run_image_embedding(
    analysis_batches: list,
    image_batch_futures: Any,
    tracker: HTTPProgressTracker | None,
    video_id: str,
    timing: TimingRegistry,
) -> tuple[list, Any]:
    """Stage 8b: Generate image embeddings."""
    with timing.record("Image Embedding") as t:
        t.start()
        embedding_batch_futures = image_embedding_chunk_task.map(  # type: ignore
            analysis_batches,
            wait_for=[image_batch_futures],
        )
        embedding_results = [a for batch in embedding_batch_futures.result() for a in batch]
        await ImageEmbeddingTask.summary_artifact(embedding_results)
        if tracker:
            await tracker.complete_stage(video_id, ImageEmbeddingTask.__name__)
        t.stop()
    return embedding_results, embedding_batch_futures


async def run_image_qdrant_indexing(
    embedding_results: list,
    embedding_batch_futures: Any,
    analysis_batch_size: int,
    tracker: HTTPProgressTracker | None,
    video_id: str,
    timing: TimingRegistry,
) -> tuple[list, Any]:
    """Stage 9a: Index image embeddings in Qdrant."""
    with timing.record("Image Qdrant Indexing") as t:
        t.start()
        image_index_batches = make_batches(embedding_results, analysis_batch_size)
        image_index_futures = image_qdrant_indexing_chunk_task.map(  # type: ignore
            image_index_batches,
            wait_for=embedding_batch_futures,
        )
        image_index_ids = [uid for batch in image_index_futures.result() for uid in batch]
        await ImageQdrantIndexingTask.summary_artifact(image_index_ids)
        if tracker:
            await tracker.complete_stage(video_id, ImageQdrantIndexingTask.__name__)
        t.stop()
    return image_index_ids, image_index_futures


async def run_image_ocr(
    analysis_batches: list,
    image_batch_futures: Any,
    tracker: HTTPProgressTracker | None,
    video_id: str,
    timing: TimingRegistry,
) -> tuple[list, Any]:
    """Stage 9c: Run OCR on extracted images."""
    with timing.record("Image OCR") as t:
        t.start()
        ocr_batch_futures = image_ocr_chunk_task.map(  # type: ignore
            analysis_batches,
            wait_for=[image_batch_futures],
        )
        ocr_results = [a for batch in ocr_batch_futures.result() for a in batch]
        await ImageOCRTask.summary_artifact(ocr_results)
        if tracker:
            await tracker.complete_stage(video_id, ImageOCRTask.__name__)
        t.stop()
    return ocr_results, ocr_batch_futures


async def run_ocr_indexing(
    ocr_results: list,
    ocr_batch_futures: Any,
    ocr_batch_size: int,
    tracker: HTTPProgressTracker | None,
    video_id: str,
    timing: TimingRegistry,
) -> tuple[list, Any]:
    """Stage 10: Index OCR text into Elasticsearch."""
    with timing.record("OCR Indexing") as t:
        t.start()
        ocr_index_batches = make_batches(ocr_results, ocr_batch_size)
        ocr_index_futures = ocr_indexing_chunk_task.map(  # type: ignore
            ocr_index_batches,
            wait_for=ocr_batch_futures,
        )
        ocr_index_ids = [uid for batch in ocr_index_futures.result() for uid in batch]
        await OCRIndexingTask.summary_artifact(ocr_index_ids)
        if tracker:
            await tracker.complete_stage(video_id, OCRIndexingTask.__name__)
        t.stop()
    return ocr_index_ids, ocr_index_futures


async def run_image_caption_embedding(
    caption_embedding_batches: list,
    caption_batch_futures: Any,
    tracker: HTTPProgressTracker | None,
    video_id: str,
    timing: TimingRegistry,
) -> tuple[list, Any]:
    """Stage 9b: Generate image caption text embeddings."""
    with timing.record("Image Caption Embedding") as t:
        t.start()
        caption_embedding_futures = image_caption_embedding_chunk_task.map(  # type: ignore
            caption_embedding_batches,
            wait_for=[caption_batch_futures],
        )
        caption_embedding_results = [
            a for batch in caption_embedding_futures.result() for a in batch
        ]
        await ImageCaptionEmbeddingTask.summary_artifact(caption_embedding_results)
        if tracker:
            await tracker.complete_stage(video_id, ImageCaptionEmbeddingTask.__name__)
        t.stop()
    return caption_embedding_results, caption_embedding_futures


async def run_image_caption_multimodal_embedding(
    caption_mm_batches: list,
    caption_batch_futures: Any,
    tracker: HTTPProgressTracker | None,
    video_id: str,
    timing: TimingRegistry,
) -> tuple[list, Any]:
    """Stage 9b: Generate image caption multimodal embeddings."""
    with timing.record("Image Caption MM Embedding") as t:
        t.start()
        caption_multimodal_embedding_futures = image_caption_multimodal_embedding_chunk_task.map(  # type: ignore
            caption_mm_batches,
            wait_for=[caption_batch_futures],
        )
        caption_multimodal_embedding_results = [
            a for batch in caption_multimodal_embedding_futures.result() for a in batch
        ]
        await ImageCaptionMultimodalEmbeddingTask.summary_artifact(
            caption_multimodal_embedding_results
        )
        if tracker:
            await tracker.complete_stage(video_id, ImageCaptionMultimodalEmbeddingTask.__name__)
        t.stop()
    return caption_multimodal_embedding_results, caption_multimodal_embedding_futures


async def run_segment_caption_qdrant_indexing(
    segment_caption_embedding_results: list,
    segment_caption_multimodal_embedding_results: list,
    segment_caption_embedding_futures: Any,
    segment_caption_multimodal_embedding_futures: Any,
    segment_batch_size: int,
    tracker: HTTPProgressTracker | None,
    video_id: str,
    timing: TimingRegistry,
) -> list:
    """Stage 11: Index segment caption embeddings in Qdrant."""
    with timing.record("Segment Caption Qdrant Indexing") as t:
        t.start()
        segment_caption_index_text_batches = make_batches(
            segment_caption_embedding_results, segment_batch_size
        )
        segment_caption_index_mm_batches = make_batches(
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
        segment_caption_index_ids = [
            uid for batch in segment_caption_index_futures.result() for uid in batch
        ]
        await SegmentCaptionQdrantIndexingTask.summary_artifact(segment_caption_index_ids)
        if tracker:
            await tracker.complete_stage(video_id, SegmentCaptionQdrantIndexingTask.__name__)
        t.stop()
    return segment_caption_index_ids


async def run_caption_qdrant_indexing(
    caption_embedding_results: list,
    caption_multimodal_embedding_results: list,
    caption_embedding_futures: Any,
    caption_multimodal_embedding_futures: Any,
    analysis_batch_size: int,
    tracker: HTTPProgressTracker | None,
    video_id: str,
    timing: TimingRegistry,
) -> list:
    """Stage 11: Index image caption embeddings in Qdrant."""
    with timing.record("Caption Qdrant Indexing") as t:
        t.start()
        caption_index_text_batches = make_batches(caption_embedding_results, analysis_batch_size)
        caption_index_mm_batches = make_batches(
            caption_multimodal_embedding_results, analysis_batch_size
        )
        caption_index_futures = caption_qdrant_indexing_chunk_task.map(  # type: ignore
            caption_index_text_batches,
            caption_index_mm_batches,
            wait_for=[caption_embedding_futures, caption_multimodal_embedding_futures],
        )
        caption_index_ids = [uid for batch in caption_index_futures.result() for uid in batch]
        await CaptionQdrantIndexingTask.summary_artifact(caption_index_ids)
        if tracker:
            await tracker.complete_stage(video_id, CaptionQdrantIndexingTask.__name__)
        t.stop()
    return caption_index_ids


async def run_kg_pipeline(
    segment_caption_results: list,
    segment_caption_futures: Any,
    tracker: HTTPProgressTracker | None,
    video_id: str,
    timing: TimingRegistry,
) -> tuple[list, Any]:
    """Stage 12: Run Knowledge Graph pipeline on segment captions."""
    with timing.record("Knowledge Graph Pipeline") as t:
        t.start()
        kg_futures = kg_pipeline_task.submit(  # type: ignore
            segment_caption_results,
            wait_for=segment_caption_futures,
        )
        kg_results = kg_futures.result()
        await KGPipelineTask.summary_artifact(kg_results)
        if tracker:
            await tracker.complete_stage(video_id, KGPipelineTask.__name__)
        t.stop()
    return kg_results, kg_futures


async def run_audio_branch(
    asr_batches: list,
    preprocess_fut: Any,
    tracker: HTTPProgressTracker | None,
    video_id: str,
    timing: TimingRegistry,
) -> dict[str, Any]:
    """
    Run the complete audio/segment branch:
    ASR → AudioSegment → SegmentEmbedding/Caption → Embeddings → Qdrant indexing
    """
    asr_results, asr_batch_futures = await run_asr(
        asr_batches, preprocess_fut, tracker, video_id, timing
    )

    audio_segment_results = await run_audio_segment(asr_results, tracker, video_id, timing)

    segment_batch_size: int = 5
    segment_batches = make_batches(audio_segment_results, segment_batch_size)

    segment_embedding_results, segment_embedding_futures = await run_segment_embedding(
        segment_batches, audio_segment_results, tracker, video_id, timing
    )
    segment_caption_results, segment_caption_futures = await run_segment_caption(
        segment_batches, audio_segment_results, tracker, video_id, timing
    )

    segment_index_ids, segment_index_futures = await run_segment_qdrant_indexing(
        segment_embedding_results,
        segment_embedding_futures,
        segment_batch_size,
        tracker,
        video_id,
        timing,
    )

    segment_caption_batches = make_batches(segment_caption_results, segment_batch_size)

    (
        segment_caption_embedding_results,
        segment_caption_embedding_futures,
    ) = await run_segment_caption_embedding(
        segment_caption_batches, segment_caption_futures, tracker, video_id, timing
    )
    (
        segment_caption_multimodal_embedding_results,
        segment_caption_multimodal_embedding_futures,
    ) = await run_segment_caption_multimodal_embedding(
        segment_caption_batches, segment_caption_futures, tracker, video_id, timing
    )

    kg_results, kg_futures = await run_kg_pipeline(
        segment_caption_results,
        segment_caption_futures,
        tracker,
        video_id,
        timing,
    )

    return {
        "asr_results": asr_results,
        "audio_segment_results": audio_segment_results,
        "segment_embedding_results": segment_embedding_results,
        "segment_caption_results": segment_caption_results,
        "segment_caption_embedding_results": segment_caption_embedding_results,
        "segment_caption_multimodal_embedding_results": segment_caption_multimodal_embedding_results,
        "segment_index_ids": segment_index_ids,
        "segment_caption_embedding_futures": segment_caption_embedding_futures,
        "segment_caption_multimodal_embedding_futures": segment_caption_multimodal_embedding_futures,
        "segment_batch_size": segment_batch_size,
        "kg_results": kg_results,
    }


async def run_image_branch(
    image_batches: list,
    preprocess_fut: Any,
    tracker: HTTPProgressTracker | None,
    video_id: str,
    timing: TimingRegistry,
) -> dict[str, Any]:
    """
    Run the complete image branch:
    ImageExtraction → ImageCaption/ImageEmbedding/ImageOCR → CaptionEmbeddings → Qdrant/Elasticsearch indexing
    """
    # Image Extraction
    image_results, image_batch_futures = await run_image_extraction(
        image_batches, preprocess_fut, tracker, video_id, timing
    )

    # Analysis batches for downstream tasks
    analysis_batch_size: int = 10
    analysis_batches = make_batches(image_results, analysis_batch_size)

    # Image Caption, Embedding, and OCR (parallel)
    caption_results, caption_batch_futures = await run_image_caption(
        analysis_batches, image_batch_futures, tracker, video_id, timing
    )
    embedding_results, embedding_batch_futures = await run_image_embedding(
        analysis_batches, image_batch_futures, tracker, video_id, timing
    )
    ocr_results, ocr_batch_futures = await run_image_ocr(
        analysis_batches, image_batch_futures, tracker, video_id, timing
    )

    # Image Qdrant Indexing
    image_index_ids, image_index_futures = await run_image_qdrant_indexing(
        embedding_results, embedding_batch_futures, analysis_batch_size, tracker, video_id, timing
    )

    # OCR Indexing into Elasticsearch
    ocr_batch_size: int = 20
    ocr_index_ids, ocr_index_futures = await run_ocr_indexing(
        ocr_results, ocr_batch_futures, ocr_batch_size, tracker, video_id, timing
    )

    # Caption batches for embedding
    caption_embedding_batch_size: int = 5
    caption_mm_batch_size: int = 2
    caption_embedding_batches = make_batches(caption_results, caption_embedding_batch_size)
    caption_mm_batches = make_batches(caption_results, caption_mm_batch_size)

    # Caption Embeddings
    caption_embedding_results, caption_embedding_futures = await run_image_caption_embedding(
        caption_embedding_batches, caption_batch_futures, tracker, video_id, timing
    )
    (
        caption_multimodal_embedding_results,
        caption_multimodal_embedding_futures,
    ) = await run_image_caption_multimodal_embedding(
        caption_mm_batches, caption_batch_futures, tracker, video_id, timing
    )

    return {
        "image_results": image_results,
        "caption_results": caption_results,
        "embedding_results": embedding_results,
        "ocr_results": ocr_results,
        "caption_embedding_results": caption_embedding_results,
        "caption_multimodal_embedding_results": caption_multimodal_embedding_results,
        "image_index_ids": image_index_ids,
        "ocr_index_ids": ocr_index_ids,
        "caption_embedding_futures": caption_embedding_futures,
        "caption_multimodal_embedding_futures": caption_multimodal_embedding_futures,
        "analysis_batch_size": analysis_batch_size,
    }


async def run_final_qdrant_indexing(
    audio_branch: dict[str, Any],
    image_branch: dict[str, Any],
    tracker: HTTPProgressTracker | None,
    video_id: str,
    timing: TimingRegistry,
) -> dict[str, Any]:
    """Run final Qdrant indexing for captions."""
    segment_caption_index_ids = await run_segment_caption_qdrant_indexing(
        audio_branch["segment_caption_embedding_results"],
        audio_branch["segment_caption_multimodal_embedding_results"],
        audio_branch["segment_caption_embedding_futures"],
        audio_branch["segment_caption_multimodal_embedding_futures"],
        audio_branch["segment_batch_size"],
        tracker,
        video_id,
        timing,
    )

    caption_index_ids = await run_caption_qdrant_indexing(
        image_branch["caption_embedding_results"],
        image_branch["caption_multimodal_embedding_results"],
        image_branch["caption_embedding_futures"],
        image_branch["caption_multimodal_embedding_futures"],
        image_branch["analysis_batch_size"],
        tracker,
        video_id,
        timing,
    )

    return {
        "segment_caption_index_ids": segment_caption_index_ids,
        "caption_index_ids": caption_index_ids,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Summary artifact creation
# ─────────────────────────────────────────────────────────────────────────────


async def create_summary_artifact(
    video_id: str,
    flow_run_id: str,
    n_shots: int,
    audio_branch: dict[str, Any],
    image_branch: dict[str, Any],
    final_indexing: dict[str, Any],
    timing: TimingRegistry,
) -> None:
    """Create final summary markdown artifact with timing information."""
    completed_at = datetime.now().isoformat()

    timing_table = timing.to_markdown_table()

    # KG stats
    kg_artifact = audio_branch.get('kg_results')
    kg_entities = kg_artifact.total_canonical_entities if kg_artifact else 0
    kg_raw_entities = kg_artifact.total_raw_entities if kg_artifact else 0
    kg_events = kg_artifact.total_events if kg_artifact else 0
    kg_micro_events = kg_artifact.total_micro_events if kg_artifact else 0
    kg_communities = kg_artifact.total_communities if kg_artifact else 0
    kg_relationships = kg_artifact.total_relationships if kg_artifact else 0
    kg_event_edges = kg_artifact.total_event_edges if kg_artifact else 0
    kg_micro_edges = kg_artifact.total_micro_event_edges if kg_artifact else 0
    kg_nodes_embed = kg_artifact.total_nodes_with_embeddings if kg_artifact else 0
    kg_modularity = kg_artifact.graph_modularity if kg_artifact else 0.0

    # Cost tracking
    kg_prompt_tokens = kg_artifact.total_prompt_tokens if kg_artifact else 0
    kg_completion_tokens = kg_artifact.total_completion_tokens if kg_artifact else 0
    kg_cost = kg_artifact.total_llm_cost if kg_artifact else 0.0
    kg_llm_calls = kg_artifact.llm_calls if kg_artifact else 0
    kg_model = kg_artifact.llm_model if kg_artifact else ""

    # Format cost display
    cost_display = f"${kg_cost:.4f}" if kg_cost > 0 else "N/A"
    tokens_display = f"{kg_prompt_tokens:,}" if kg_prompt_tokens > 0 else "N/A"

    kg_section = ""
    if kg_artifact:
        kg_section = f"""
## Knowledge Graph Pipeline

### Entity Statistics
| Field | Count |
|-------|-------|
| Raw Entities (extracted) | `{kg_raw_entities}` |
| Canonical Entities (resolved) | `{kg_entities}` |
| Entity Resolution Ratio | `{kg_raw_entities / kg_entities:.2f}x` if kg_entities > 0 else 'N/A' |
| Global Relationships | `{kg_relationships}` |

### Event Layer
| Field | Count |
|-------|-------|
| Big Events | `{kg_events}` |
| Micro-Events | `{kg_micro_events}` |
| Event-to-Event Edges | `{kg_event_edges}` |
| Micro-Event Edges | `{kg_micro_edges}` |

### Community Structure
| Field | Value |
|-------|-------|
| Communities Detected | `{kg_communities}` |
| Graph Modularity | `{kg_modularity:.4f}` |
| Nodes with Node2Vec Embeddings | `{kg_nodes_embed}` |

### LLM Cost & Usage
| Field | Value |
|-------|-------|
| LLM Model | `{kg_model or 'N/A'}` |
| LLM Calls | `{kg_llm_calls}` |
| Prompt Tokens | `{tokens_display}` |
| Completion Tokens | `{kg_completion_tokens:,}` if kg_completion_tokens > 0 else 'N/A' |
| Estimated Cost | `{cost_display}` |

"""

    await acreate_markdown_artifact(
        description=f"Processing summary for video {video_id}",
        markdown=(
            f"# Video Processing Complete\n\n"
            f"**Video ID:** `{video_id}`\n"
            f"**Flow Run:** `{flow_run_id}`\n"
            f"**Completed:** {completed_at}\n\n"
            f"## Artifact Summary\n\n"
            f"| Artifact | Count |\n"
            f"|----------|-------|\n"
            f"| Shots | {n_shots} |\n"
            f"| ASR Artifacts | {len(audio_branch['asr_results'])} |\n"
            f"| Audio Segments | {len(audio_branch['audio_segment_results'])} |\n"
            f"| Segment Embeddings | {len(audio_branch['segment_embedding_results'])} |\n"
            f"| Segment Captions | {len(audio_branch['segment_caption_results'])} |\n"
            f"| Segment Caption Text Embeddings | {len(audio_branch['segment_caption_embedding_results'])} |\n"
            f"| Segment Caption MM Embeddings | {len(audio_branch['segment_caption_multimodal_embedding_results'])} |\n"
            f"| Image Artifacts | {len(image_branch['image_results'])} |\n"
            f"| Caption Artifacts | {len(image_branch['caption_results'])} |\n"
            f"| Image Embeddings | {len(image_branch['embedding_results'])} |\n"
            f"| OCR Artifacts | {len(image_branch['ocr_results'])} |\n"
            f"| OCR Elasticsearch Documents | {len(image_branch['ocr_index_ids'])} |\n"
            f"| Caption Text Embeddings | {len(image_branch['caption_embedding_results'])} |\n"
            f"| Caption Multimodal Embeddings | {len(image_branch['caption_multimodal_embedding_results'])} |\n"
            f"| Image Qdrant Points | {len(image_branch['image_index_ids'])} |\n"
            f"| Caption Qdrant Points | {len(final_indexing['caption_index_ids'])} |\n"
            f"| Segment Qdrant Points | {len(audio_branch['segment_index_ids'])} |\n"
            f"| Segment Caption Qdrant Points | {len(final_indexing['segment_caption_index_ids'])} |\n\n"
            f"{kg_section}"
            f"## Timing Breakdown\n\n"
            f"{timing_table}\n"
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main flow
# ─────────────────────────────────────────────────────────────────────────────


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

    timing = TimingRegistry()

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
        video_artifact = await run_video_registration(video_input, tracker, video_id, timing)

        # ── Stage 2: Shot detection ───────────────────────────────────────────
        shot_result, shots_fut = await run_autoshot(video_artifact, tracker, video_id, timing)
        n_shots = len(shot_result.metadata.get("segments", []))  # type: ignore
        logger.info(f"[Flow:{flow_run_name}] Shot detection done | {n_shots} shot(s) detected")

        # ── Stage 3: Preprocessing ────────────────────────────────────────────
        asr_batches, image_batches = await run_preprocess(shot_result, shots_fut, timing)
        n_asr = sum(len(b) for b in asr_batches)
        n_img = sum(len(b) for b in image_batches)
        logger.info(
            f"[Flow:{flow_run_name}] Preprocess done | "
            f"{n_asr} audio chunk(s) in {len(asr_batches)} ASR batch(es) | "
            f"{n_img} frame(s) in {len(image_batches)} image batch(es)"
        )

        # ── Stage 4: Fan out both branches ────────────────────────────────────
        logger.info(
            f"[Flow:{flow_run_name}] Fanning out ASR ({len(asr_batches)} batch(es)) "
            f"and image extraction ({len(image_batches)} batch(es)) in parallel"
        )

        # Submit both branches for parallel execution
        preprocess_fut = preprocess_video_task.submit(shot_result, wait_for=[shots_fut])  # type: ignore

        # Run audio and image branches
        audio_branch = await run_audio_branch(
            asr_batches, preprocess_fut, tracker, video_id, timing
        )
        image_branch = await run_image_branch(
            image_batches, preprocess_fut, tracker, video_id, timing
        )

        # ── Stage 11: Final Qdrant indexing ───────────────────────────────────
        final_indexing = await run_final_qdrant_indexing(
            audio_branch, image_branch, tracker, video_id, timing
        )

        # ── Create summary artifact ───────────────────────────────────────────
        await create_summary_artifact(
            video_id, flow_run_id, n_shots, audio_branch, image_branch, final_indexing, timing
        )

        logger.info(
            f"[Flow:{flow_run_name}] Pipeline complete | "
            f"shots={n_shots} | "
            f"asr={len(audio_branch['asr_results'])} | audio_seg={len(audio_branch['audio_segment_results'])} | "
            f"seg_embed={len(audio_branch['segment_embedding_results'])} | seg_cap={len(audio_branch['segment_caption_results'])} | "
            f"seg_cap_text_embed={len(audio_branch['segment_caption_embedding_results'])} | "
            f"seg_cap_mm_embed={len(audio_branch['segment_caption_multimodal_embedding_results'])} | "
            f"images={len(image_branch['image_results'])} | captions={len(image_branch['caption_results'])} | "
            f"img_embed={len(image_branch['embedding_results'])} | "
            f"ocr={len(image_branch['ocr_results'])} | ocr_index={len(image_branch['ocr_index_ids'])} | "
            f"cap_text_embed={len(image_branch['caption_embedding_results'])} | "
            f"cap_mm_embed={len(image_branch['caption_multimodal_embedding_results'])} | "
            f"seg_qdrant={len(audio_branch['segment_index_ids'])} | seg_cap_qdrant={len(final_indexing['segment_caption_index_ids'])} | "
            f"img_qdrant={len(image_branch['image_index_ids'])} | cap_qdrant={len(final_indexing['caption_index_ids'])}"
        )

        if tracker:
            await tracker.complete_run(video_id)

        return {}

    except Exception as e:
        if tracker:
            await tracker.complete_run(video_id, RunStatus.FAILED, str(e))
        logger.exception(f"Pipeline failed for video {video_id}: {e}")
        raise
