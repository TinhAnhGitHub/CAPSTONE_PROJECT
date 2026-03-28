from prefect import get_run_logger, task

from video_pipeline.config import get_settings
from video_pipeline.core.artifact import AutoshotArtifact
from video_pipeline.core.client.storage.minio import MinioStorageClient
from video_pipeline.task.asr.helper import extract_single_audio_segment
from video_pipeline.task.asr.main import ASRItem
from video_pipeline.task.base.base_task import TaskConfig
from video_pipeline.task.image_extraction.helper import get_segment_frame_indices
from video_pipeline.task.image_extraction.main import ImageItem


PREPROCESS_CONFIG = TaskConfig.from_yaml("video_preprocess")
_base_kwargs = PREPROCESS_CONFIG.to_task_kwargs()


@task(**_base_kwargs)  # type: ignore
async def preprocess_video_task(
    autoshot_artifact: AutoshotArtifact,
    asr_batch_size: int = 10,
    image_batch_size: int = 35,
    frames_per_segment: int = 5,
) -> tuple[list[list[ASRItem]], list[list[ImageItem]]]:
    """Download the video once and prepare both ASR and image extraction inputs.

    For each shot segment:
    - Extracts one WAV audio chunk  → ASRItem
    - Computes N representative frame indices → N ImageItems

    Returns batches for independent fan-out:
    - asr_batches:   list[list[ASRItem]]   — each batch goes to asr_chunk_task
    - image_batches: list[list[ImageItem]] — each batch goes to image_chunk_task

    Audio temp files are owned by asr_chunk_task and deleted after inference.
    Frame bytes are extracted lazily inside image_chunk_task.preprocess().
    """
    logger = get_run_logger()
    settings = get_settings()

    segments: list[list[int]] = autoshot_artifact.metadata["segments"]  # type: ignore
    fps = autoshot_artifact.related_video_fps
    video_url = autoshot_artifact.related_video_minio_url
    video_extension = autoshot_artifact.related_video_extension

    logger.info(
        f"[VideoPreprocess] Starting | "
        f"autoshot_id={autoshot_artifact.artifact_id} "
        f"video_url={video_url} segments={len(segments)}"
    )

    minio_client = MinioStorageClient(
        endpoint=settings.minio.endpoint,
        access_key=settings.minio.access_key,
        secret_key=settings.minio.secret_key,
        secure=settings.minio.secure,
    )
    logger.info(f"[VideoPreprocess] MinIO client initialized at {settings.minio.endpoint}")

    asr_items: list[ASRItem] = []
    image_items: list[ImageItem] = []

    logger.info(
        f"[VideoPreprocess] Downloading video and extracting "
        f"{len(segments)} audio chunk(s) + "
        f"{len(segments) * frames_per_segment} frame index(es)..."
    )

    async with minio_client.fetch_object_from_s3(
        s3_url=video_url,
        suffix=f".{video_extension}",
    ) as video_path:
        for i, (start_frame, end_frame) in enumerate(segments):
            audio_path = extract_single_audio_segment(str(video_path), start_frame, end_frame, fps)
            asr_items.append((autoshot_artifact, start_frame, end_frame, audio_path))

            for frame_index in get_segment_frame_indices(
                start_frame, end_frame, frames_per_segment
            ):
                image_items.append((autoshot_artifact, frame_index))

            logger.debug(
                f"[VideoPreprocess] Segment {i + 1}/{len(segments)} done | "
                f"frames=[{start_frame}, {end_frame}] audio={audio_path}"
            )

    asr_batches = [
        asr_items[i : i + asr_batch_size] for i in range(0, len(asr_items), asr_batch_size)
    ]
    image_batches = [
        image_items[i : i + image_batch_size] for i in range(0, len(image_items), image_batch_size)
    ]

    logger.info(
        f"[VideoPreprocess] Done | "
        f"{len(asr_items)} audio chunk(s) → {len(asr_batches)} ASR batch(es) | "
        f"{len(image_items)} frame index(es) → {len(image_batches)} image batch(es)"
    )

    return asr_batches, image_batches
