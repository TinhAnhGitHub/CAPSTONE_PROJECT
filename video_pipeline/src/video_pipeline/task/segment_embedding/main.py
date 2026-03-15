from __future__ import annotations

import io
from PIL import Image as PILImage
import numpy as np

from prefect import get_run_logger, task
from prefect.artifacts import acreate_markdown_artifact

from video_pipeline.task.base.base_task import TaskConfig, BaseTask
from video_pipeline.core.client.progress import StageRegistry
from video_pipeline.core.artifact import AudioSegmentArtifact, SegmentEmbeddingArtifact
from video_pipeline.core.storage.pg_tracker import ArtifactPersistentVisitor
from video_pipeline.core.client.storage.minio import MinioStorageClient
from video_pipeline.core.client.storage.pg.runtime import get_postgres_client, shutdown_postgres_client
from video_pipeline.core.client.inference.qwenvl_embed import (
    QwenVLEmbeddingClient,
    QwenVLEmbeddingConfig,
)
from video_pipeline.config import get_settings
from video_pipeline.task.image_extraction.helper import FastFrameReader, get_segment_frame_indices


SEGMENT_EMBEDDING_CONFIG = TaskConfig.from_yaml("segment_embedding")


@StageRegistry.register
class SegmentEmbeddingTask(BaseTask[list[AudioSegmentArtifact], list[SegmentEmbeddingArtifact]]):
    """Embed audio segments using QwenVL by extracting 6 frames per segment."""

    config = SEGMENT_EMBEDDING_CONFIG

    async def preprocess(
        self,
        input_data: list[AudioSegmentArtifact],
    ) -> list[tuple[AudioSegmentArtifact, bytes]]:
        """Load video bytes once for all segments."""
        logger = get_run_logger()
        logger.info(f"[SegmentEmbeddingTask] Loading video for {len(input_data)} segment(s)")

        if not input_data:
            return []

        related_video_minio_url = input_data[0].related_video_minio_url

        async with self.minio_client.fetch_object_from_s3(
            s3_url=related_video_minio_url,
            suffix='.mp4'
        ) as local_video_path:
            with open(local_video_path, 'rb') as f:
                video_bytes = f.read()
    
        preprocessed: list[tuple[AudioSegmentArtifact, bytes]] = [
            (seg, video_bytes) for seg in input_data
        ]

        logger.info(f"[SegmentEmbeddingTask] Video loaded ({len(video_bytes)} bytes)")
        return preprocessed

    async def execute(
        self,
        preprocessed: list[tuple[AudioSegmentArtifact, bytes]],
        client: QwenVLEmbeddingClient,
    ) -> list[SegmentEmbeddingArtifact]:
        """Extract 6 frames from each segment and embed them."""
        logger = get_run_logger()
        n_frames = self.config.additional_kwargs.get("n_frames", 6)

        logger.info(
            f"[SegmentEmbeddingTask] Embedding {len(preprocessed)} segment(s) with {n_frames} frames each"
        )

        artifacts = []
        for seg, video_bytes in preprocessed:
            reader = FastFrameReader(video_bytes)
            fps = reader.fps

            start_frame = int(seg.start_sec * fps)
            end_frame = int(seg.end_sec * fps)

            frame_indices = get_segment_frame_indices(start_frame, end_frame, n_frames)

            image_bytes_list = []
            for fi in frame_indices:
                try:
                    frame_bytes = reader.get_frame(fi)
                    image_bytes_list.append(frame_bytes)
                except Exception as e:
                    logger.warning(f"Failed to extract frame {fi}: {e}")

            if not image_bytes_list:
                logger.warning(f"No frames extracted for segment {seg.segment_index}, skipping")
                continue

            def _to_jpeg(data: bytes, size: int = 640) -> bytes:
                buf = io.BytesIO()
                img = PILImage.open(io.BytesIO(data)).convert("RGB")
                img = img.resize((size, size), PILImage.Resampling.LANCZOS)
                img.save(buf, format="JPEG", quality=90)
                return buf.getvalue()

            jpeg_list = [_to_jpeg(img) for img in image_bytes_list]
            embedding = await client.ainfer_video(jpeg_list)

            if embedding:
                artifact = SegmentEmbeddingArtifact(
                    related_audio_segment_artifact_id=seg.artifact_id,
                    related_video_id=seg.related_video_id,
                    related_video_minio_url=seg.related_video_minio_url,
                    related_video_extension=seg.related_video_extension,
                    related_video_fps=seg.related_video_fps,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    start_timestamp=seg.start_timestamp,
                    end_timestamp=seg.end_timestamp,
                    start_sec=seg.start_sec,
                    end_sec=seg.end_sec,
                    frame_indices=frame_indices,
                    embedding_dim=len(embedding),
                    user_id=seg.user_id,
                    caption_text=seg.audio_text,
                    object_name=(
                        f"embedding/segment/{seg.related_video_id}/"
                        f"{start_frame}_{end_frame}_{seg.start_timestamp}_{seg.end_timestamp}.npy"
                    ),
                    metadata={
                        "embedding": embedding,
                        "n_frames_extracted": len(frame_indices),
                    },
                )
                artifacts.append(artifact)

        logger.info(f"[SegmentEmbeddingTask] Created {len(artifacts)} embedding artifact(s)")
        return artifacts

    async def postprocess(
        self, result: list[SegmentEmbeddingArtifact]
    ) -> list[SegmentEmbeddingArtifact]:
        """Persist segment embeddings to database."""
        logger = get_run_logger()
        logger.info(f"[SegmentEmbeddingTask] Uploading {len(result)} embedding(s) to MinIO")

        for res in result:
            embedding = (res.metadata or {}).get("embedding", [])
            if embedding:
                npy_bytes = io.BytesIO()
                np.save(npy_bytes, np.array(embedding))
                npy_bytes.seek(0)

                await self.artifact_visitor.visit_artifact(res, upload_to_minio=npy_bytes)
            else:
                await self.artifact_visitor.visit_artifact(res)

        return result

    @staticmethod
    async def summary_artifact(final_result: list[SegmentEmbeddingArtifact]) -> None:
        """Create a Prefect artifact summarizing segment embeddings."""
        if not final_result:
            return

        first = final_result[0]

        segment_rows = ""
        for i, seg in enumerate(final_result):
            n_frames = len(seg.frame_indices)
            segment_rows += (
                f"| {i + 1} | {seg.start_timestamp} | {seg.end_timestamp} | "
                f"{n_frames} | {seg.embedding_dim} |\n"
            )

        markdown = (
            f"# Segment Embedding Summary\n\n"
            f"| Field | Value |\n"
            f"|-------|-------|\n"
            f"| **Video ID** | `{first.related_video_id}` |\n"
            f"| **User ID** | `{first.user_id}` |\n"
            f"| **Segments Embedded** | `{len(final_result)}` |\n\n"
            f"## Segment Embeddings\n\n"
            f"| # | Start | End | Frames | Embedding Dim |\n"
            f"|---|-------|-----|--------|---------------|\n"
            f"{segment_rows}"
        )

        await acreate_markdown_artifact(
            key=f"segment-embedding-{first.related_video_id}".lower(),
            markdown=markdown,
            description=f"Segment embedding summary for video {first.related_video_id}",
        )


@task(**{**SEGMENT_EMBEDDING_CONFIG.to_task_kwargs(), "name": "Segment Embedding"})  # type: ignore
async def segment_embedding_chunk_task(
    segments: list[AudioSegmentArtifact],
) -> list[SegmentEmbeddingArtifact]:
    """Process audio segments into embeddings.

    Args:
        segments: List of AudioSegmentArtifact

    Returns:
        List of SegmentEmbeddingArtifact
    """
    logger = get_run_logger()
    settings = get_settings()

    logger.info(f"[SegmentEmbedding] Starting | {len(segments)} segment(s)")

    minio_client = MinioStorageClient(
        endpoint=settings.minio.endpoint,
        access_key=settings.minio.access_key,
        secret_key=settings.minio.secret_key,
        secure=settings.minio.secure,
    )
    postgres_client = await get_postgres_client()

    embed_config = QwenVLEmbeddingConfig(
        base_url=SEGMENT_EMBEDDING_CONFIG.additional_kwargs["base_url"]
    )

    task_impl = SegmentEmbeddingTask(
        artifact_visitor=ArtifactPersistentVisitor(minio_client, postgres_client),
        minio_client=minio_client,
    )

    client = QwenVLEmbeddingClient(config=embed_config)
    try:
        artifacts = await task_impl.execute_template(segments, client)
    finally:
        await client.close()
        await shutdown_postgres_client(postgres_client)

    logger.info(f"[SegmentEmbedding] Done | {len(artifacts)} artifact(s) produced")
    return artifacts
