from __future__ import annotations

import io
from PIL import Image as PILImage
import numpy as np

from prefect import get_run_logger, task
from prefect.artifacts import acreate_markdown_artifact

from video_pipeline.task.base.base_task import TaskConfig, BaseTask
from video_pipeline.core.client.progress import StageRegistry
from video_pipeline.core.artifact import SegmentCaptionArtifact, SegmentEmbeddingArtifact
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
class SegmentEmbeddingTask(BaseTask[list[SegmentCaptionArtifact], list[SegmentEmbeddingArtifact]]):
    """Embed segment captions with frames using QwenVL multimodal embedding.

    Takes SegmentCaptionArtifact (caption + frame info) and produces a single
    multimodal embedding per segment that combines visual frames and caption text
    using ainfer_video with the caption text.
    """

    config = SEGMENT_EMBEDDING_CONFIG

    async def preprocess(
        self,
        input_data: list[SegmentCaptionArtifact],
    ) -> list[SegmentCaptionArtifact]:
        """Return segments unchanged - video will be loaded in execute().

        Note: We don't load video bytes here to avoid memory bloat.
        Video is streamed to disk in execute() and processed within
        the context manager to ensure proper cleanup.
        """
        logger = get_run_logger()
        logger.info(f"[SegmentEmbeddingTask] Preparing {len(input_data)} segment(s)")
        return input_data

    async def execute(
        self,
        preprocessed: list[SegmentCaptionArtifact],
        client: QwenVLEmbeddingClient,
    ) -> list[SegmentEmbeddingArtifact]:
        """Extract frames from each segment and embed with caption text.

        Downloads video via streaming (no memory bloat) and processes
        all segments within the context manager. Uses ainfer_video with
        caption text for true multimodal embedding.
        """
        logger = get_run_logger()
        n_frames = self.config.additional_kwargs.get("n_frames", 6)

        if not preprocessed:
            return []

        logger.info(
            f"[SegmentEmbeddingTask] Embedding {len(preprocessed)} segment(s) with {n_frames} frames each"
        )

        related_video_minio_url = preprocessed[0].related_video_minio_url
        artifacts = []

        async with self.minio_client.fetch_object_streaming(
            s3_url=related_video_minio_url,
            suffix='.mp4'
        ) as video_path:
            for seg in preprocessed:
                reader = FastFrameReader(video_path)

                start_frame = seg.start_frame
                end_frame = seg.end_frame

                frame_indices = get_segment_frame_indices(start_frame, end_frame, n_frames)

                image_bytes_list = []
                for fi in frame_indices:
                    try:
                        frame_bytes = reader.get_frame(fi)
                        image_bytes_list.append(frame_bytes)
                    except Exception as e:
                        logger.warning(f"Failed to extract frame {fi}: {e}")

                if not image_bytes_list:
                    logger.warning(f"No frames extracted for segment {start_frame}-{end_frame}, skipping")
                    continue

                def _to_jpeg(data: bytes, size: int = 640) -> bytes:
                    buf = io.BytesIO()
                    img = PILImage.open(io.BytesIO(data)).convert("RGB")
                    img = img.resize((size, size), PILImage.Resampling.LANCZOS)
                    img.save(buf, format="JPEG", quality=90)
                    return buf.getvalue()

                jpeg_list = [_to_jpeg(img) for img in image_bytes_list]

                caption_text = seg.summary_caption
                embedding = await client.ainfer_video(jpeg_list, text=caption_text)

                if embedding:
                    artifact = SegmentEmbeddingArtifact(
                        related_audio_segment_artifact_id=seg.related_audio_segment_artifact_id,
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
                        caption_text=caption_text,
                        object_name=(
                            f"embedding/segment/{seg.related_video_id}/"
                            f"{start_frame}_{end_frame}_{seg.start_timestamp}_{seg.end_timestamp}.npy"
                        ),
                        metadata={
                            "embedding": embedding,
                            "n_frames_extracted": len(frame_indices),
                            "caption_preview": caption_text[:100] if caption_text else "",
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
            f"# Segment Embedding Summary (Multimodal)\n\n"
            f"| Field | Value |\n"
            f"|-------|-------|\n"
            f"| **Video ID** | `{first.related_video_id}` |\n"
            f"| **User ID** | `{first.user_id}` |\n"
            f"| **Segments Embedded** | `{len(final_result)}` |\n\n"
            f"## Segment Embeddings (Frames + Caption)\n\n"
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
    segments: list[SegmentCaptionArtifact],
) -> list[SegmentEmbeddingArtifact]:
    """Process segment captions into multimodal embeddings.

    Takes SegmentCaptionArtifacts, extracts frames, and creates multimodal
    embeddings that combine visual frames and caption text.

    Args:
        segments: List of SegmentCaptionArtifact

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
