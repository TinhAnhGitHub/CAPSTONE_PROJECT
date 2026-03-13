from __future__ import annotations

import io

import numpy as np
from PIL import Image as PILImage
from prefect import get_run_logger, task

from video_pipeline.task.base.base_task import TaskConfig, BaseTask
from video_pipeline.core.client.progress import StageRegistry
from video_pipeline.core.artifact import (
    SegmentCaptionArtifact,
    SegmentCaptionMultimodalEmbedArtifact,
    AudioSegmentArtifact,
)
from video_pipeline.core.storage.pg_tracker import ArtifactPersistentVisitor
from video_pipeline.core.client.storage.minio import MinioStorageClient
from video_pipeline.core.client.storage.pg.runtime import get_postgres_client, shutdown_postgres_client
from video_pipeline.core.client.inference.qwenvl_embed import QwenVLEmbeddingClient, QwenVLEmbeddingConfig
from video_pipeline.config import get_settings
from video_pipeline.task.image_extraction.helper import FastFrameReader, get_segment_frame_indices


SEGMENT_CAPTION_MULTIMODAL_EMBEDDING_CONFIG = TaskConfig.from_yaml("segment_caption_multimodal_embedding")
_base_kwargs = SEGMENT_CAPTION_MULTIMODAL_EMBEDDING_CONFIG.to_task_kwargs()

_PreprocessedItem = tuple[SegmentCaptionArtifact, list[bytes]]  # artifact + frame images


@StageRegistry.register
class SegmentCaptionMultimodalEmbeddingTask(BaseTask[list[SegmentCaptionArtifact], list[SegmentCaptionMultimodalEmbedArtifact]]):
    """Embed segment captions using QwenVL multimodal model by extracting frames.

    preprocess() loads video and extracts frames for each segment.
    execute_single() embeds the frames via QwenVL.
    postprocess() uploads .npy embedding files to MinIO and persists to Postgres.
    """

    config = SEGMENT_CAPTION_MULTIMODAL_EMBEDDING_CONFIG

    def __init__(
        self,
        artifact_visitor: ArtifactPersistentVisitor,
        minio_client: MinioStorageClient,
        n_frames: int = 6,
        **kwargs
    ):
        super().__init__(artifact_visitor, minio_client, **kwargs)
        self.n_frames = n_frames

    async def preprocess(
        self,
        input_data: list[SegmentCaptionArtifact],
    ) -> list[_PreprocessedItem]:
        """Load video bytes and extract frames for all segments."""
        logger = get_run_logger()
        logger.info(f"[SegmentCaptionMultimodalEmbeddingTask] Loading video for {len(input_data)} segment(s)")

        if not input_data:
            return []

        first_artifact = input_data[0]
        video_minio_url = first_artifact.related_video_minio_url
        if not video_minio_url:
            logger.warning("No video URL found in segment caption, skipping frame extraction")
            return []

        async with self.minio_client.fetch_object_from_s3(
            s3_url=video_minio_url, suffix=".mp4"
        ) as video_path:
            with open(video_path, "rb") as f:
                video_bytes = f.read()

            preprocessed: list[_PreprocessedItem] = []
            for seg in input_data:
                reader = FastFrameReader(video_bytes)

                start_frame = seg.start_frame
                end_frame = seg.end_frame

                frame_indices = get_segment_frame_indices(start_frame, end_frame, self.n_frames)

                image_bytes_list = []
                for fi in frame_indices:
                    try:
                        frame_bytes = reader.get_frame(fi)
                        image_bytes_list.append(frame_bytes)
                    except Exception as e:
                        logger.warning(f"Failed to extract frame {fi}: {e}")

                if image_bytes_list:
                    preprocessed.append((seg, image_bytes_list))

        logger.info(f"[SegmentCaptionMultimodalEmbeddingTask] Video loaded — {len(preprocessed)} segment(s) ready")
        return preprocessed

    async def execute(
        self,
        preprocessed: list[_PreprocessedItem],
        client: QwenVLEmbeddingClient,
    ) -> list[tuple[SegmentCaptionMultimodalEmbedArtifact, bytes]]:
        """Wrapper that calls execute_single for the batch."""
        return await self.execute_single(preprocessed, client)

    async def execute_single(
        self,
        item: list[_PreprocessedItem],
        client: QwenVLEmbeddingClient,
    ) -> list[tuple[SegmentCaptionMultimodalEmbedArtifact, bytes]]:
        """Embed the batch of segment frame images via QwenVL multimodal model.

        Args:
            item: list of (SegmentCaptionArtifact, image_bytes_list) — the full batch.
            client: QwenVL embedding client.

        Returns:
            list of (SegmentCaptionMultimodalEmbedArtifact, npy_bytes).
        """
        logger = get_run_logger()
        logger.info(f"[SegmentCaptionMultimodalEmbeddingTask] Batch embedding {len(item)} segment(s)")

        output: list[tuple[SegmentCaptionMultimodalEmbedArtifact, bytes]] = []

        for caption_artifact, image_bytes_list in item:
            def _to_jpeg(data: bytes) -> bytes:
                buf = io.BytesIO()
                PILImage.open(io.BytesIO(data)).convert("RGB").save(buf, format="JPEG", quality=90)
                return buf.getvalue()

            jpeg_list = [_to_jpeg(img) for img in image_bytes_list]
            embeddings: list[list[float]] = await client.ainfer_image(jpeg_list)

            if embeddings:
                # Average the embeddings from all frames
                avg_embedding = np.mean(embeddings, axis=0).tolist()
                embedding_dim = len(avg_embedding)

                logger.info(
                    f"[SegmentCaptionMultimodalEmbeddingTask] Done | "
                    f"segment={caption_artifact.start_frame}-{caption_artifact.end_frame} "
                    f"frames={len(jpeg_list)} dim={embedding_dim}"
                )

                artifact = SegmentCaptionMultimodalEmbedArtifact(
                    related_video_fps=caption_artifact.related_video_fps,
                    related_video_id=caption_artifact.related_video_id,
                    start_frame=caption_artifact.start_frame,
                    end_frame=caption_artifact.end_frame,
                    start_timestamp=caption_artifact.start_timestamp,
                    end_timestamp=caption_artifact.end_timestamp,
                    related_segment_caption_url=caption_artifact.minio_url_path,
                    segment_cap_id=caption_artifact.artifact_id,
                    user_id=caption_artifact.user_id,
                    object_name=(
                        f"embedding/multimodal_segment_caption/{caption_artifact.related_video_id}/"
                        f"{caption_artifact.start_frame}_{caption_artifact.end_frame}_"
                        f"{caption_artifact.start_timestamp}_{caption_artifact.end_timestamp}.npy"
                    ),
                    metadata={
                        "embedding_dim": embedding_dim,
                        "n_frames": len(jpeg_list),
                    },
                )

                npy_buffer = io.BytesIO()
                np.save(npy_buffer, np.array(avg_embedding, dtype=np.float32))
                output.append((artifact, npy_buffer.getvalue()))

        logger.info(
            f"[SegmentCaptionMultimodalEmbeddingTask] Batch done — {len(output)} embedding(s) produced"
        )
        return output

    async def postprocess(self, result: list[tuple[SegmentCaptionMultimodalEmbedArtifact, bytes]]) -> list[SegmentCaptionMultimodalEmbedArtifact]:
        """Upload .npy files to MinIO and persist artifact metadata to Postgres."""
        artifacts: list[SegmentCaptionMultimodalEmbedArtifact] = []
        for artifact, npy_bytes in result:
            await self.artifact_visitor.visit_artifact(
                artifact, upload_to_minio=io.BytesIO(npy_bytes)
            )
            artifacts.append(artifact)
        return artifacts

    def format_result(self, result: SegmentCaptionMultimodalEmbedArtifact) -> str:
        meta = result.metadata or {}
        dim = meta.get("embedding_dim", "?")
        n_frames = meta.get("n_frames", "?")
        return (
            f"### Segment {result.start_frame}-{result.end_frame} — {result.start_timestamp}\n\n"
            f"- **Caption URL:** `{result.related_segment_caption_url}`\n"
            f"- **Frames:** {n_frames}\n"
            f"- **Embedding Dim:** {dim}\n"
        )

    @staticmethod
    async def summary_artifact(final_result: list[SegmentCaptionMultimodalEmbedArtifact]) -> None:
        """Create a Prefect artifact summarizing segment caption multimodal embeddings."""
        if not final_result:
            return

        first = final_result[0]

        segment_rows = ""
        for i, seg in enumerate(final_result):
            meta = seg.metadata or {}
            dim = meta.get("embedding_dim", "?")
            n_frames = meta.get("n_frames", "?")
            segment_rows += (
                f"| {i + 1} | {seg.start_timestamp} | {seg.end_timestamp} | "
                f"{seg.start_frame}-{seg.end_frame} | {n_frames} | {dim} |\n"
            )

        markdown = (
            f"# Segment Caption Multimodal Embedding Summary\n\n"
            f"| Field | Value |\n"
            f"|-------|-------|\n"
            f"| **Video ID** | `{first.related_video_id}` |\n"
            f"| **User ID** | `{first.user_id}` |\n"
            f"| **Segments Embedded** | `{len(final_result)}` |\n\n"
            f"## Multimodal Embeddings\n\n"
            f"| # | Start | End | Frames | N Frames | Embedding Dim |\n"
            f"|---|-------|-----|--------|----------|---------------|\n"
            f"{segment_rows}"
        )

        from prefect.artifacts import acreate_markdown_artifact
        await acreate_markdown_artifact(
            key=f"segment-caption-mm-embedding-{first.related_video_id}".lower(),
            markdown=markdown,
            description=f"Segment caption multimodal embedding summary for video {first.related_video_id}",
        )


@task(**{**_base_kwargs, "name": "Segment Caption Multimodal Embedding Chunk"})  # type: ignore
async def segment_caption_multimodal_embedding_chunk_task(
    items: list[SegmentCaptionArtifact],
) -> list[SegmentCaptionMultimodalEmbedArtifact]:
    """Embed frames from segment captions using QwenVL.

    Downloads video, extracts frames for each segment, then embeds them
    via QwenVL multimodal model.

    Args:
        items: Batch of SegmentCaptionArtifacts whose segments to embed.

    Returns:
        List of SegmentCaptionMultimodalEmbedArtifacts, one per segment.
    """
    logger = get_run_logger()
    settings = get_settings()

    logger.info(
        f"[SegmentCaptionMultimodalEmbeddingChunk] Starting | {len(items)} item(s) in batch"
    )

    minio_client = MinioStorageClient(
        endpoint=settings.minio.endpoint,
        access_key=settings.minio.access_key,
        secret_key=settings.minio.secret_key,
        secure=settings.minio.secure,
    )
    postgres_client = await get_postgres_client()

    n_frames = SEGMENT_CAPTION_MULTIMODAL_EMBEDDING_CONFIG.additional_kwargs.get("n_frames", 6)
    embedding_config = QwenVLEmbeddingConfig(
        base_url=SEGMENT_CAPTION_MULTIMODAL_EMBEDDING_CONFIG.additional_kwargs.get(
            "base_url", "http://qwen_vl_embedding:8080/embedding"
        ),
    )
    logger.info(
        f"[SegmentCaptionMultimodalEmbeddingChunk] QwenVL config | base_url={embedding_config.base_url}"
    )

    task_impl = SegmentCaptionMultimodalEmbeddingTask(
        artifact_visitor=ArtifactPersistentVisitor(minio_client, postgres_client),
        minio_client=minio_client,
        n_frames=n_frames,
    )
    client = QwenVLEmbeddingClient(config=embedding_config)

    try:
        preprocessed = await task_impl.preprocess(items)
        if preprocessed:
            batch_result = await task_impl.execute_single(preprocessed, client=client)
            artifacts = await task_impl.postprocess(batch_result)
            await SegmentCaptionMultimodalEmbeddingTask.summary_artifact(artifacts)
        else:
            artifacts = []
    finally:
        await client.close()
        await shutdown_postgres_client(postgres_client)

    logger.info(
        f"[SegmentCaptionMultimodalEmbeddingChunk] Done | {len(artifacts)} artifact(s) produced"
    )
    return artifacts