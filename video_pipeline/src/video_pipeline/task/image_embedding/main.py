from __future__ import annotations

import io
from typing import cast
import re
from PIL import Image as PILImage
import numpy as np
from prefect import get_run_logger, task
from prefect.artifacts import acreate_markdown_artifact, acreate_table_artifact

from video_pipeline.task.base.base_task import TaskConfig, BaseTask
from video_pipeline.core.client.progress import StageRegistry
from video_pipeline.core.artifact import ImageCaptionArtifact, ImageEmbeddingArtifact
from video_pipeline.core.storage.pg_tracker import ArtifactPersistentVisitor
from video_pipeline.core.client.storage.minio import MinioStorageClient
from video_pipeline.core.client.storage.pg.runtime import get_postgres_client, shutdown_postgres_client
from video_pipeline.core.client.inference.qwenvl_embed import QwenVLEmbeddingClient, QwenVLEmbeddingConfig
from video_pipeline.config import get_settings


IMAGE_EMBEDDING_CONFIG = TaskConfig.from_yaml("image_embedding")
_base_kwargs = IMAGE_EMBEDDING_CONFIG.to_task_kwargs()

_PreprocessedItem = tuple[ImageCaptionArtifact, bytes]


@StageRegistry.register
class ImageEmbeddingTask(BaseTask[list[ImageCaptionArtifact], list[ImageEmbeddingArtifact]]):
    """Embed images with their captions using QwenVL multimodal embedding."""

    config = IMAGE_EMBEDDING_CONFIG

    async def preprocess(self, input_data: list[ImageCaptionArtifact]) -> list[_PreprocessedItem]:
        logger = get_run_logger()
        logger.info(f"[ImageEmbeddingTask] Downloading {len(input_data)} image(s) from MinIO")

        preprocessed: list[_PreprocessedItem] = []
        for artifact in input_data:
            prefix = f"s3://{artifact.user_id}/"
            image_object_name = artifact.image_minio_url.removeprefix(prefix)
            image_bytes = self.minio_client.get_object_bytes(
                bucket=artifact.user_id,
                object_name=image_object_name,
            )
            preprocessed.append((artifact, image_bytes))

        logger.info(f"[ImageEmbeddingTask] Preprocessing done — {len(preprocessed)} image(s) ready")
        return preprocessed

    async def execute(
        self,
        preprocessed: list[_PreprocessedItem],
        client: QwenVLEmbeddingClient,
    ) -> list[tuple[ImageEmbeddingArtifact, bytes]]:
        """Embed each image + caption pair via QwenVL.

        Uses _infer_single_image with the caption text to produce a true
        multimodal embedding that combines visual and textual information.

        Args:
            preprocessed: list of (ImageCaptionArtifact, image_bytes).
            client: QwenVL embedding client.

        Returns:
            list of (ImageEmbeddingArtifact, npy_bytes) ready for postprocess.
        """
        logger = get_run_logger()
        logger.info(f"[ImageEmbeddingTask] Embedding {len(preprocessed)} image(s) with captions")

        import asyncio

        async def embed_single(
            caption_artifact: ImageCaptionArtifact, image_bytes: bytes
        ) -> tuple[ImageEmbeddingArtifact, bytes] | None:
            def _to_jpeg(data: bytes, size: int = 640) -> bytes:
                buf = io.BytesIO()
                img = PILImage.open(io.BytesIO(data)).convert("RGB")
                img = img.resize((size, size), PILImage.Resampling.LANCZOS)
                img.save(buf, format="JPEG", quality=90)
                return buf.getvalue()

            jpeg_bytes = _to_jpeg(image_bytes)
            caption_text = cast(dict, caption_artifact.metadata)['caption']

            embedding_vector = await client._infer_single_image(jpeg_bytes, text=caption_text)

            if not embedding_vector:
                logger.warning(f"No embedding for frame {caption_artifact.frame_index}")
                return None

            logger.info(
                f"[ImageEmbeddingTask] Done | frame={caption_artifact.frame_index} "
                f"dim={len(embedding_vector)}"
            )
            artifact = ImageEmbeddingArtifact(
                caption_text=cast(dict, caption_artifact.metadata)['caption'],
                frame_index=caption_artifact.frame_index,
                timestamp=caption_artifact.timestamp,
                timestamp_sec=caption_artifact.timestamp_sec,
                related_video_id=caption_artifact.related_video_id,
                related_video_fps=caption_artifact.related_video_fps,
                image_minio_url=caption_artifact.image_minio_url,
                extension=".npy",
                image_id=caption_artifact.image_id,
                user_id=caption_artifact.user_id,
                object_name=(
                    f"embedding/image/{caption_artifact.related_video_id}/"
                    f"{caption_artifact.frame_index:08d}_{caption_artifact.timestamp}.npy"
                ),
                metadata={
                    "embedding_dim": len(embedding_vector),
                    "caption_preview": caption_text[:100] if caption_text else "",
                },
            )

            npy_buffer = io.BytesIO()
            np.save(npy_buffer, np.array(embedding_vector, dtype=np.float32))
            npy_bytes = npy_buffer.getvalue()

            return (artifact, npy_bytes)

        tasks = [embed_single(artifact, img_bytes) for artifact, img_bytes in preprocessed]
        results = await asyncio.gather(*tasks)

        output = [r for r in results if r is not None]
        logger.info(f"[ImageEmbeddingTask] Batch done — {len(output)} embedding(s) produced")
        return output

    async def postprocess(self, result: list[tuple[ImageEmbeddingArtifact, bytes]]) -> list[ImageEmbeddingArtifact]:  # type: ignore[override]
        """Upload embedding .npy files to MinIO and persist artifact metadata to Postgres."""
        artifacts: list[ImageEmbeddingArtifact] = []
        for artifact, npy_bytes in result:
            await self.artifact_visitor.visit_artifact(
                artifact, upload_to_minio=io.BytesIO(npy_bytes)
            )
            artifacts.append(artifact)
        return artifacts

    @staticmethod
    async def summary_artifact(final_result: list[ImageEmbeddingArtifact]) -> None:
        """Create a Prefect markdown artifact summarising an image embedding batch."""
        if not final_result:
            return

        first = final_result[0]
        video_id = first.related_video_id
        key = re.sub(r"[^a-z0-9-]", "-", f"image-embedding-{video_id}".lower())

        embedding_dim = (first.metadata or {}).get("embedding_dim", "?")

        frame_rows = ""
        for artifact in final_result:
            frame_rows += (
                f"| {artifact.frame_index} | {artifact.timestamp} "
                f"| `{artifact.minio_url_path}` |\n"
            )

        markdown = (
f"# Image Embedding Summary (Multimodal)\n\n"
f"| Field | Value |\n"
f"|-------|-------|\n"
f"| **Video ID** | `{video_id}` |\n"
f"| **User ID** | `{first.user_id}` |\n"
f"| **FPS** | `{first.related_video_fps}` |\n"
f"| **Frames Embedded** | `{len(final_result)} |\n"
f"| **Embedding Dim** | `{embedding_dim}` |\n\n"
f"## Embedded Frames (Image + Caption)\n\n"
f"| Frame | Timestamp | Embedding (.npy) |\n"
f"|-------|-----------|------------------|\n"
f"{frame_rows}"
        )

        await acreate_markdown_artifact(
            key=key,
            markdown=markdown,
            description=f"Image embedding summary for video {video_id}",
        )

        await acreate_table_artifact(
            table=[
                {"Field": "Video ID", "Value": str(video_id)},
                {"Field": "User ID", "Value": str(first.user_id)},
                {"Field": "FPS", "Value": str(first.related_video_fps)},
                {"Field": "Frames Embedded", "Value": str(len(final_result))},
                {"Field": "Embedding Dim", "Value": str(embedding_dim)},
            ],
            key=f"{key}-summary-table",
            description=f"Image embedding stats table for video {video_id}",
        )

        await acreate_table_artifact(
            table=[
                {
                    "Frame": artifact.frame_index,
                    "Timestamp": artifact.timestamp,
                    "Embedding (.npy)": str(artifact.minio_url_path),
                }
                for artifact in final_result
            ],
            key=f"{key}-frames-table",
            description=f"Embedded frames for video {video_id}",
        )


@task(**{**_base_kwargs, "name": "Image Embedding Chunk"})  # type: ignore
async def image_embedding_chunk_task(
    items: list[ImageCaptionArtifact],
) -> list[ImageEmbeddingArtifact]:
    """Embed a batch of images with captions using ImageEmbeddingTask.

    Takes ImageCaptionArtifacts (image + caption), downloads images, and creates
    multimodal embeddings that combine visual and textual information.

    Args:
        items: Batch of ImageCaptionArtifacts to embed.

    Returns:
        List of ImageEmbeddingArtifacts, one per frame in the batch.
    """
    logger = get_run_logger()
    settings = get_settings()

    logger.info(f"[ImageEmbeddingChunk] Starting | {len(items)} frame(s) in batch")

    minio_client = MinioStorageClient(
        endpoint=settings.minio.endpoint,
        access_key=settings.minio.access_key,
        secret_key=settings.minio.secret_key,
        secure=settings.minio.secure,
    )
    postgres_client = await get_postgres_client()
    logger.info(f"[ImageEmbeddingChunk] Clients initialized | minio={settings.minio.endpoint}")

    embedding_config = QwenVLEmbeddingConfig(
        base_url=IMAGE_EMBEDDING_CONFIG.additional_kwargs.get("base_url", "http://qwen_vl_embedding:8080/embedding"),
    )
    logger.info(f"[ImageEmbeddingChunk] QwenVL config | base_url={embedding_config.base_url}")

    task_impl = ImageEmbeddingTask(
        artifact_visitor=ArtifactPersistentVisitor(minio_client, postgres_client),
        minio_client=minio_client,
    )
    client = QwenVLEmbeddingClient(config=embedding_config)

    try:
        all_artifacts = await task_impl.execute_template(items, client)

    finally:
        await client.close()
        await shutdown_postgres_client(postgres_client)

    logger.info(f"[ImageEmbeddingChunk] Done | {len(all_artifacts)} artifact(s) produced")
    return all_artifacts
