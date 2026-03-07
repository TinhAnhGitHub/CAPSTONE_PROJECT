from __future__ import annotations

import io

import numpy as np
from PIL import Image as PILImage
from prefect import get_run_logger, task

from video_pipeline.task.base.base_task import TaskConfig, BaseTask
from video_pipeline.core.client.progress import StageRegistry
from video_pipeline.core.artifact import ImageCaptionArtifact, ImageCaptionMultimodalEmbeddingArtifact
from video_pipeline.core.storage.pg_tracker import ArtifactPersistentVisitor
from video_pipeline.core.client.storage.minio import MinioStorageClient
from video_pipeline.core.client.storage.pg import PostgresClient, PgConfig
from video_pipeline.core.client.inference.qwenvl_embed import QwenVLEmbeddingClient, QwenVLEmbeddingConfig
from video_pipeline.config import get_settings


IMAGE_CAPTION_MULTIMODAL_EMBEDDING_CONFIG = TaskConfig.from_yaml("image_caption_multimodal_embedding")
_base_kwargs = IMAGE_CAPTION_MULTIMODAL_EMBEDDING_CONFIG.to_task_kwargs()

_PreprocessedItem = tuple[ImageCaptionArtifact, bytes]


@StageRegistry.register
class ImageCaptionMultimodalEmbeddingTask(BaseTask[list[ImageCaptionArtifact], list[ImageCaptionMultimodalEmbeddingArtifact]]):
    async def preprocess(self, input_data: list[ImageCaptionArtifact]) -> list[_PreprocessedItem]:
        """Download original image bytes for every caption artifact."""
        logger = get_run_logger()
        logger.info(
            f"[ImageCaptionMultimodalEmbeddingTask] Downloading {len(input_data)} image(s) from MinIO"
        )

        preprocessed: list[_PreprocessedItem] = []
        for artifact in input_data:
            prefix = f"s3://{artifact.user_id}/"
            image_object_name = artifact.image_minio_url.removeprefix(prefix)
            image_bytes = self.minio_client.get_object_bytes(
                bucket=artifact.user_id,
                object_name=image_object_name,
            )
            preprocessed.append((artifact, image_bytes))

        logger.info(
            f"[ImageCaptionMultimodalEmbeddingTask] Preprocessing done — "
            f"{len(preprocessed)} image(s) ready"
        )
        return preprocessed

    async def execute_single(
        self,
        item: list[_PreprocessedItem],
        client: QwenVLEmbeddingClient,
    ) -> list[tuple[ImageCaptionMultimodalEmbeddingArtifact, bytes]]:
        """Embed the batch of images via QwenVL multimodal model.

        Args:
            item: list of (ImageCaptionArtifact, image_bytes) — the full batch.
            client: QwenVL embedding client.
            context: Task execution context.

        Returns:
            list of (ImageCaptionMultimodalEmbeddingArtifact, npy_bytes).
        """
        logger = get_run_logger()
        logger.info(f"[ImageCaptionMultimodalEmbeddingTask] Batch embedding {len(item)} image(s)")

        def _to_jpeg(data: bytes) -> bytes:
            buf = io.BytesIO()
            PILImage.open(io.BytesIO(data)).convert("RGB").save(buf, format="JPEG", quality=90)
            return buf.getvalue()

        jpeg_list = [_to_jpeg(image_bytes) for _, image_bytes in item]
        embeddings: list[list[float]] = await client.ainfer_image(jpeg_list)

        output: list[tuple[ImageCaptionMultimodalEmbeddingArtifact, bytes]] = []
        for (caption_artifact, _), embedding_vector in zip(item, embeddings):
            logger.info(
                f"[ImageCaptionMultimodalEmbeddingTask] Done | "
                f"frame={caption_artifact.frame_index} dim={len(embedding_vector)}"
            )

            artifact = ImageCaptionMultimodalEmbeddingArtifact(
                time_stamp=caption_artifact.time_stamp,
                related_frame_fps=caption_artifact.related_video_fps,
                frame_index=caption_artifact.frame_index,
                related_video_id=caption_artifact.related_video_id,
                image_caption_minio_url=caption_artifact.minio_url_path,
                caption_id=caption_artifact.artifact_id,
                image_id=caption_artifact.image_id,
                image_minio_url=caption_artifact.image_minio_url,
                user_id=caption_artifact.user_id,
                object_name=(
                    f"embedding/multimodal_caption/{caption_artifact.related_video_id}/"
                    f"{caption_artifact.frame_index:08d}_{caption_artifact.time_stamp}.npy"
                ),
                metadata={"embedding_dim": len(embedding_vector)},
            )

            npy_buffer = io.BytesIO()
            np.save(npy_buffer, np.array(embedding_vector, dtype=np.float32))
            output.append((artifact, npy_buffer.getvalue()))

        logger.info(
            f"[ImageCaptionMultimodalEmbeddingTask] Batch done — {len(output)} embedding(s) produced"
        )
        return output

    async def postprocess(self, result: list[tuple[ImageCaptionMultimodalEmbeddingArtifact, bytes]]) -> list[ImageCaptionMultimodalEmbeddingArtifact]: 
        artifacts: list[ImageCaptionMultimodalEmbeddingArtifact] = []
        for artifact, npy_bytes in result:
            await self.artifact_visitor.visit_artifact(
                artifact, upload_to_minio=io.BytesIO(npy_bytes)
            )
            artifacts.append(artifact)
        return artifacts

    def format_result(self, result: ImageCaptionMultimodalEmbeddingArtifact) -> str:
        meta = result.metadata or {}
        dim = meta.get("embedding_dim", "?")
        return (
            f"### Frame {result.frame_index} — {result.time_stamp}\n\n"
            f"- **Image URL:** `{result.image_minio_url}`\n"
            f"- **Caption URL:** `{result.image_caption_minio_url}`\n"
            f"- **Embedding Dim:** {dim}\n"
        )


@task(**{**_base_kwargs, "name": "Image Caption Multimodal Embedding Chunk"})  # type: ignore
async def image_caption_multimodal_embedding_chunk_task(
    items: list[ImageCaptionArtifact],
) -> list[ImageCaptionMultimodalEmbeddingArtifact]:
    """Embed images associated with caption artifacts using QwenVL.

    Downloads original images in preprocess(), then embeds the whole batch
    via QwenVL multimodal model concurrently.

    Args:
        items: Batch of ImageCaptionArtifacts whose source images to embed.
        context: Task execution context.

    Returns:
        List of ImageCaptionMultimodalEmbeddingArtifacts, one per caption.
    """
    logger = get_run_logger()
    settings = get_settings()

    logger.info(
        f"[ImageCaptionMultimodalEmbeddingChunk] Starting | {len(items)} item(s) in batch"
    )

    minio_client = MinioStorageClient(
        endpoint=settings.minio.endpoint,
        access_key=settings.minio.access_key,
        secret_key=settings.minio.secret_key,
        secure=settings.minio.secure,
    )
    postgres_client = PostgresClient(
        config=PgConfig(database_url=settings.postgres.connection_string)  # type: ignore
    )

    embedding_config = QwenVLEmbeddingConfig(
        base_url=IMAGE_CAPTION_MULTIMODAL_EMBEDDING_CONFIG.additional_kwargs.get(
            "base_url", "http://qwen_vl_embedding:8080/embedding"
        ),
    )
    logger.info(
        f"[ImageCaptionMultimodalEmbeddingChunk] QwenVL config | base_url={embedding_config.base_url}"
    )

    task_impl = ImageCaptionMultimodalEmbeddingTask(
        artifact_visitor=ArtifactPersistentVisitor(minio_client, postgres_client),
        minio_client=minio_client,
    )
    client = QwenVLEmbeddingClient(config=embedding_config)

    try:
        preprocessed = await task_impl.preprocess(items)
        batch_result = await task_impl.execute_single(preprocessed, client=client)
        batch_artifacts = await task_impl.postprocess(batch_result)
    finally:
        await client.close()

    logger.info(
        f"[ImageCaptionMultimodalEmbeddingChunk] Done | {len(batch_artifacts)} artifact(s) produced"
    )
    return batch_artifacts
