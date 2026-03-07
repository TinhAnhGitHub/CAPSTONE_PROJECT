from __future__ import annotations

import io
import re
from PIL import Image as PILImage
import numpy as np
from prefect import get_run_logger, task
from prefect.artifacts import acreate_markdown_artifact, acreate_table_artifact

from video_pipeline.task.base.base_task import TaskConfig, BaseTask
from video_pipeline.core.client.progress import StageRegistry
from video_pipeline.core.artifact import ImageArtifact, ImageEmbeddingArtifact
from video_pipeline.core.storage.pg_tracker import ArtifactPersistentVisitor
from video_pipeline.core.client.storage.minio import MinioStorageClient
from video_pipeline.core.client.storage.pg import PostgresClient, PgConfig
from video_pipeline.core.client.inference.qwenvl_embed import QwenVLEmbeddingClient, QwenVLEmbeddingConfig
from video_pipeline.config import get_settings


IMAGE_EMBEDDING_CONFIG = TaskConfig.from_yaml("image_embedding")
_base_kwargs = IMAGE_EMBEDDING_CONFIG.to_task_kwargs()

_PreprocessedItem = tuple[ImageArtifact, bytes]


@StageRegistry.register
class ImageEmbeddingTask(BaseTask[list[ImageArtifact], list[ImageEmbeddingArtifact]]):
    """Embed extracted frames using QwenVL embedding model in batch.

    preprocess() downloads image bytes for the batch from MinIO once.
    execute_single() sends the whole batch to QwenVL concurrently and builds embedding artifacts.
    postprocess() uploads embedding .npy files to MinIO and persists to Postgres.
    """

    config = IMAGE_EMBEDDING_CONFIG

    async def preprocess(self, input_data: list[ImageArtifact]) -> list[_PreprocessedItem]:
        """Download image bytes for every artifact in the batch."""
        logger = get_run_logger()
        logger.info(f"[ImageEmbeddingTask] Downloading {len(input_data)} image(s) from MinIO")

        preprocessed: list[_PreprocessedItem] = []
        for artifact in input_data:
            assert artifact.object_name is not None, (
                f"ImageArtifact {artifact.artifact_id} has no object_name"
            )
            image_bytes = self.minio_client.get_object_bytes(
                bucket=artifact.user_id,
                object_name=artifact.object_name,
            )
            preprocessed.append((artifact, image_bytes))

        logger.info(f"[ImageEmbeddingTask] Preprocessing done — {len(preprocessed)} image(s) ready")
        return preprocessed

    async def execute(
        self,
        preprocessed: list[_PreprocessedItem],
        client: QwenVLEmbeddingClient,
    ) -> list[tuple[ImageEmbeddingArtifact, bytes]]:
        """Embed a batch of image frames via QwenVL concurrently.

        Args:
            item: list of (ImageArtifact, frame_bytes) — the full batch.
            client: QwenVL embedding client.
            context: Task execution context.

        Returns:
            list of (ImageEmbeddingArtifact, npy_bytes) ready for postprocess.
        """
        logger = get_run_logger()
        logger.info(f"[ImageEmbeddingTask] Batch embedding {len(preprocessed)} frame(s)")

        
        def _to_jpeg(data: bytes) -> bytes:
            buf = io.BytesIO()
            PILImage.open(io.BytesIO(data)).convert("RGB").save(buf, format="JPEG", quality=90)
            return buf.getvalue()

        image_bytes_list = [_to_jpeg(frame_bytes) for _, frame_bytes in preprocessed]
        embeddings: list[list[float]] = await client.ainfer_image(image_bytes_list)

        output: list[tuple[ImageEmbeddingArtifact, bytes]] = []
        for (image_artifact, _), embedding_vector in zip(preprocessed, embeddings):
            logger.info(
                f"[ImageEmbeddingTask] Embedding done | frame={image_artifact.frame_index} "
                f"dim={len(embedding_vector)}"
            )

            artifact = ImageEmbeddingArtifact(
                frame_index=image_artifact.frame_index,
                time_stamp=image_artifact.timestamp,
                related_video_id=image_artifact.related_video_id,
                related_video_fps=image_artifact.related_video_fps,
                image_minio_url=image_artifact.minio_url_path,
                extension=".npy",
                image_id=image_artifact.artifact_id,
                user_id=image_artifact.user_id,
                object_name=(
                    f"embedding/image/{image_artifact.related_video_id}/"
                    f"{image_artifact.frame_index:08d}_{image_artifact.timestamp}.npy"
                ),
                metadata={"embedding_dim": len(embedding_vector)},
            )

            npy_buffer = io.BytesIO()
            np.save(npy_buffer, np.array(embedding_vector, dtype=np.float32))
            npy_bytes = npy_buffer.getvalue()

            output.append((artifact, npy_bytes))

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
                f"| {artifact.frame_index} | {artifact.time_stamp} "
                f"| `{artifact.minio_url_path}` |\n"
            )

        markdown = (
f"# Image Embedding Summary\n\n"
f"| Field | Value |\n"
f"|-------|-------|\n"
f"| **Video ID** | `{video_id}` |\n"
f"| **User ID** | `{first.user_id}` |\n"
f"| **FPS** | `{first.related_video_fps}` |\n"
f"| **Frames Embedded** | `{len(final_result)}` |\n"
f"| **Embedding Dim** | `{embedding_dim}` |\n\n"
f"## Embedded Frames\n\n"
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
                    "Timestamp": artifact.time_stamp,
                    "Embedding (.npy)": str(artifact.minio_url_path),
                }
                for artifact in final_result
            ],
            key=f"{key}-frames-table",
            description=f"Embedded frames for video {video_id}",
        )


@task(**{**_base_kwargs, "name": "Image Embedding Chunk"})  # type: ignore
async def image_embedding_chunk_task(
    items: list[ImageArtifact],
) -> list[ImageEmbeddingArtifact]:
    """Embed a batch of image frames using ImageEmbeddingTask.execute().

    Downloads image bytes in preprocess(), then calls execute_single() once with
    the whole batch, embedding all frames concurrently via QwenVL.

    Args:
        items: Batch of ImageArtifacts to embed.
        context: Task execution context.

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
    postgres_client = PostgresClient(
        config=PgConfig(database_url=settings.postgres.connection_string)  # type: ignore
    )
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

    logger.info(f"[ImageEmbeddingChunk] Done | {len(all_artifacts)} artifact(s) produced")
    return all_artifacts
