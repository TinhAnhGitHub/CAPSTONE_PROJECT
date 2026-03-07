from __future__ import annotations

import io

import numpy as np
from prefect import get_run_logger, task

from video_pipeline.task.base.base_task import TaskConfig, BaseTask
from video_pipeline.core.client.progress import StageRegistry
from video_pipeline.core.artifact import ImageCaptionArtifact, TextCaptionEmbeddingArtifact
from video_pipeline.core.storage.pg_tracker import ArtifactPersistentVisitor
from video_pipeline.core.client.storage.minio import MinioStorageClient
from video_pipeline.core.client.storage.pg import PostgresClient, PgConfig
from video_pipeline.core.client.inference.te_client import MMBertClient, MMBertConfig
from video_pipeline.config import get_settings


IMAGE_CAPTION_EMBEDDING_CONFIG = TaskConfig.from_yaml("image_caption_embedding")
_base_kwargs = IMAGE_CAPTION_EMBEDDING_CONFIG.to_task_kwargs()

# After preprocess: caption text extracted from artifact metadata
_PreprocessedItem = tuple[ImageCaptionArtifact, str]


@StageRegistry.register
class ImageCaptionEmbeddingTask(BaseTask[list[ImageCaptionArtifact], list[TextCaptionEmbeddingArtifact]]):
    """Embed caption texts using mmBERT in batch.

    preprocess() extracts the caption text from each artifact's metadata.
    execute_single() sends the whole batch to mmBERT in a single request.
    postprocess() uploads .npy embedding files to MinIO and persists to Postgres.
    """

    config = IMAGE_CAPTION_EMBEDDING_CONFIG

    async def preprocess(self, input_data: list[ImageCaptionArtifact]) -> list[_PreprocessedItem]:
        """Extract caption text from each artifact's metadata."""
        logger = get_run_logger()
        logger.info(f"[ImageCaptionEmbeddingTask] Preparing {len(input_data)} caption(s)")

        preprocessed: list[_PreprocessedItem] = []
        for artifact in input_data:
            caption_text: str = (artifact.metadata or {}).get("caption", "")
            preprocessed.append((artifact, caption_text))

        logger.info(f"[ImageCaptionEmbeddingTask] Preprocessing done — {len(preprocessed)} caption(s) ready")
        return preprocessed

    async def execute_single(
        self,
        item: list[_PreprocessedItem],
        client: MMBertClient,
        context: TaskExecutionContext,
    ) -> list[tuple[TextCaptionEmbeddingArtifact, bytes]]:
        """Embed a batch of caption texts via mmBERT in one request.

        Args:
            item: list of (ImageCaptionArtifact, caption_text) — the full batch.
            client: mmBERT embedding client.
            context: Task execution context.

        Returns:
            list of (TextCaptionEmbeddingArtifact, npy_bytes) ready for postprocess.
        """
        logger = get_run_logger()
        logger.info(f"[ImageCaptionEmbeddingTask] Embedding {len(item)} caption(s) via mmBERT")

        texts = [caption_text for _, caption_text in item]
        embeddings: list[list[float]] | None = await client.ainfer(texts)

        if embeddings is None:
            raise RuntimeError("mmBERT ainfer returned None — check server health")

        output: list[tuple[TextCaptionEmbeddingArtifact, bytes]] = []
        for (caption_artifact, caption_text), embedding_vector in zip(item, embeddings):
            logger.info(
                f"[ImageCaptionEmbeddingTask] Embedding done | frame={caption_artifact.frame_index} "
                f"dim={len(embedding_vector)} text={caption_text[:60]!r}"
            )

            artifact = TextCaptionEmbeddingArtifact(
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
                    f"embedding/image_caption/{caption_artifact.related_video_id}/"
                    f"{caption_artifact.frame_index:08d}_{caption_artifact.time_stamp}.npy"
                ),
                metadata={"embedding_dim": len(embedding_vector)},
            )

            npy_buffer = io.BytesIO()
            np.save(npy_buffer, np.array(embedding_vector, dtype=np.float32))
            output.append((artifact, npy_buffer.getvalue()))

        logger.info(f"[ImageCaptionEmbeddingTask] Batch done — {len(output)} embedding(s) produced")
        return output

    async def postprocess(self, result: list[tuple[TextCaptionEmbeddingArtifact, bytes]]) -> list[TextCaptionEmbeddingArtifact]:  # type: ignore[override]
        """Upload .npy files to MinIO and persist artifact metadata to Postgres."""
        artifacts: list[TextCaptionEmbeddingArtifact] = []
        for artifact, npy_bytes in result:
            await self.artifact_visitor.visit_artifact(
                artifact, upload_to_minio=io.BytesIO(npy_bytes)
            )
            artifacts.append(artifact)
        return artifacts

    def format_result(self, result: TextCaptionEmbeddingArtifact) -> str:
        meta = result.metadata or {}
        dim = meta.get("embedding_dim", "?")
        return (
            f"### Frame {result.frame_index} — {result.time_stamp}\n\n"
            f"- **Caption URL:** `{result.image_caption_minio_url}`\n"
            f"- **Embedding Dim:** {dim}\n"
        )


@task(**{**_base_kwargs, "name": "Image Caption Embedding Chunk"})  # type: ignore
async def image_caption_embedding_chunk_task(
    items: list[ImageCaptionArtifact],
    context: TaskExecutionContext,
) -> list[TextCaptionEmbeddingArtifact]:
    """Embed a batch of caption texts using mmBERT.

    Sends all captions in a single batched request to the mmBERT server.

    Args:
        items: Batch of ImageCaptionArtifacts whose text to embed.
        context: Task execution context.

    Returns:
        List of TextCaptionEmbeddingArtifacts, one per caption in the batch.
    """
    logger = get_run_logger()
    settings = get_settings()

    logger.info(f"[ImageCaptionEmbeddingChunk] Starting | {len(items)} caption(s) in batch")

    minio_client = MinioStorageClient(
        endpoint=settings.minio.endpoint,
        access_key=settings.minio.access_key,
        secret_key=settings.minio.secret_key,
        secure=settings.minio.secure,
    )
    postgres_client = PostgresClient(
        config=PgConfig(database_url=settings.postgres.connection_string)  # type: ignore
    )

    mmbert_config = MMBertConfig(
        model_name=IMAGE_CAPTION_EMBEDDING_CONFIG.additional_kwargs.get("model_name", "mmbert"),
        base_url=IMAGE_CAPTION_EMBEDDING_CONFIG.additional_kwargs.get("base_url", "http://mmbert:8000"),
    )
    logger.info(f"[ImageCaptionEmbeddingChunk] mmBERT config | base_url={mmbert_config.base_url}")

    task_impl = ImageCaptionEmbeddingTask(
        artifact_visitor=ArtifactPersistentVisitor(minio_client, postgres_client),
        minio_client=minio_client,
    )
    client = MMBertClient(config=mmbert_config)

    try:
        preprocessed = await task_impl.preprocess(items)
        all_artifacts: list[TextCaptionEmbeddingArtifact] = []
        async for batch_result in task_impl.execute([preprocessed], client, context):
            batch_artifacts = await task_impl.postprocess(batch_result)
            all_artifacts.extend(batch_artifacts)
        await task_impl.create_summary_artifact(all_artifacts, context)
    finally:
        await client.close()

    logger.info(f"[ImageCaptionEmbeddingChunk] Done | {len(all_artifacts)} artifact(s) produced")
    return all_artifacts
