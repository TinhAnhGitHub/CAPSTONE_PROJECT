"""Image Qdrant indexing task."""

from __future__ import annotations

from prefect import get_run_logger, task

from video_pipeline.core.artifact import ImageEmbeddingArtifact
from video_pipeline.core.client.progress import StageRegistry
from video_pipeline.core.client.storage.minio import MinioStorageClient
from video_pipeline.core.client.storage.pg.runtime import get_postgres_client, shutdown_postgres_client
from video_pipeline.core.storage.pg_tracker import ArtifactPersistentVisitor
from video_pipeline.task.base.base_task import BaseTask
from video_pipeline.task.qdrant_indexing.config import (
    IMAGE_QDRANT_INDEXING_CONFIG,
    IMAGE_DENSE_FIELD,
    build_image_collection_name,
    get_image_index_configs,
)
from video_pipeline.task.qdrant_indexing.utils import (
    load_npy_from_minio,
    make_qdrant_client,
    create_summary_artifact,
)


@StageRegistry.register
class ImageQdrantIndexingTask(BaseTask[list[ImageEmbeddingArtifact], list[str]]):
    """Index image embeddings into Qdrant as dense vectors."""

    config = IMAGE_QDRANT_INDEXING_CONFIG

    async def preprocess(self, input_data: list[ImageEmbeddingArtifact]):
        logger = get_run_logger()
        logger.info(f"[ImageQdrantIndexingTask] Downloading {len(input_data)} embedding(s) from MinIO")

        preprocessed = []
        for artifact in input_data:
            assert artifact.object_name, f"ImageEmbeddingArtifact {artifact.artifact_id} has no object_name"
            vector = load_npy_from_minio(self.minio_client, artifact.user_id, artifact.object_name)
            preprocessed.append((artifact, vector))

        logger.info(f"[ImageQdrantIndexingTask] Preprocessing done — {len(preprocessed)} vector(s) ready")
        return preprocessed

    async def execute(self, preprocessed, client):
        logger = get_run_logger()
        logger.info(f"[ImageQdrantIndexingTask] Indexing {len(preprocessed)} image embedding(s)")

        index_configs, field_names = get_image_index_configs()
        await client.create_collection_if_not_exists(index_configs, field_names)

        data = [
            {
                "id": artifact.artifact_id,
                IMAGE_DENSE_FIELD: vector,
                # payload
                "frame_index": artifact.frame_index,
                "timestamp": artifact.timestamp,
                "timestamp_sec": artifact.timestamp_sec,
                "related_video_id": artifact.related_video_id,
                "related_video_fps": artifact.related_video_fps,
                "image_minio_url": artifact.image_minio_url,
                "user_id": artifact.user_id,
                "minio_url": artifact.minio_url_path,
                "caption_text": artifact.caption_text,
            }
            for artifact, vector in preprocessed
        ]

        inserted_ids = await client.insert_vectors(data)
        logger.info(f"[ImageQdrantIndexingTask] Indexed {len(inserted_ids)} point(s)")
        return inserted_ids

    async def postprocess(self, result: list[str]) -> list[str]:
        return result

    @staticmethod
    async def summary_artifact(final_result: list[str]) -> None:
        await create_summary_artifact(
            task_name="Image Qdrant Indexing",
            collection_name=build_image_collection_name(),
            points_count=len(final_result),
        )


@task(**{**IMAGE_QDRANT_INDEXING_CONFIG.to_task_kwargs(), "name": "Image Qdrant Indexing Chunk"}) #type:ignore
async def image_qdrant_indexing_chunk_task(items: list[ImageEmbeddingArtifact]) -> list[str]:
    """Index a batch of image embeddings into Qdrant as dense vectors."""
    from video_pipeline.config import get_settings

    logger = get_run_logger()
    settings = get_settings()
    logger.info(f"[ImageQdrantIndexingChunk] Starting | {len(items)} artifact(s) in batch")

    minio_client = MinioStorageClient(
        endpoint=settings.minio.endpoint,
        access_key=settings.minio.access_key,
        secret_key=settings.minio.secret_key,
        secure=settings.minio.secure,
    )
    postgres_client = await get_postgres_client()
    qdrant_client = make_qdrant_client(build_image_collection_name())
    await qdrant_client.connect()

    task_impl = ImageQdrantIndexingTask(
        artifact_visitor=ArtifactPersistentVisitor(minio_client, postgres_client),
        minio_client=minio_client,
    )

    try:
        indexed_ids = await task_impl.execute_template(items, qdrant_client)
    finally:
        await qdrant_client.close()
        await shutdown_postgres_client(postgres_client)

    logger.info(f"[ImageQdrantIndexingChunk] Done | {len(indexed_ids)} point(s) indexed")
    return indexed_ids