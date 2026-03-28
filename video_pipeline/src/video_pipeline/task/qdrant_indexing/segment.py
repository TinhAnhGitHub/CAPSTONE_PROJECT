"""Segment Qdrant indexing task."""

from __future__ import annotations

from prefect import get_run_logger, task

from video_pipeline.core.artifact import SegmentEmbeddingArtifact
from video_pipeline.core.client.progress import StageRegistry
from video_pipeline.core.client.storage.minio import MinioStorageClient
from video_pipeline.core.client.storage.pg.runtime import get_postgres_client, shutdown_postgres_client
from video_pipeline.core.storage.pg_tracker import ArtifactPersistentVisitor
from video_pipeline.task.base.base_task import BaseTask
from video_pipeline.task.qdrant_indexing.config import (
    SEGMENT_QDRANT_INDEXING_CONFIG,
    SEGMENT_DENSE_FIELD,
    build_segment_collection_name,
    get_segment_index_configs,
)
from video_pipeline.task.qdrant_indexing.utils import (
    load_npy_from_minio,
    make_qdrant_client,
    create_summary_artifact,
)


SegmentPreprocessed = tuple[SegmentEmbeddingArtifact, list[float]]


@StageRegistry.register
class SegmentQdrantIndexingTask(BaseTask[list[SegmentEmbeddingArtifact], list[str]]):
    """Index segment embeddings into Qdrant as dense vectors."""

    config = SEGMENT_QDRANT_INDEXING_CONFIG

    async def preprocess(self, input_data: list[SegmentEmbeddingArtifact]) -> list[SegmentPreprocessed]:
        logger = get_run_logger()
        logger.info(f"[SegmentQdrantIndexingTask] Downloading {len(input_data)} segment embedding(s) from MinIO")

        preprocessed = []
        for artifact in input_data:
            assert artifact.object_name, f"SegmentEmbeddingArtifact {artifact.artifact_id} has no object_name"
            vector = load_npy_from_minio(self.minio_client, artifact.user_id, artifact.object_name)
            preprocessed.append((artifact, vector))

        logger.info(f"[SegmentQdrantIndexingTask] Preprocessing done — {len(preprocessed)} vector(s) ready")
        return preprocessed

    async def execute(self, preprocessed: list[SegmentPreprocessed], client) -> list[str]:
        logger = get_run_logger()
        logger.info(f"[SegmentQdrantIndexingTask] Indexing {len(preprocessed)} segment embedding(s)")

        index_configs, field_names = get_segment_index_configs()
        await client.create_collection_if_not_exists(index_configs, field_names)

        data = [
            {
                "id": artifact.artifact_id,
                SEGMENT_DENSE_FIELD: vector,
                # payload
                "related_audio_segment_artifact_id": artifact.related_audio_segment_artifact_id,
                "related_video_id": artifact.related_video_id,
                "start_frame": artifact.start_frame,
                "end_frame": artifact.end_frame,
                "start_timestamp": artifact.start_timestamp,
                "end_timestamp": artifact.end_timestamp,
                "start_sec": artifact.start_sec,
                "end_sec": artifact.end_sec,
                "frame_indices": artifact.frame_indices,
                "embedding_dim": artifact.embedding_dim,
                "user_id": artifact.user_id,
                "minio_url": artifact.minio_url_path,
                "caption_text": artifact.caption_text,
            }
            for artifact, vector in preprocessed
        ]

        inserted_ids = await client.insert_vectors(data)
        logger.info(f"[SegmentQdrantIndexingTask] Indexed {len(inserted_ids)} point(s)")
        return inserted_ids

    async def postprocess(self, result: list[str]) -> list[str]:
        return result

    @staticmethod
    async def summary_artifact(final_result: list[str]) -> None:
        await create_summary_artifact(
            task_name="Segment Qdrant Indexing",
            collection_name=build_segment_collection_name(),
            points_count=len(final_result),
        )


@task(**{**SEGMENT_QDRANT_INDEXING_CONFIG.to_task_kwargs(), "name": "Segment Qdrant Indexing Chunk"}) #type:ignore
async def segment_qdrant_indexing_chunk_task(items: list[SegmentEmbeddingArtifact]) -> list[str]:
    """Index a batch of segment embeddings into Qdrant."""
    from video_pipeline.config import get_settings

    logger = get_run_logger()
    settings = get_settings()
    logger.info(f"[SegmentQdrantIndexingChunk] Starting | {len(items)} artifact(s) in batch")

    minio_client = MinioStorageClient(
        endpoint=settings.minio.endpoint,
        access_key=settings.minio.access_key,
        secret_key=settings.minio.secret_key,
        secure=settings.minio.secure,
    )
    postgres_client = await get_postgres_client()
    qdrant_client = make_qdrant_client(build_segment_collection_name())
    await qdrant_client.connect()

    task_impl = SegmentQdrantIndexingTask(
        artifact_visitor=ArtifactPersistentVisitor(minio_client, postgres_client),
        minio_client=minio_client,
    )

    try:
        indexed_ids = await task_impl.execute_template(items, qdrant_client)
    finally:
        await qdrant_client.close()
        await shutdown_postgres_client(postgres_client)

    logger.info(f"[SegmentQdrantIndexingChunk] Done | {len(indexed_ids)} point(s) indexed")
    return indexed_ids