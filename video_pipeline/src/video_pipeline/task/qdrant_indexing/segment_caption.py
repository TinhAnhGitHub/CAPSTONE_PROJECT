"""Segment Caption Qdrant indexing task."""

from __future__ import annotations

from prefect import get_run_logger, task
from qdrant_client.models import SparseVector

from video_pipeline.core.artifact import TextCapSegmentEmbedArtifact
from video_pipeline.core.client.progress import StageRegistry
from video_pipeline.core.client.storage.minio import MinioStorageClient
from video_pipeline.core.client.storage.pg.runtime import get_postgres_client, shutdown_postgres_client
from video_pipeline.core.storage.pg_tracker import ArtifactPersistentVisitor
from video_pipeline.task.base.base_task import BaseTask
from video_pipeline.task.qdrant_indexing.config import (
    SEGMENT_CAPTION_QDRANT_INDEXING_CONFIG,
    SEGMENT_CAPTION_DENSE_FIELD,
    SEGMENT_CAPTION_SPARSE_FIELD,
    build_segment_caption_collection_name,
    get_segment_caption_index_configs,
)
from video_pipeline.task.qdrant_indexing.utils import (
    load_npy_from_minio,
    make_qdrant_client,
    encode_sparse_vectors,
    create_summary_artifact,
)


SegmentCaptionPreprocessed = tuple[TextCapSegmentEmbedArtifact, list[float], SparseVector, str]


@StageRegistry.register
class SegmentCaptionQdrantIndexingTask(BaseTask[list[TextCapSegmentEmbedArtifact], list[str]]):
    """Index segment caption text embeddings into Qdrant (dense + sparse)."""

    config = SEGMENT_CAPTION_QDRANT_INDEXING_CONFIG

    async def preprocess(self, input_data: list[TextCapSegmentEmbedArtifact]) -> list[SegmentCaptionPreprocessed]:
        logger = get_run_logger()
        logger.info(f"[SegmentCaptionQdrantIndexingTask] Preprocessing {len(input_data)} artifact(s)")

        items_with_vectors = []
        texts_for_sparse = []

        for artifact in input_data:
            assert artifact.object_name, f"TextCapSegmentEmbedArtifact {artifact.artifact_id} has no object_name"
            vector = load_npy_from_minio(self.minio_client, artifact.user_id, artifact.object_name)
            caption_text = (artifact.metadata or {}).get("summary_caption", "")
            items_with_vectors.append((artifact, vector, caption_text))
            texts_for_sparse.append(caption_text)

        logger.info(f"[SegmentCaptionQdrantIndexingTask] Encoding {len(texts_for_sparse)} sparse vector(s)")
        sparse_vectors = encode_sparse_vectors(texts_for_sparse)

        preprocessed = [
            (art, vec, sparse_vec, caption_text)
            for (art, vec, caption_text), sparse_vec in zip(items_with_vectors, sparse_vectors)
        ]

        logger.info(f"[SegmentCaptionQdrantIndexingTask] Preprocessing done — {len(preprocessed)} item(s) ready")
        return preprocessed

    async def execute(self, preprocessed: list[SegmentCaptionPreprocessed], client) -> list[str]:
        logger = get_run_logger()
        logger.info(f"[SegmentCaptionQdrantIndexingTask] Indexing {len(preprocessed)} point(s)")

        index_configs, field_names = get_segment_caption_index_configs()
        await client.create_collection_if_not_exists(index_configs, field_names)

        data = [
            {
                "id": artifact.artifact_id,
                SEGMENT_CAPTION_DENSE_FIELD: dense_vec,
                SEGMENT_CAPTION_SPARSE_FIELD: sparse_vec,
                "related_video_id": artifact.related_video_id,
                "related_video_fps": artifact.related_video_fps,
                "start_frame": artifact.start_frame,
                "end_frame": artifact.end_frame,
                "start_timestamp": artifact.start_timestamp,
                "end_timestamp": artifact.end_timestamp,
                "start_sec": artifact.start_sec,
                "end_sec": artifact.end_sec,
                "segment_caption_minio_url": artifact.related_segment_caption_url,
                "segment_cap_id": artifact.segment_cap_id,
                "user_id": artifact.user_id,
                "minio_url": artifact.minio_url_path,
                "caption_text": caption_text,
            }
            for artifact, dense_vec, sparse_vec, caption_text in preprocessed
        ]

        inserted_ids = await client.insert_vectors(data)
        return inserted_ids

    async def postprocess(self, result: list[str]) -> list[str]:
        return result

    @staticmethod
    async def summary_artifact(final_result: list[str]) -> None:
        await create_summary_artifact(
            task_name="Segment Caption Qdrant Indexing",
            collection_name=build_segment_caption_collection_name(),
            points_count=len(final_result),
            extra_info={"Index type": "Hybrid (dense + sparse)"},
        )


@task(**{**SEGMENT_CAPTION_QDRANT_INDEXING_CONFIG.to_task_kwargs(), "name": "Segment Caption Qdrant Indexing Chunk"}) #type:ignore
async def segment_caption_qdrant_indexing_chunk_task(items: list[TextCapSegmentEmbedArtifact]) -> list[str]:
    from video_pipeline.config import get_settings

    logger = get_run_logger()
    settings = get_settings()
    logger.info(f"[SegmentCaptionQdrantIndexingChunk] Starting | {len(items)} artifact(s)")

    minio_client = MinioStorageClient(
        endpoint=settings.minio.endpoint,
        access_key=settings.minio.access_key,
        secret_key=settings.minio.secret_key,
        secure=settings.minio.secure,
    )
    postgres_client = await get_postgres_client()
    qdrant_client = make_qdrant_client(build_segment_caption_collection_name())
    await qdrant_client.connect()

    task_impl = SegmentCaptionQdrantIndexingTask(
        artifact_visitor=ArtifactPersistentVisitor(minio_client, postgres_client),
        minio_client=minio_client,
    )

    try:
        indexed_ids = await task_impl.execute_template(items, qdrant_client)
    finally:
        await qdrant_client.close()
        await shutdown_postgres_client(postgres_client)

    logger.info(f"[SegmentCaptionQdrantIndexingChunk] Done | {len(indexed_ids)} point(s) indexed")
    return indexed_ids