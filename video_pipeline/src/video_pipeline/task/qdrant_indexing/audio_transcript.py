"""Audio Transcript Qdrant indexing task.

Indexes audio transcript embeddings into Qdrant for semantic search
over spoken content extracted from ASR.
"""

from __future__ import annotations

from prefect import get_run_logger, task

from video_pipeline.core.artifact import AudioTranscriptEmbedArtifact
from video_pipeline.core.client.progress import StageRegistry
from video_pipeline.core.client.storage.minio import MinioStorageClient
from video_pipeline.core.client.storage.pg.runtime import get_postgres_client, shutdown_postgres_client
from video_pipeline.core.storage.pg_tracker import ArtifactPersistentVisitor
from video_pipeline.task.base.base_task import BaseTask
from video_pipeline.task.qdrant_indexing.config import (
    AUDIO_TRANSCRIPT_QDRANT_INDEXING_CONFIG,
    AUDIO_TRANSCRIPT_DENSE_FIELD,
    build_audio_transcript_collection_name,
    get_audio_transcript_index_configs,
)
from video_pipeline.task.qdrant_indexing.utils import (
    load_npy_from_minio,
    make_qdrant_client,
    create_summary_artifact,
)


AudioTranscriptPreprocessed = tuple[AudioTranscriptEmbedArtifact, list[float]]


@StageRegistry.register
class AudioTranscriptQdrantIndexingTask(BaseTask[list[AudioTranscriptEmbedArtifact], list[str]]):
    """Index audio transcript embeddings into Qdrant as dense vectors.

    This task indexes AudioTranscriptEmbedArtifact objects into Qdrant,
    enabling semantic similarity search over spoken content from ASR.

    The embeddings are stored as dense vectors (768-dim from mmBERT),
    with payload metadata for retrieval and filtering.

    Pipeline Position:
        Runs after AudioTranscriptEmbedding.
        Input: AudioTranscriptEmbedArtifact
        Output: List of Qdrant point IDs

    Collection Schema:
        - Vector: audio_transcript_dense (768-dim, COSINE)
        - Payload: segment_index, timestamps, video_id, user_id, audio_text preview
    """

    config = AUDIO_TRANSCRIPT_QDRANT_INDEXING_CONFIG

    async def preprocess(
        self, input_data: list[AudioTranscriptEmbedArtifact]
    ) -> list[AudioTranscriptPreprocessed]:
        """Download embedding vectors from MinIO.

        Args:
            input_data: List of AudioTranscriptEmbedArtifact to index

        Returns:
            List of (AudioTranscriptEmbedArtifact, vector) tuples
        """
        logger = get_run_logger()
        logger.info(
            f"[AudioTranscriptQdrantIndexingTask] Downloading {len(input_data)} embedding(s) from MinIO"
        )

        preprocessed = []
        for artifact in input_data:
            assert artifact.object_name, (
                f"AudioTranscriptEmbedArtifact {artifact.artifact_id} has no object_name"
            )
            vector = load_npy_from_minio(self.minio_client, artifact.user_id, artifact.object_name)
            preprocessed.append((artifact, vector))

        logger.info(
            f"[AudioTranscriptQdrantIndexingTask] Preprocessing done — {len(preprocessed)} vector(s) ready"
        )
        return preprocessed

    async def execute(
        self, preprocessed: list[AudioTranscriptPreprocessed], client
    ) -> list[str]:
        """Index embeddings into Qdrant.

        Args:
            preprocessed: List of (artifact, vector) tuples
            client: QdrantStorageClient instance

        Returns:
            List of inserted Qdrant point IDs
        """
        logger = get_run_logger()
        logger.info(
            f"[AudioTranscriptQdrantIndexingTask] Indexing {len(preprocessed)} embedding(s)"
        )

        index_configs, field_names = get_audio_transcript_index_configs()
        await client.create_collection_if_not_exists(index_configs, field_names)

        data = [
            {
                "id": artifact.artifact_id,
                AUDIO_TRANSCRIPT_DENSE_FIELD: vector,
                # payload
                "related_audio_segment_artifact_id": artifact.related_audio_segment_artifact_id,
                "related_video_id": artifact.related_video_id,
                "segment_index": artifact.segment_index,
                "start_frame": artifact.start_frame,
                "end_frame": artifact.end_frame,
                "start_timestamp": artifact.start_timestamp,
                "end_timestamp": artifact.end_timestamp,
                "start_sec": artifact.start_sec,
                "end_sec": artifact.end_sec,
                "embedding_dim": artifact.embedding_dim,
                "user_id": artifact.user_id,
                "minio_url": artifact.minio_url_path,
                "audio_text": artifact.audio_text,
            }
            for artifact, vector in preprocessed
        ]

        inserted_ids = await client.insert_vectors(data)
        logger.info(f"[AudioTranscriptQdrantIndexingTask] Indexed {len(inserted_ids)} point(s)")
        return inserted_ids

    async def postprocess(self, result: list[str]) -> list[str]:
        """Return the inserted IDs directly."""
        return result

    @staticmethod
    async def summary_artifact(final_result: list[str]) -> None:
        """Create a Prefect artifact summarizing the indexing operation.

        Args:
            final_result: List of inserted Qdrant point IDs
        """
        await create_summary_artifact(
            task_name="Audio Transcript Qdrant Indexing",
            collection_name=build_audio_transcript_collection_name(),
            points_count=len(final_result),
            extra_info={"Embedding Type": "Dense (mmBERT 768-dim)", "Source": "ASR Transcript"},
        )


@task(
    **{**AUDIO_TRANSCRIPT_QDRANT_INDEXING_CONFIG.to_task_kwargs(), "name": "Audio Transcript Qdrant Indexing Chunk"}  # type: ignore
)
async def audio_transcript_qdrant_indexing_chunk_task(
    items: list[AudioTranscriptEmbedArtifact],
) -> list[str]:
    """Index a batch of audio transcript embeddings into Qdrant.

    Args:
        items: Batch of AudioTranscriptEmbedArtifact from audio_transcript_embedding task

    Returns:
        List of Qdrant point IDs that were inserted
    """
    from video_pipeline.config import get_settings

    logger = get_run_logger()
    settings = get_settings()
    logger.info(
        f"[AudioTranscriptQdrantIndexingChunk] Starting | {len(items)} artifact(s) in batch"
    )

    minio_client = MinioStorageClient(
        endpoint=settings.minio.endpoint,
        access_key=settings.minio.access_key,
        secret_key=settings.minio.secret_key,
        secure=settings.minio.secure,
    )
    postgres_client = await get_postgres_client()
    qdrant_client = make_qdrant_client(build_audio_transcript_collection_name())
    await qdrant_client.connect()

    task_impl = AudioTranscriptQdrantIndexingTask(
        artifact_visitor=ArtifactPersistentVisitor(minio_client, postgres_client),
        minio_client=minio_client,
    )

    try:
        indexed_ids = await task_impl.execute_template(items, qdrant_client)
    finally:
        await qdrant_client.close()
        await shutdown_postgres_client(postgres_client)

    logger.info(
        f"[AudioTranscriptQdrantIndexingChunk] Done | {len(indexed_ids)} point(s) indexed"
    )
    return indexed_ids