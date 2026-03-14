"""Caption Qdrant indexing task."""

from __future__ import annotations

from prefect import get_run_logger, task

from qdrant_client.models import SparseVector

from video_pipeline.core.artifact import TextCaptionEmbeddingArtifact, ImageCaptionMultimodalEmbeddingArtifact
from video_pipeline.core.client.progress import StageRegistry
from video_pipeline.core.client.storage.minio import MinioStorageClient
from video_pipeline.core.client.storage.pg.runtime import get_postgres_client, shutdown_postgres_client
from video_pipeline.core.storage.pg_tracker import ArtifactPersistentVisitor
from video_pipeline.task.base.base_task import BaseTask
from video_pipeline.task.qdrant_indexing.config import (
    CAPTION_QDRANT_INDEXING_CONFIG,
    CAPTION_TEXT_DENSE_FIELD,
    CAPTION_MM_DENSE_FIELD,
    CAPTION_SPARSE_FIELD,
    build_caption_collection_name,
    get_caption_index_configs,
)
from video_pipeline.task.qdrant_indexing.utils import (
    load_npy_from_minio,
    make_qdrant_client,
    encode_sparse_vectors,
    create_summary_artifact,
)


CaptionPair = tuple[TextCaptionEmbeddingArtifact, ImageCaptionMultimodalEmbeddingArtifact]
CaptionPreprocessed = tuple[
    TextCaptionEmbeddingArtifact,
    ImageCaptionMultimodalEmbeddingArtifact,
    list[float],  # text dense vector
    list[float],  # multimodal dense vector
    SparseVector,  # SPLADE sparse vector
]


@StageRegistry.register
class CaptionQdrantIndexingTask(BaseTask[list[CaptionPair], list[str]]):
    """Index caption embeddings into Qdrant as hybrid vectors.

    Each point carries:
      - caption_text_dense: dense mmBERT text embedding
      - caption_mm_dense: dense QwenVL multimodal embedding
      - caption_sparse: sparse SPLADE encoding of the raw caption text
    """

    config = CAPTION_QDRANT_INDEXING_CONFIG

    async def preprocess(self, input_data: list[CaptionPair]) -> list[CaptionPreprocessed]:
        logger = get_run_logger()
        logger.info(f"[CaptionQdrantIndexingTask] Preprocessing {len(input_data)} pair(s)")

        pairs_with_vectors = []
        texts_for_sparse = []

        for text_artifact, mm_artifact in input_data:
            assert text_artifact.object_name, f"TextCaptionEmbeddingArtifact {text_artifact.artifact_id} has no object_name"
            assert mm_artifact.object_name, f"ImageCaptionMultimodalEmbeddingArtifact {mm_artifact.artifact_id} has no object_name"

            text_vec = load_npy_from_minio(self.minio_client, text_artifact.user_id, text_artifact.object_name)
            mm_vec = load_npy_from_minio(self.minio_client, mm_artifact.user_id, mm_artifact.object_name)

            caption_text = (text_artifact.metadata or {}).get("caption", "")
            pairs_with_vectors.append((text_artifact, mm_artifact, text_vec, mm_vec, caption_text))
            texts_for_sparse.append(caption_text)

        # Batch-encode sparse vectors with SPLADE
        logger.info(f"[CaptionQdrantIndexingTask] Encoding {len(texts_for_sparse)} sparse vector(s) via SPLADE")
        sparse_vectors = encode_sparse_vectors(texts_for_sparse)

        preprocessed = [
            (text_art, mm_art, text_vec, mm_vec, sparse_vec)
            for (text_art, mm_art, text_vec, mm_vec, _), sparse_vec in zip(pairs_with_vectors, sparse_vectors)
        ]

        logger.info(f"[CaptionQdrantIndexingTask] Preprocessing done — {len(preprocessed)} item(s) ready")
        return preprocessed

    async def execute(self, preprocessed: list[CaptionPreprocessed], client) -> list[str]:
        logger = get_run_logger()
        logger.info(f"[CaptionQdrantIndexingTask] Indexing {len(preprocessed)} caption hybrid point(s)")

        index_configs, field_names = get_caption_index_configs()
        await client.create_collection_if_not_exists(index_configs, field_names)

        data = [
            {
                "id": text_artifact.artifact_id,
                CAPTION_TEXT_DENSE_FIELD: text_vec,
                CAPTION_MM_DENSE_FIELD: mm_vec,
                CAPTION_SPARSE_FIELD: sparse_vec,
                # payload
                "frame_index": text_artifact.frame_index,
                "timestamp": text_artifact.timestamp,
                "timestamp_sec": text_artifact.timestamp_sec,
                "related_video_id": text_artifact.related_video_id,
                "related_video_fps": text_artifact.related_video_fps,
                "caption_minio_url": text_artifact.image_caption_minio_url,
                "image_minio_url": text_artifact.image_minio_url,
                "user_id": text_artifact.user_id,
                "caption_id": text_artifact.caption_id,
                "image_id": text_artifact.image_id,
                "mm_embedding_minio_url": mm_artifact.minio_url_path,
            }
            for text_artifact, mm_artifact, text_vec, mm_vec, sparse_vec in preprocessed
        ]

        inserted_ids = await client.insert_vectors(data)
        logger.info(f"[CaptionQdrantIndexingTask] Indexed {len(inserted_ids)} hybrid point(s)")
        return inserted_ids

    async def postprocess(self, result: list[str]) -> list[str]:
        return result

    @staticmethod
    async def summary_artifact(final_result: list[str]) -> None:
        await create_summary_artifact(
            task_name="Caption Qdrant Indexing",
            collection_name=build_caption_collection_name(),
            points_count=len(final_result),
            extra_info={"Index type": "Hybrid (dense + sparse)"},
        )


@task(**{**CAPTION_QDRANT_INDEXING_CONFIG.to_task_kwargs(), "name": "Caption Qdrant Indexing Chunk"}) #type:ignore
async def caption_qdrant_indexing_chunk_task(
    text_items: list[TextCaptionEmbeddingArtifact],
    mm_items: list[ImageCaptionMultimodalEmbeddingArtifact],
) -> list[str]:
    """Index a batch of caption embeddings into Qdrant as hybrid vectors.

    Pairs text embedding artifacts with multimodal embedding artifacts by caption_id.
    """
    from video_pipeline.config import get_settings

    logger = get_run_logger()
    settings = get_settings()
    logger.info(f"[CaptionQdrantIndexingChunk] Starting | {len(text_items)} text + {len(mm_items)} multimodal artifact(s)")

    assert len(text_items) == len(mm_items), f"text_items ({len(text_items)}) and mm_items ({len(mm_items)}) must have the same length"

    minio_client = MinioStorageClient(
        endpoint=settings.minio.endpoint,
        access_key=settings.minio.access_key,
        secret_key=settings.minio.secret_key,
        secure=settings.minio.secure,
    )
    postgres_client = await get_postgres_client()
    qdrant_client = make_qdrant_client(build_caption_collection_name())
    await qdrant_client.connect()

    # Pair by caption_id
    text_by_caption_id = {a.caption_id: a for a in text_items}
    pairs = [
        (text_by_caption_id[mm_artifact.caption_id], mm_artifact)
        for mm_artifact in mm_items
        if mm_artifact.caption_id in text_by_caption_id
    ]

    task_impl = CaptionQdrantIndexingTask(
        artifact_visitor=ArtifactPersistentVisitor(minio_client, postgres_client),
        minio_client=minio_client,
    )

    try:
        indexed_ids = await task_impl.execute_template(pairs, qdrant_client)
    finally:
        await qdrant_client.close()
        await shutdown_postgres_client(postgres_client)

    logger.info(f"[CaptionQdrantIndexingChunk] Done | {len(indexed_ids)} hybrid point(s) indexed")
    return indexed_ids