from __future__ import annotations

import io
import re
import numpy as np
from typing import Any

from fastembed import SparseTextEmbedding
from qdrant_client.models import Distance, SparseVector

from prefect import get_run_logger, task
from prefect.artifacts import acreate_markdown_artifact, acreate_table_artifact

from video_pipeline.task.base.base_task import TaskConfig, BaseTask
from video_pipeline.core.client.progress import StageRegistry
from video_pipeline.core.artifact import (
    ImageEmbeddingArtifact,
    TextCaptionEmbeddingArtifact,
    ImageCaptionMultimodalEmbeddingArtifact,
)
from video_pipeline.core.storage.pg_tracker import ArtifactPersistentVisitor
from video_pipeline.core.client.storage.minio import MinioStorageClient
from video_pipeline.core.client.storage.pg import PostgresClient, PgConfig
from video_pipeline.core.client.storage.qdrant.client import QdrantStorageClient
from video_pipeline.core.client.storage.qdrant.config import QdrantConfig, QdrantIndexConfig
from video_pipeline.config import get_settings


QDRANT_INDEXING_CONFIG = TaskConfig.from_yaml("qdrant_indexing")
_base_kwargs = QDRANT_INDEXING_CONFIG.to_task_kwargs()
_akwargs: dict[str, Any] = QDRANT_INDEXING_CONFIG.additional_kwargs

# Qdrant vector field names
IMAGE_DENSE_FIELD = "image_dense"
CAPTION_TEXT_DENSE_FIELD = "caption_text_dense"
CAPTION_MM_DENSE_FIELD = "caption_mm_dense"
CAPTION_SPARSE_FIELD = "caption_sparse"

# Types
_ImagePreprocessed = tuple[ImageEmbeddingArtifact, list[float]]
_CaptionPair = tuple[TextCaptionEmbeddingArtifact, ImageCaptionMultimodalEmbeddingArtifact]
_CaptionPreprocessed = tuple[
    TextCaptionEmbeddingArtifact,
    ImageCaptionMultimodalEmbeddingArtifact,
    list[float],   # text dense vector
    list[float],   # multimodal dense vector
    SparseVector,  # SPLADE sparse vector
]


def _load_npy_from_minio(minio_client: MinioStorageClient, user_id: str, object_name: str) -> list[float]:
    """Download a .npy embedding file from MinIO and return as a Python list of floats."""
    data = minio_client.get_object_bytes(bucket=user_id, object_name=object_name)
    arr = np.load(io.BytesIO(data))
    return arr.flatten().tolist()


def _build_image_collection_name() -> str:
    return f"{_akwargs.get('collection_base', 'video_embeddings')}_image"


def _build_caption_collection_name() -> str:
    return f"{_akwargs.get('collection_base', 'video_embeddings')}_caption"


def _image_index_configs() -> tuple[list[QdrantIndexConfig], list[str]]:
    on_disk: bool = _akwargs.get("on_disk", False)
    image_dim: int = _akwargs.get("image_dim", 1536)
    cfg = QdrantIndexConfig(
        vector_size=image_dim,
        distance=Distance.COSINE,
        on_disk=on_disk,
    )
    return [cfg], [IMAGE_DENSE_FIELD]


def _caption_index_configs() -> tuple[list[QdrantIndexConfig], list[str]]:
    on_disk: bool = _akwargs.get("on_disk", False)
    caption_dim: int = _akwargs.get("caption_dim", 768)
    mm_dim: int = _akwargs.get("mm_dim", 1536)

    text_dense_cfg = QdrantIndexConfig(
        vector_size=caption_dim,
        distance=Distance.COSINE,
        on_disk=on_disk,
    )
    mm_dense_cfg = QdrantIndexConfig(
        vector_size=mm_dim,
        distance=Distance.COSINE,
        on_disk=on_disk,
    )
    sparse_cfg = QdrantIndexConfig(
        vector_size=0,  # unused for sparse
        distance=Distance.COSINE,
        on_disk=on_disk,
        is_sparse=True,
    )
    return (
        [text_dense_cfg, mm_dense_cfg, sparse_cfg],
        [CAPTION_TEXT_DENSE_FIELD, CAPTION_MM_DENSE_FIELD, CAPTION_SPARSE_FIELD],
    )


# ---------------------------------------------------------------------------
# Image indexing (dense only)
# ---------------------------------------------------------------------------

@StageRegistry.register
class ImageQdrantIndexingTask(BaseTask[list[ImageEmbeddingArtifact], list[str]]):
    """Index image embeddings into Qdrant as dense vectors.

    preprocess() downloads the .npy embedding file for each artifact from MinIO.
    execute()    creates the image collection (if needed) and upserts all points.
    postprocess() returns the list of upserted point IDs.
    """

    config = QDRANT_INDEXING_CONFIG

    async def preprocess(self, input_data: list[ImageEmbeddingArtifact]) -> list[_ImagePreprocessed]:
        logger = get_run_logger()
        logger.info(f"[ImageQdrantIndexingTask] Downloading {len(input_data)} embedding(s) from MinIO")

        preprocessed: list[_ImagePreprocessed] = []
        for artifact in input_data:
            assert artifact.object_name, f"ImageEmbeddingArtifact {artifact.artifact_id} has no object_name"
            vector = _load_npy_from_minio(self.minio_client, artifact.user_id, artifact.object_name)
            preprocessed.append((artifact, vector))

        logger.info(f"[ImageQdrantIndexingTask] Preprocessing done — {len(preprocessed)} vector(s) ready")
        return preprocessed

    async def execute(
        self,
        preprocessed: list[_ImagePreprocessed],
        client: QdrantStorageClient,
    ) -> list[str]:
        logger = get_run_logger()
        logger.info(f"[ImageQdrantIndexingTask] Indexing {len(preprocessed)} image embedding(s)")

        index_configs, field_names = _image_index_configs()
        await client.create_collection_if_not_exists(index_configs, field_names)

        data = [
            {
                "id": artifact.artifact_id,
                IMAGE_DENSE_FIELD: vector,
                # payload
                "frame_index": artifact.frame_index,
                "timestamp": artifact.time_stamp,
                "related_video_id": artifact.related_video_id,
                "related_video_fps": artifact.related_video_fps,
                "image_minio_url": artifact.image_minio_url,
                "user_id": artifact.user_id,
                "minio_url": artifact.minio_url_path,
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
        if not final_result:
            return
        key = re.sub(r"[^a-z0-9-]", "-", "image-qdrant-indexing")
        markdown = (
            f"# Image Qdrant Indexing Summary\n\n"
            f"| Field | Value |\n"
            f"|-------|-------|\n"
            f"| **Collection** | `{_build_image_collection_name()}` |\n"
            f"| **Points Indexed** | `{len(final_result)}` |\n"
        )
        await acreate_markdown_artifact(
            key=key,
            markdown=markdown,
            description="Image dense index summary",
        )
        await acreate_table_artifact(
            table=[{"Field": "Points Indexed", "Value": str(len(final_result))}],
            key=f"{key}-table",
            description="Image Qdrant indexing stats",
        )


# ---------------------------------------------------------------------------
# Caption indexing (hybrid: text dense + multimodal dense + sparse)
# ---------------------------------------------------------------------------

@StageRegistry.register
class CaptionQdrantIndexingTask(BaseTask[list[_CaptionPair], list[str]]):
    """Index caption embeddings into Qdrant as hybrid vectors.

    Each point carries:
      - caption_text_dense   : dense mmBERT text embedding
      - caption_mm_dense     : dense QwenVL multimodal embedding
      - caption_sparse       : sparse SPLADE encoding of the raw caption text

    preprocess() downloads both .npy files per pair and encodes sparse vectors
                 in batch using fastembed SPLADE.
    execute()    creates the caption collection and upserts hybrid points.
    postprocess() returns upserted point IDs.
    """

    config = QDRANT_INDEXING_CONFIG

    async def preprocess(self, input_data: list[_CaptionPair]) -> list[_CaptionPreprocessed]:
        logger = get_run_logger()
        logger.info(f"[CaptionQdrantIndexingTask] Preprocessing {len(input_data)} pair(s)")

        # Download dense vectors
        pairs_with_vectors: list[tuple[
            TextCaptionEmbeddingArtifact,
            ImageCaptionMultimodalEmbeddingArtifact,
            list[float],
            list[float],
            str,  # raw caption text for sparse encoding
        ]] = []

        for text_artifact, mm_artifact in input_data:
            assert text_artifact.object_name, (
                f"TextCaptionEmbeddingArtifact {text_artifact.artifact_id} has no object_name"
            )
            assert mm_artifact.object_name, (
                f"ImageCaptionMultimodalEmbeddingArtifact {mm_artifact.artifact_id} has no object_name"
            )

            text_vec = _load_npy_from_minio(self.minio_client, text_artifact.user_id, text_artifact.object_name)
            mm_vec = _load_npy_from_minio(self.minio_client, mm_artifact.user_id, mm_artifact.object_name)

            caption_text: str = (text_artifact.metadata or {}).get("caption", "")
            pairs_with_vectors.append((text_artifact, mm_artifact, text_vec, mm_vec, caption_text))

        # Batch-encode sparse vectors with SPLADE
        logger.info(f"[CaptionQdrantIndexingTask] Encoding {len(pairs_with_vectors)} sparse vector(s) via SPLADE")
        texts = [caption_text for *_, caption_text in pairs_with_vectors]
        sparse_model = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")
        sparse_embeddings = list(sparse_model.embed(texts))

        preprocessed: list[_CaptionPreprocessed] = []
        for (text_artifact, mm_artifact, text_vec, mm_vec, _), sparse_emb in zip(
            pairs_with_vectors, sparse_embeddings
        ):
            sparse_vector = SparseVector(
                indices=sparse_emb.indices.tolist(),
                values=sparse_emb.values.tolist(),
            )
            preprocessed.append((text_artifact, mm_artifact, text_vec, mm_vec, sparse_vector))

        logger.info(f"[CaptionQdrantIndexingTask] Preprocessing done — {len(preprocessed)} item(s) ready")
        return preprocessed

    async def execute(
        self,
        preprocessed: list[_CaptionPreprocessed],
        client: QdrantStorageClient,
    ) -> list[str]:
        logger = get_run_logger()
        logger.info(f"[CaptionQdrantIndexingTask] Indexing {len(preprocessed)} caption hybrid point(s)")

        index_configs, field_names = _caption_index_configs()
        await client.create_collection_if_not_exists(index_configs, field_names)

        data = [
            {
                "id": text_artifact.artifact_id,
                CAPTION_TEXT_DENSE_FIELD: text_vec,
                CAPTION_MM_DENSE_FIELD: mm_vec,
                CAPTION_SPARSE_FIELD: sparse_vec,
                # payload
                "frame_index": text_artifact.frame_index,
                "timestamp": text_artifact.time_stamp,
                "related_video_id": text_artifact.related_video_id,
                "related_frame_fps": text_artifact.related_frame_fps,
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
        if not final_result:
            return
        key = re.sub(r"[^a-z0-9-]", "-", "caption-qdrant-indexing")
        markdown = (
            f"# Caption Qdrant Indexing Summary\n\n"
            f"| Field | Value |\n"
            f"|-------|-------|\n"
            f"| **Collection** | `{_build_caption_collection_name()}` |\n"
            f"| **Index type** | Hybrid (text dense + multimodal dense + SPLADE sparse) |\n"
            f"| **Points Indexed** | `{len(final_result)}` |\n"
        )
        await acreate_markdown_artifact(
            key=key,
            markdown=markdown,
            description="Caption hybrid index summary",
        )
        await acreate_table_artifact(
            table=[
                {"Field": "Collection", "Value": _build_caption_collection_name()},
                {"Field": "Points Indexed", "Value": str(len(final_result))},
            ],
            key=f"{key}-table",
            description="Caption Qdrant indexing stats",
        )


# ---------------------------------------------------------------------------
# Prefect task functions (fan-out compatible via .map())
# ---------------------------------------------------------------------------

def _make_qdrant_client(collection_name: str) -> QdrantStorageClient:
    settings = get_settings()
    timeout: int = _akwargs.get("qdrant_timeout", 30)
    config = QdrantConfig(
        host=settings.qdrant.host,
        port=settings.qdrant.port,
        collection_name=collection_name,
        timeout=timeout,
        use_grpc=True,
        prefer_grpc=True,
    )
    return QdrantStorageClient(config=config)


@task(**{**_base_kwargs, "name": "Image Qdrant Indexing Chunk"})  # type: ignore
async def image_qdrant_indexing_chunk_task(
    items: list[ImageEmbeddingArtifact],
) -> list[str]:
    """Index a batch of image embeddings into Qdrant as dense vectors.

    Args:
        items: Batch of ImageEmbeddingArtifacts to index.

    Returns:
        List of Qdrant point IDs that were upserted.
    """
    logger = get_run_logger()
    settings = get_settings()
    logger.info(f"[ImageQdrantIndexingChunk] Starting | {len(items)} artifact(s) in batch")

    minio_client = MinioStorageClient(
        endpoint=settings.minio.endpoint,
        access_key=settings.minio.access_key,
        secret_key=settings.minio.secret_key,
        secure=settings.minio.secure,
    )
    postgres_client = PostgresClient(
        config=PgConfig(database_url=settings.postgres.connection_string)  # type: ignore
    )

    qdrant_client = _make_qdrant_client(_build_image_collection_name())
    await qdrant_client.connect()

    task_impl = ImageQdrantIndexingTask(
        artifact_visitor=ArtifactPersistentVisitor(minio_client, postgres_client),
        minio_client=minio_client,
    )

    try:
        indexed_ids = await task_impl.execute_template(items, qdrant_client)
    finally:
        await qdrant_client.close()

    logger.info(f"[ImageQdrantIndexingChunk] Done | {len(indexed_ids)} point(s) indexed")
    return indexed_ids


@task(**{**_base_kwargs, "name": "Caption Qdrant Indexing Chunk"})  # type: ignore
async def caption_qdrant_indexing_chunk_task(
    text_items: list[TextCaptionEmbeddingArtifact],
    mm_items: list[ImageCaptionMultimodalEmbeddingArtifact],
) -> list[str]:
    """Index a batch of caption embeddings into Qdrant as hybrid (dense + sparse) vectors.

    Pairs text embedding artifacts with multimodal embedding artifacts by position.
    The two lists must be aligned (same frame order).

    Args:
        text_items: Batch of TextCaptionEmbeddingArtifacts (mmBERT dense).
        mm_items:   Batch of ImageCaptionMultimodalEmbeddingArtifacts (QwenVL dense).

    Returns:
        List of Qdrant point IDs that were upserted.
    """
    logger = get_run_logger()
    settings = get_settings()
    logger.info(
        f"[CaptionQdrantIndexingChunk] Starting | "
        f"{len(text_items)} text + {len(mm_items)} multimodal artifact(s)"
    )

    assert len(text_items) == len(mm_items), (
        f"text_items ({len(text_items)}) and mm_items ({len(mm_items)}) must have the same length"
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

    qdrant_client = _make_qdrant_client(_build_caption_collection_name())
    await qdrant_client.connect()

    pairs: list[_CaptionPair] = list(zip(text_items, mm_items))

    task_impl = CaptionQdrantIndexingTask(
        artifact_visitor=ArtifactPersistentVisitor(minio_client, postgres_client),
        minio_client=minio_client,
    )

    try:
        indexed_ids = await task_impl.execute_template(pairs, qdrant_client)
    finally:
        await qdrant_client.close()

    logger.info(f"[CaptionQdrantIndexingChunk] Done | {len(indexed_ids)} hybrid point(s) indexed")
    return indexed_ids
