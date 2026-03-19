from __future__ import annotations

import io

import numpy as np
from prefect import get_run_logger, task

from video_pipeline.task.base.base_task import TaskConfig, BaseTask
from video_pipeline.core.client.progress import StageRegistry
from video_pipeline.core.artifact import SegmentCaptionArtifact, TextCapSegmentEmbedArtifact
from video_pipeline.core.storage.pg_tracker import ArtifactPersistentVisitor
from video_pipeline.core.client.storage.minio import MinioStorageClient
from video_pipeline.core.client.storage.pg.runtime import (
    get_postgres_client,
    shutdown_postgres_client,
)
from video_pipeline.core.client.inference.te_client import MMBertClient, MMBertConfig
from video_pipeline.config import get_settings


SEGMENT_CAPTION_EMBEDDING_CONFIG = TaskConfig.from_yaml("segment_caption_embedding")
_base_kwargs = SEGMENT_CAPTION_EMBEDDING_CONFIG.to_task_kwargs()

_PreprocessedItem = tuple[SegmentCaptionArtifact, str]


@StageRegistry.register
class SegmentCaptionEmbeddingTask(
    BaseTask[list[SegmentCaptionArtifact], list[TextCapSegmentEmbedArtifact]]
):
    """Embed segment caption texts using mmBERT in batch.

    preprocess() extracts the summary_caption text from each artifact.
    execute_single() sends the whole batch to mmBERT in a single request.
    postprocess() uploads .npy embedding files to MinIO and persists to Postgres.
    """

    config = SEGMENT_CAPTION_EMBEDDING_CONFIG

    async def preprocess(self, input_data: list[SegmentCaptionArtifact]) -> list[_PreprocessedItem]:
        """Extract caption text from each artifact's summary_caption."""
        logger = get_run_logger()
        logger.info(f"[SegmentCaptionEmbeddingTask] Preparing {len(input_data)} segment caption(s)")

        preprocessed: list[_PreprocessedItem] = []
        for artifact in input_data:
            caption_text: str = artifact.summary_caption
            preprocessed.append((artifact, caption_text))

        logger.info(
            f"[SegmentCaptionEmbeddingTask] Preprocessing done — {len(preprocessed)} caption(s) ready"
        )
        return preprocessed

    async def execute(
        self,
        preprocessed: list[_PreprocessedItem],
        client: MMBertClient,
    ) -> list[tuple[TextCapSegmentEmbedArtifact, bytes]]:
        """Wrapper that calls execute_single for the batch."""
        return await self.execute_single(preprocessed, client)

    async def execute_single(
        self,
        item: list[_PreprocessedItem],
        client: MMBertClient,
    ) -> list[tuple[TextCapSegmentEmbedArtifact, bytes]]:
        """Embed a batch of caption texts via mmBERT in one request.

        Args:
            item: list of (SegmentCaptionArtifact, caption_text) — the full batch.
            client: mmBERT embedding client.

        Returns:
            list of (TextCapSegmentEmbedArtifact, npy_bytes) ready for postprocess.
        """
        logger = get_run_logger()
        logger.info(
            f"[SegmentCaptionEmbeddingTask] Embedding {len(item)} segment caption(s) via mmBERT"
        )

        texts = [caption_text for _, caption_text in item]
        embeddings: list[list[float]] | None = await client.ainfer(texts)

        if embeddings is None:
            raise RuntimeError("mmBERT ainfer returned None — check server health")

        output: list[tuple[TextCapSegmentEmbedArtifact, bytes]] = []
        for (caption_artifact, caption_text), embedding_vector in zip(item, embeddings):
            logger.info(
                f"[SegmentCaptionEmbeddingTask] Embedding done | segment={caption_artifact.start_frame}-{caption_artifact.end_frame} "
                f"dim={len(embedding_vector)} text={caption_text[:60]!r}"
            )

            artifact = TextCapSegmentEmbedArtifact(
                related_video_fps=caption_artifact.related_video_fps,
                related_video_id=caption_artifact.related_video_id,
                start_frame=caption_artifact.start_frame,
                end_frame=caption_artifact.end_frame,
                start_timestamp=caption_artifact.start_timestamp,
                end_timestamp=caption_artifact.end_timestamp,
                start_sec=caption_artifact.start_sec,
                end_sec=caption_artifact.end_sec,
                related_segment_caption_url=caption_artifact.minio_url_path,
                segment_cap_id=caption_artifact.artifact_id,
                user_id=caption_artifact.user_id,
                object_name=(
                    f"embedding/caption_segment/{caption_artifact.related_video_id}/"
                    f"{caption_artifact.start_frame}_{caption_artifact.end_frame}_"
                    f"{caption_artifact.start_timestamp}_{caption_artifact.end_timestamp}.npy"
                ),
                metadata={
                    "embedding_dim": len(embedding_vector),
                    "summary_caption": caption_text,
                },
            )

            npy_buffer = io.BytesIO()
            np.save(npy_buffer, np.array(embedding_vector, dtype=np.float32))
            output.append((artifact, npy_buffer.getvalue()))

        logger.info(
            f"[SegmentCaptionEmbeddingTask] Batch done — {len(output)} embedding(s) produced"
        )
        return output

    async def postprocess(
        self, result: list[tuple[TextCapSegmentEmbedArtifact, bytes]]
    ) -> list[TextCapSegmentEmbedArtifact]:
        """Upload .npy files to MinIO and persist artifact metadata to Postgres."""
        artifacts: list[TextCapSegmentEmbedArtifact] = []
        for artifact, npy_bytes in result:
            await self.artifact_visitor.visit_artifact(
                artifact, upload_to_minio=io.BytesIO(npy_bytes)
            )
            artifacts.append(artifact)
        return artifacts

    @staticmethod
    async def summary_artifact(final_result: list[TextCapSegmentEmbedArtifact]) -> None:
        """Create a Prefect artifact summarizing segment caption embeddings."""
        if not final_result:
            return

        first = final_result[0]

        segment_rows = ""
        for i, seg in enumerate(final_result):
            meta = seg.metadata or {}
            dim = meta.get("embedding_dim", "?")
            segment_rows += (
                f"| {i + 1} | {seg.start_timestamp} | {seg.end_timestamp} | "
                f"{seg.start_frame}-{seg.end_frame} | {dim} |\n"
            )

        markdown = (
            f"# Segment Caption Embedding Summary\n\n"
            f"| Field | Value |\n"
            f"|-------|-------|\n"
            f"| **Video ID** | `{first.related_video_id}` |\n"
            f"| **User ID** | `{first.user_id}` |\n"
            f"| **Segments Embedded** | `{len(final_result)}` |\n\n"
            f"## Segment Caption Embeddings\n\n"
            f"| # | Start | End | Frames | Embedding Dim |\n"
            f"|---|-------|-----|--------|---------------|\n"
            f"{segment_rows}"
        )

        from prefect.artifacts import acreate_markdown_artifact

        await acreate_markdown_artifact(
            key=f"segment-caption-embedding-{first.related_video_id}".lower(),
            markdown=markdown,
            description=f"Segment caption embedding summary for video {first.related_video_id}",
        )


@task(**{**_base_kwargs, "name": "Segment Caption Embedding Chunk"})  # type: ignore
async def segment_caption_embedding_chunk_task(
    items: list[SegmentCaptionArtifact],
) -> list[TextCapSegmentEmbedArtifact]:
    """Embed a batch of segment caption texts using mmBERT.

    Sends all captions in a single batched request to the mmBERT server.

    Args:
        items: Batch of SegmentCaptionArtifacts whose text to embed.

    Returns:
        List of TextCapSegmentEmbedArtifacts, one per caption in the batch.
    """
    logger = get_run_logger()
    settings = get_settings()

    logger.info(f"[SegmentCaptionEmbeddingChunk] Starting | {len(items)} caption(s) in batch")

    minio_client = MinioStorageClient(
        endpoint=settings.minio.endpoint,
        access_key=settings.minio.access_key,
        secret_key=settings.minio.secret_key,
        secure=settings.minio.secure,
    )
    postgres_client = await get_postgres_client()

    mmbert_config = MMBertConfig(
        model_name=SEGMENT_CAPTION_EMBEDDING_CONFIG.additional_kwargs.get("model_name", "mmbert"),
        base_url=SEGMENT_CAPTION_EMBEDDING_CONFIG.additional_kwargs.get(
            "base_url", "http://mmbert:8000"
        ),
    )
    logger.info(f"[SegmentCaptionEmbeddingChunk] mmBERT config | base_url={mmbert_config.base_url}")

    task_impl = SegmentCaptionEmbeddingTask(
        artifact_visitor=ArtifactPersistentVisitor(minio_client, postgres_client),
        minio_client=minio_client,
    )
    client = MMBertClient(config=mmbert_config)

    try:
        preprocessed = await task_impl.preprocess(items)
        batch_result = await task_impl.execute_single(preprocessed, client=client)
        artifacts = await task_impl.postprocess(batch_result)
        await SegmentCaptionEmbeddingTask.summary_artifact(artifacts)
    finally:
        await client.close()
        await shutdown_postgres_client(postgres_client)

    logger.info(f"[SegmentCaptionEmbeddingChunk] Done | {len(artifacts)} artifact(s) produced")
    return artifacts
