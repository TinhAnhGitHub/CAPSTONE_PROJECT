"""Audio Transcript Embedding Task.

This task embeds the audio_text from AudioSegmentArtifact using mmBERT,
enabling semantic search over spoken content via text-based embeddings.

Flow:
    AudioSegmentArtifact → mmBERT embedding → AudioTranscriptEmbedArtifact

The audio_text field contains the raw ASR transcript, which is embedded
to enable semantic search over spoken content independent of the
segment caption quality.
"""

from __future__ import annotations

import io
from typing import Any

import numpy as np
from prefect import get_run_logger, task
from prefect.artifacts import acreate_markdown_artifact

from video_pipeline.config import get_settings
from video_pipeline.core.artifact import AudioSegmentArtifact, AudioTranscriptEmbedArtifact
from video_pipeline.core.client.inference.te_client import MMBertClient, MMBertConfig
from video_pipeline.core.client.progress import StageRegistry
from video_pipeline.core.client.storage.minio import MinioStorageClient
from video_pipeline.core.client.storage.pg.runtime import (
    get_postgres_client,
    shutdown_postgres_client,
)
from video_pipeline.core.storage.pg_tracker import ArtifactPersistentVisitor
from video_pipeline.task.base.base_task import BaseTask, TaskConfig


AUDIO_TRANSCRIPT_EMBEDDING_CONFIG = TaskConfig.from_yaml("audio_transcript_embedding")
_base_kwargs = AUDIO_TRANSCRIPT_EMBEDDING_CONFIG.to_task_kwargs()

# Preprocessed item: (AudioSegmentArtifact, audio_text)
_PreprocessedItem = tuple[AudioSegmentArtifact, str]


@StageRegistry.register
class AudioTranscriptEmbeddingTask(
    BaseTask[list[AudioSegmentArtifact], list[AudioTranscriptEmbedArtifact]]
):
    """Embed audio transcript text using mmBERT dense embeddings.

    This task processes the audio_text field from AudioSegmentArtifact,
    generating 768-dimensional dense embeddings suitable for semantic
    similarity search over spoken content.

    Attributes:
        config: Task configuration loaded from tasks.yaml

    Pipeline Position:
        Runs after AudioSegment, parallel to SegmentEmbedding and SegmentCaption.
        Input: AudioSegmentArtifact (contains audio_text from ASR)
        Output: AudioTranscriptEmbedArtifact (dense embedding + metadata)
    """

    config = AUDIO_TRANSCRIPT_EMBEDDING_CONFIG

    async def preprocess(
        self, input_data: list[AudioSegmentArtifact]
    ) -> list[_PreprocessedItem]:
        """Extract audio_text from each AudioSegmentArtifact.

        Args:
            input_data: List of AudioSegmentArtifact from audio_segment task

        Returns:
            List of (AudioSegmentArtifact, audio_text) tuples ready for embedding
        """
        logger = get_run_logger()
        logger.info(f"[AudioTranscriptEmbeddingTask] Preparing {len(input_data)} segment(s)")

        preprocessed: list[_PreprocessedItem] = []
        for artifact in input_data:
            audio_text: str = artifact.audio_text.strip()
            if not audio_text:
                logger.warning(
                    f"[AudioTranscriptEmbeddingTask] Segment {artifact.segment_index} has empty audio_text, skipping"
                )
                continue
            preprocessed.append((artifact, audio_text))

        logger.info(
            f"[AudioTranscriptEmbeddingTask] Preprocessing done — {len(preprocessed)} text(s) ready for embedding"
        )
        return preprocessed

    async def execute(
        self,
        preprocessed: list[_PreprocessedItem],
        client: MMBertClient,
    ) -> list[tuple[AudioTranscriptEmbedArtifact, bytes]]:
        """Embed audio transcript texts via mmBERT in batched requests.

        Args:
            preprocessed: List of (AudioSegmentArtifact, audio_text) tuples
            client: mmBERT embedding client

        Returns:
            List of (AudioTranscriptEmbedArtifact, npy_bytes) ready for postprocess
        """
        logger = get_run_logger()
        logger.info(
            f"[AudioTranscriptEmbeddingTask] Embedding {len(preprocessed)} transcript(s) via mmBERT"
        )

        texts = [audio_text for _, audio_text in preprocessed]
        embeddings: list[list[float]] | None = await client.ainfer(texts)

        if embeddings is None:
            raise RuntimeError("mmBERT ainfer returned None — check server health")

        output: list[tuple[AudioTranscriptEmbedArtifact, bytes]] = []
        for (segment_artifact, audio_text), embedding_vector in zip(preprocessed, embeddings):
            text_preview = audio_text[:80] + "..." if len(audio_text) > 80 else audio_text
            logger.info(
                f"[AudioTranscriptEmbeddingTask] Embedded segment {segment_artifact.segment_index} | "
                f"dim={len(embedding_vector)} text={text_preview!r}"
            )

            artifact = AudioTranscriptEmbedArtifact(
                related_audio_segment_artifact_id=segment_artifact.artifact_id,
                related_video_id=segment_artifact.related_video_id,
                related_video_minio_url=segment_artifact.related_video_minio_url,
                related_video_extension=segment_artifact.related_video_extension,
                related_video_fps=segment_artifact.related_video_fps,
                segment_index=segment_artifact.segment_index,
                start_frame=segment_artifact.start_frame,
                end_frame=segment_artifact.end_frame,
                start_timestamp=segment_artifact.start_timestamp,
                end_timestamp=segment_artifact.end_timestamp,
                start_sec=segment_artifact.start_sec,
                end_sec=segment_artifact.end_sec,
                audio_text=audio_text,
                embedding_dim=len(embedding_vector),
                user_id=segment_artifact.user_id,
                object_name=(
                    f"embedding/audio_transcript/{segment_artifact.related_video_id}/"
                    f"segment_{segment_artifact.segment_index}_"
                    f"{segment_artifact.start_frame}_{segment_artifact.end_frame}.npy"
                ),
                metadata={
                    "embedding_dim": len(embedding_vector),
                    "audio_text_length": len(audio_text),
                    "audio_text_preview": text_preview,
                },
            )

            npy_buffer = io.BytesIO()
            np.save(npy_buffer, np.array(embedding_vector, dtype=np.float32))
            output.append((artifact, npy_buffer.getvalue()))

        logger.info(
            f"[AudioTranscriptEmbeddingTask] Batch done — {len(output)} embedding(s) produced"
        )
        return output

    async def postprocess(
        self, result: list[tuple[AudioTranscriptEmbedArtifact, bytes]]
    ) -> list[AudioTranscriptEmbedArtifact]:
        """Upload .npy files to MinIO and persist artifact metadata to Postgres.

        Args:
            result: List of (AudioTranscriptEmbedArtifact, npy_bytes) from execute

        Returns:
            List of persisted AudioTranscriptEmbedArtifact
        """
        logger = get_run_logger()
        logger.info(f"[AudioTranscriptEmbeddingTask] Uploading {len(result)} embedding(s) to MinIO")

        artifacts: list[AudioTranscriptEmbedArtifact] = []
        for artifact, npy_bytes in result:
            await self.artifact_visitor.visit_artifact(
                artifact, upload_to_minio=io.BytesIO(npy_bytes)
            )
            artifacts.append(artifact)

        logger.info(f"[AudioTranscriptEmbeddingTask] Persisted {len(artifacts)} artifact(s)")
        return artifacts

    @staticmethod
    async def summary_artifact(final_result: list[AudioTranscriptEmbedArtifact]) -> None:
        """Create a Prefect artifact summarizing audio transcript embeddings.

        Displays per-segment embedding statistics with text previews for
        quick verification of transcript quality and coverage.

        Args:
            final_result: List of AudioTranscriptEmbedArtifact from postprocess
        """
        if not final_result:
            return

        first = final_result[0]

        # Sort by segment index for ordered display
        sorted_results = sorted(final_result, key=lambda x: x.segment_index)

        segment_rows = ""
        total_text_length = 0
        for seg in sorted_results:
            meta = seg.metadata or {}
            dim = meta.get("embedding_dim", seg.embedding_dim)
            text_len = meta.get("audio_text_length", len(seg.audio_text))
            total_text_length += text_len
            text_preview = meta.get("audio_text_preview", seg.audio_text[:80])

            segment_rows += (
                f"| {seg.segment_index} | {seg.start_timestamp} | {seg.end_timestamp} | "
                f"{seg.start_frame}-{seg.end_frame} | {dim} | {text_len} | {text_preview} |\n"
            )

        avg_text_len = total_text_length / len(final_result) if final_result else 0

        markdown = (
            f"# Audio Transcript Embedding Summary\n\n"
            f"Semantic embeddings generated from ASR transcripts using mmBERT.\n\n"
            f"| Field | Value |\n"
            f"|-------|-------|\n"
            f"| **Video ID** | `{first.related_video_id}` |\n"
            f"| **User ID** | `{first.user_id}` |\n"
            f"| **Segments Embedded** | `{len(final_result)}` |\n"
            f"| **Embedding Dimension** | `{first.embedding_dim}` |\n"
            f"| **Total Text Characters** | `{total_text_length}` |\n"
            f"| **Avg Text per Segment** | `{avg_text_len:.1f}` |\n\n"
            f"## Segment Details\n\n"
            f"| # | Start | End | Frames | Dim | Text Len | Preview |\n"
            f"|---|-------|-----|--------|-----|----------|--------|\n"
            f"{segment_rows}\n"
            f"### Notes\n\n"
            f"- **Source**: ASR transcript (audio_text) from AudioSegmentArtifact\n"
            f"- **Model**: mmBERT dense text embeddings\n"
            f"- **Purpose**: Semantic search over spoken content\n"
        )

        await acreate_markdown_artifact(
            key=f"audio-transcript-embedding-{first.related_video_id}".lower(),
            markdown=markdown,
            description=f"Audio transcript embedding summary for video {first.related_video_id}",
        )


@task(**{**_base_kwargs, "name": "Audio Transcript Embedding Chunk"})  # type: ignore
async def audio_transcript_embedding_chunk_task(
    items: list[AudioSegmentArtifact],
) -> list[AudioTranscriptEmbedArtifact]:
    """Embed a batch of audio transcript texts using mmBERT.

    Processes the audio_text field from AudioSegmentArtifact to generate
    dense embeddings suitable for semantic similarity search.

    Args:
        items: Batch of AudioSegmentArtifact from audio_segment task

    Returns:
        List of AudioTranscriptEmbedArtifact, one per segment with non-empty text
    """
    logger = get_run_logger()
    settings = get_settings()

    logger.info(f"[AudioTranscriptEmbeddingChunk] Starting | {len(items)} segment(s) in batch")

    minio_client = MinioStorageClient(
        endpoint=settings.minio.endpoint,
        access_key=settings.minio.access_key,
        secret_key=settings.minio.secret_key,
        secure=settings.minio.secure,
    )
    postgres_client = await get_postgres_client()

    mmbert_config = MMBertConfig(
        model_name=AUDIO_TRANSCRIPT_EMBEDDING_CONFIG.additional_kwargs.get(
            "model_name", "mmbert"
        ),
        base_url=AUDIO_TRANSCRIPT_EMBEDDING_CONFIG.additional_kwargs.get(
            "base_url", "http://mmbert:8000"
        ),
    )
    logger.info(
        f"[AudioTranscriptEmbeddingChunk] mmBERT config | base_url={mmbert_config.base_url}"
    )

    task_impl = AudioTranscriptEmbeddingTask(
        artifact_visitor=ArtifactPersistentVisitor(minio_client, postgres_client),
        minio_client=minio_client,
    )
    client = MMBertClient(config=mmbert_config)

    try:
        preprocessed = await task_impl.preprocess(items)
        batch_result = await task_impl.execute(preprocessed, client=client)
        artifacts = await task_impl.postprocess(batch_result)
    finally:
        await client.close()
        await shutdown_postgres_client(postgres_client)

    logger.info(
        f"[AudioTranscriptEmbeddingChunk] Done | {len(artifacts)} artifact(s) produced"
    )
    return artifacts