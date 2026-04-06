from __future__ import annotations

import re

from prefect import get_run_logger, task
from prefect.artifacts import acreate_markdown_artifact, acreate_table_artifact

from video_pipeline.task.base.base_task import TaskConfig, BaseTask
from video_pipeline.core.client.progress import StageRegistry
from video_pipeline.core.artifact import AutoshotArtifact, ASRArtifact
from video_pipeline.core.storage.pg_tracker import ArtifactPersistentVisitor
from video_pipeline.core.client.storage.minio import MinioStorageClient
from video_pipeline.core.client.storage.pg.runtime import (
    get_postgres_client,
    shutdown_postgres_client,
)
from video_pipeline.core.client.inference.asr_client import QwenASRClient, QwenASRConfig
from video_pipeline.config import get_settings

from .helper import (
    frames_to_timestamp,
    delete_audio_file,
    parse_asr_response,
)

ASR_CONFIG = TaskConfig.from_yaml("asr_transcription")

ASRItem = tuple[AutoshotArtifact, int, int, str]


@StageRegistry.register
class ASRTask(BaseTask[list[ASRItem], list[ASRArtifact]]):
    """Process a batch of audio segments for ASR transcription."""

    config = ASR_CONFIG

    async def preprocess(self, input_data: list[ASRItem]) -> list[ASRItem]:
        logger = get_run_logger()
        logger.info(f"[ASRTask] Preprocessing {len(input_data)} segment(s)")
        return input_data

    async def execute(
        self, preprocessed: list[ASRItem], client: QwenASRClient
    ) -> list[ASRArtifact]:
        """Run ASR inference on audio segments then delete temp files."""
        audio_paths = [data[-1] for data in preprocessed]
        try:
            raw_results = await client.ainfer(audio_paths)
            if raw_results is None:
                raise RuntimeError("ASR client returned None — inference failed")
        finally:
            for audio_path in audio_paths:
                delete_audio_file(audio_path)

        asr_artifact_list = []

        for raw_result, asr_data in zip(raw_results, preprocessed):
            autoshot_artifact, start_frame, end_frame, audio_path = asr_data

            fps = autoshot_artifact.related_video_fps
            start_ts = frames_to_timestamp(start_frame, fps)
            end_ts = frames_to_timestamp(end_frame, fps)
            duration_sec = round((end_frame - start_frame) / fps, 3)
            text = parse_asr_response(raw_result["text"]) if raw_result else ""
            asr_artifact = ASRArtifact(
                related_autoshot_artifact_id=autoshot_artifact.artifact_id,
                related_video_minio_url=autoshot_artifact.related_video_minio_url,
                related_video_extension=autoshot_artifact.related_video_extension,
                related_video_fps=fps,
                related_video_id=autoshot_artifact.related_video_id,
                user_id=autoshot_artifact.user_id,
                metadata={
                    "timestamp": [start_ts, end_ts],
                    "frame_num": [start_frame, end_frame],
                    "text": text,
                    "duration": duration_sec,
                },
            )
            asr_artifact_list.append(asr_artifact)

        return asr_artifact_list

    async def postprocess(self, result: list[ASRArtifact]) -> list[ASRArtifact]:
        """Persist ASR artifact to database."""
        for res in result:
            await self.artifact_visitor.visit_artifact(res)
        return result

    @staticmethod
    async def summary_artifact(final_result: list[ASRArtifact]) -> None:
        """Create a Prefect markdown artifact summarising an ASR batch."""
        if not final_result:
            return

        first = final_result[0]
        raw_key = f"asr-{first.related_autoshot_artifact_id}".lower()
        key = re.sub(r"[^a-z0-9-]", "-", raw_key)

        total_duration = sum(a.metadata.get("duration", 0.0) for a in final_result if a.metadata)

        segment_rows = ""
        for i, artifact in enumerate(final_result):
            meta = artifact.metadata or {}
            start_ts, end_ts = meta.get("timestamp") or ["N/A", "N/A"]
            duration = meta.get("duration", "N/A")
            text = meta.get("text", "")
            display_text = (text[:80] + "…") if len(text) > 80 else text
            segment_rows += f"| {i + 1} | {start_ts} | {end_ts} | {duration}s | {display_text} |\n"

        markdown = (
            f"# ASR Transcription Summary\n\n"
            f"| Field | Value |\n"
            f"|-------|-------|\n"
            f"| **Autoshot Artifact ID** | `{first.related_autoshot_artifact_id}` |\n"
            f"| **User ID** | `{first.user_id}` |\n"
            f"| **FPS** | `{first.related_video_fps}` |\n"
            f"| **Segments Processed** | `{len(final_result)}` |\n"
            f"| **Total Duration** | `{total_duration:.2f}s` |\n\n"
            f"## Transcript Segments\n\n"
            f"| # | Start | End | Duration | Transcript |\n"
            f"|---|-------|-----|----------|------------|\n"
            f"{segment_rows}"
        )

        await acreate_markdown_artifact(
            key=key,
            markdown=markdown,
            description=f"ASR transcription summary for autoshot {first.related_autoshot_artifact_id}",
        )

        await acreate_table_artifact(
            table=[
                {"Field": "Autoshot Artifact ID", "Value": str(first.related_autoshot_artifact_id)},
                {"Field": "User ID", "Value": str(first.user_id)},
                {"Field": "FPS", "Value": str(first.related_video_fps)},
                {"Field": "Segments Processed", "Value": str(len(final_result))},
                {"Field": "Total Duration", "Value": f"{total_duration:.2f}s"},
            ],
            key=f"{key}-summary-table",
            description=f"ASR summary table for autoshot {first.related_autoshot_artifact_id}",
        )

        segments_table = []
        for i, artifact in enumerate(final_result):
            meta = artifact.metadata or {}
            start_ts, end_ts = meta.get("timestamp") or ["N/A", "N/A"]
            duration = meta.get("duration", "N/A")
            text = meta.get("text", "")
            display_text = (text[:80] + "…") if len(text) > 80 else text
            segments_table.append(
                {
                    "#": i + 1,
                    "Start": start_ts,
                    "End": end_ts,
                    "Duration": f"{duration}s",
                    "Transcript": display_text,
                }
            )
        await acreate_table_artifact(
            table=segments_table,
            key=f"{key}-segments-table",
            description=f"ASR transcript segments for autoshot {first.related_autoshot_artifact_id}",
        )


@task(**{**ASR_CONFIG.to_task_kwargs(), "name": "ASR Chunk"})  # type:ignore
async def asr_chunk_task(
    items: list[ASRItem],
) -> list[ASRArtifact]:
    """Process a batch of ASR segments using ASRTask.execute().

    Iterates over each segment in the batch sequentially via BaseTask.execute(),
    persisting each artifact immediately after inference via postprocess().

    Args:
        items: Batch of (AutoshotArtifact, start_frame, end_frame, audio_path)
        context: Task execution context.

    Returns:
        List of ASRArtifacts, one per segment in the batch.
    """
    logger = get_run_logger()
    settings = get_settings()

    logger.info(f"[ASRChunk] Starting | {len(items)} segment(s) in batch")

    minio_client = MinioStorageClient(
        endpoint=settings.minio.endpoint,
        access_key=settings.minio.access_key,
        secret_key=settings.minio.secret_key,
        secure=settings.minio.secure,
    )
    postgres_client = await get_postgres_client()
    logger.info(f"[ASRChunk] Clients initialized | minio={settings.minio.endpoint}")

    asr_config = QwenASRConfig(model_name=ASR_CONFIG.additional_kwargs["model_name"])
    client_url: str = ASR_CONFIG.additional_kwargs["client_url"]
    logger.info(f"[ASRChunk] ASR model config | model={asr_config.model_name} url={client_url}")

    task_impl = ASRTask(
        artifact_visitor=ArtifactPersistentVisitor(minio_client, postgres_client),
        minio_client=minio_client,
    )
    try:
        async with QwenASRClient(client_url=client_url, config=asr_config) as client:
            artifacts = await task_impl.execute_template(items, client)
    finally:
        await shutdown_postgres_client(postgres_client)

    logger.info(f"[ASRChunk] Done | {len(artifacts)} artifact(s) produced")
    return artifacts
