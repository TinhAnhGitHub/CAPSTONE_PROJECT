from __future__ import annotations

from prefect import get_run_logger, task

from video_pipeline.task.base.base_task import TaskExecutionContext, TaskConfig, BaseTask
from video_pipeline.core.artifact import AutoshotArtifact, ASRArtifact
from video_pipeline.core.storage.pg_tracker import ArtifactPersistentVisitor
from video_pipeline.core.client.storage.minio import MinioStorageClient
from video_pipeline.core.client.storage.pg import PostgresClient, PgConfig
from video_pipeline.core.client.inference.asr_client import QwenASRClient, QwenASRConfig
from video_pipeline.config import get_settings

from .helper import (
    frames_to_timestamp,
    delete_audio_file,
    parse_asr_response,
)


ASR_CONFIG = TaskConfig.from_yaml("asr_transcription")
_base_kwargs = ASR_CONFIG.to_task_kwargs()

ASRItem = tuple[AutoshotArtifact, int, int, str]


class ASRTask(BaseTask[list[ASRItem], ASRArtifact]):
    """ASR task that processes a batch of audio segments sequentially.

    Receives a list of ASRItems via execute(), which iterates and calls
    execute_single for each segment.
    """

    config = ASR_CONFIG

    async def preprocess(self, input_data: list[ASRItem]) -> list[ASRItem]:
        logger = get_run_logger()
        logger.info(f"[ASRTask] Preprocessing {len(input_data)} segment(s)")
        return input_data

    async def execute_single(
        self,
        item: ASRItem,
        client: QwenASRClient,
        context: TaskExecutionContext,
    ) -> ASRArtifact:
        """Run ASR inference on one audio segment then delete the temp file.

        Args:
            item: (AutoshotArtifact, start_frame, end_frame, audio_path)
            client: Qwen ASR inference client
            context: Task execution context

        Returns:
            ASRArtifact with single-segment metadata.
        """
        logger = get_run_logger()
        autoshot_artifact, start_frame, end_frame, audio_path = item
        fps = autoshot_artifact.related_video_fps
        start_ts = frames_to_timestamp(start_frame, fps)
        end_ts = frames_to_timestamp(end_frame, fps)
        duration_sec = round((end_frame - start_frame) / fps, 3)

        logger.info(
            f"[ASRTask] Inferring segment [{start_ts} → {end_ts}] | "
            f"duration={duration_sec}s audio={audio_path}"
        )

        try:
            raw_results = await client.ainfer([audio_path])
            if raw_results is None:
                raise RuntimeError("ASR client returned None — inference failed")
            raw = raw_results[0] if raw_results else ""
            text = parse_asr_response(raw) if raw else ""
        finally:
            delete_audio_file(audio_path)
            logger.debug(f"[ASRTask] Deleted temp audio: {audio_path}")

        logger.info(
            f"[ASRTask] Segment done | [{start_ts} → {end_ts}] "
            f"text_preview={text[:60]!r}"
        )

        return ASRArtifact(
            related_autoshot_artifact_id=autoshot_artifact.artifact_id,
            related_video_minio_url=autoshot_artifact.related_video_minio_url,
            related_video_extension=autoshot_artifact.related_video_extension,
            related_video_fps=fps,
            user_id=autoshot_artifact.user_id,
            metadata={
                "timestamp": [start_ts, end_ts],
                "frame_num": [start_frame, end_frame],
                "text": text,
                "duration": duration_sec,
            },
        )

    async def postprocess(self, result: ASRArtifact) -> ASRArtifact:
        """Persist ASR artifact to database."""
        await self.artifact_visitor.visit_artifact(result)
        return result

    def format_result(self, result: ASRArtifact) -> str:
        """Format a single ASRArtifact into Markdown for the summary artifact."""
        meta = result.metadata or {}
        timestamp = meta.get("timestamp", ["?", "?"])
        text = meta.get("text", "")
        duration = meta.get("duration", 0)
        return f"""### ASR Segment

- **Autoshot Artifact ID:** `{result.related_autoshot_artifact_id}`
- **User ID:** `{result.user_id}`
- **Timestamp:** [{timestamp[0]} → {timestamp[1]}]
- **Duration:** {duration}s
- **Text:** {text[:120]}
"""



@task(**{**_base_kwargs, "name": "ASR Chunk"}) #type:ignore
async def asr_chunk_task(
    items: list[ASRItem],
    context: TaskExecutionContext,
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
    postgres_client = PostgresClient(
        config=PgConfig(database_url=settings.postgres.connection_string)  # type: ignore
    )
    logger.info(f"[ASRChunk] Clients initialized | minio={settings.minio.endpoint}")

    asr_config = QwenASRConfig(model_name=ASR_CONFIG.additional_kwargs["model_name"])
    client_url: str = ASR_CONFIG.additional_kwargs["client_url"]
    logger.info(
        f"[ASRChunk] ASR model config | model={asr_config.model_name} url={client_url}"
    )

    task_impl = ASRTask(
        artifact_visitor=ArtifactPersistentVisitor(minio_client, postgres_client),
        minio_client=minio_client,
    )
    asr_client = QwenASRClient(client_url=client_url, config=asr_config)

    try:
        preprocessed = await task_impl.preprocess(items)
        artifacts: list[ASRArtifact] = []
        async for result in task_impl.execute(preprocessed, asr_client, context):
            artifact = await task_impl.postprocess(result)
            artifacts.append(artifact)
        await task_impl.create_summary_artifact(artifacts, context)
    finally:
        await asr_client.close()

    logger.info(f"[ASRChunk] Done | {len(artifacts)} artifact(s) produced")
    return artifacts
