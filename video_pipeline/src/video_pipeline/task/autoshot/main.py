from __future__ import annotations
import tempfile
import numpy as np
from prefect import get_run_logger, task

from video_pipeline.task.base.base_task import TaskExecutionContext, BaseTask, TaskConfig
from video_pipeline.core.artifact import VideoArtifact, AutoshotArtifact
from video_pipeline.core.client.inference.autoshot_client import AutoShotClient, AutoshotConfig
from video_pipeline.core.storage.pg_tracker import ArtifactPersistentVisitor
from video_pipeline.core.client.storage.minio import MinioStorageClient
from video_pipeline.core.client.storage.pg import PostgresClient, PgConfig

from video_pipeline.config import get_settings

from .helper import get_batches, get_frames_fast, split_minio_url
from .helper import (
    preprocess_input_client,
    postprocess_output_client,
    predictions_to_scenes,
)


AUTOSHOT_CONFIG = TaskConfig.from_yaml("autoshot_detection")


class AutoshotTask(BaseTask[VideoArtifact, AutoshotArtifact]):

    config = AUTOSHOT_CONFIG

    async def preprocess(
        self, input_data: VideoArtifact
    ) -> list[tuple[np.ndarray, VideoArtifact]]:
        """Extract frames from video for autoshot processing.

        Args:
            input_data: Video artifact containing video metadata

        Returns:
            List of tuples containing (frames, video_artifact)
        """
        logger = get_run_logger()
        logger.debug(f"Preprocessing video for autoshot: {input_data.video_id}")

        video_path = input_data.video_minio_url
        bucket, object_name = split_minio_url(video_path)
        video_bytes = self.minio_client.get_object_bytes(bucket, object_name)

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
            tmp_file.write(video_bytes)
            tmp_file.flush()
            frames = get_frames_fast(tmp_file.name)

        return [(frames, input_data)]

    async def execute_single(
        self,
        item: tuple[np.ndarray, VideoArtifact],
        client: AutoShotClient,
        context: TaskExecutionContext,
    ) -> tuple[np.ndarray, VideoArtifact]:
        """Run autoshot inference on video frames.

        Args:
            item: Tuple of (frames, video_artifact)
            client: AutoShot inference client
            context: Task execution context

        Returns:
            Tuple of (predictions, video_artifact)

        Raises:
            ValueError: If autoshot client returns None
        """
        frames, artifact = item
        results = []

        for batch in get_batches(frames=frames):
            client_batch = preprocess_input_client(batch)
            client_result = await client.ainfer(client_batch)
            result = postprocess_output_client(client_result)  #type:ignore
            results.append(result)

        predictions = np.concatenate(results, axis=0)[: len(frames)]

        return predictions, artifact

    async def postprocess(
        self, result: tuple[np.ndarray, VideoArtifact]
    ) -> AutoshotArtifact:
        """Convert predictions to scene segments and create artifact.

        Args:
            result: Tuple of (predictions, video_artifact)

        Returns:
            AutoshotArtifact with detected scenes
        """
        predictions, video_artifact = result
        scenes = predictions_to_scenes(predictions).tolist()

        metadata = {"segments": scenes}

        autoshot_artifact = AutoshotArtifact(
            related_video_id=video_artifact.video_id,
            related_video_extension=video_artifact.video_extension,
            related_video_fps=video_artifact.fps,
            related_video_minio_url=video_artifact.video_minio_url,
            user_id=video_artifact.user_id,
            metadata=metadata,
        )

        await self.artifact_visitor.visit_artifact(autoshot_artifact)

        return autoshot_artifact

    def format_result(self, result: AutoshotArtifact) -> str:
        """Format AutoshotArtifact into Markdown for summary artifact.

        Args:
            result: Autoshot artifact to format

        Returns:
            Markdown formatted string
        """
        segments = result.metadata.get("segments", [])  # type: ignore
        segment_count = len(segments)

        preview_segments = segments[:5]
        preview_str = ", ".join(
            [f"[{round(s[0], 2)}s → {round(s[1], 2)}s]" for s in preview_segments]
        )

        return f"""### Autoshot Result

- **Video ID:** `{result.related_video_id}`
- **FPS:** {result.related_video_fps}
- **Total Segments:** {segment_count}
- **First 5 Segments:** {preview_str}
"""


@task(**AUTOSHOT_CONFIG.to_task_kwargs())
async def autoshot_task(
    video_artifact: VideoArtifact,
    context: TaskExecutionContext,
) -> AutoshotArtifact:
    logger = get_run_logger()

    logger.info(
        f"[AutoshotTask] Starting | video_id={video_artifact.video_id} "
        f"fps={video_artifact.fps} url={video_artifact.video_minio_url}"
    )

    settings = get_settings()

    minio_client = MinioStorageClient(
        endpoint=settings.minio.endpoint,
        access_key=settings.minio.access_key,
        secret_key=settings.minio.secret_key,
        secure=settings.minio.secure,
    )
    logger.info(f"[AutoshotTask] MinIO client initialized at {settings.minio.endpoint}")

    postgres_client = PostgresClient(
        config=PgConfig( #type:ignore
            database_url=settings.postgres.connection_string,
        )
    )
    logger.info("[AutoshotTask] Postgres client initialized")

    autoshot_config = AutoshotConfig(
        model_name=AUTOSHOT_CONFIG.additional_kwargs['model_name'],
        model_version=AUTOSHOT_CONFIG.additional_kwargs['model_version'],
    )
    logger.info(
        f"[AutoshotTask] Autoshot model config | "
        f"model={autoshot_config.model_name} version={autoshot_config.model_version}"
    )

    task_impl = AutoshotTask(
        artifact_visitor=ArtifactPersistentVisitor(minio_client, postgres_client),
        minio_client=MinioStorageClient(
            endpoint=settings.minio.endpoint,
            access_key=settings.minio.access_key,
            secret_key=settings.minio.secret_key,
            secure=settings.minio.secure,
        ),
    )

    all_produced_artifacts: list[AutoshotArtifact] = []

    logger.info(f"[AutoshotTask] Connecting to Triton at {settings.triton.url}...")
    async with AutoShotClient(url=settings.triton.url, config=autoshot_config) as client:
        logger.info("[AutoshotTask] Preprocessing — extracting frames from video...")
        preprocessed = await task_impl.preprocess(input_data=video_artifact)
        logger.info(f"[AutoshotTask] Preprocessing done — {len(preprocessed)} batch(es) ready for inference")

        async for result in task_impl.execute(preprocessed, client, context):
            artifact = await task_impl.postprocess(result)
            all_produced_artifacts.append(artifact)
            segments = artifact.metadata.get("segments", []) if artifact.metadata else []
            logger.info(
                f"[AutoshotTask] Artifact persisted | "
                f"video_id={artifact.related_video_id} shots={len(segments)}"
            )

            

    await task_impl.create_summary_artifact(all_produced_artifacts, context)
    logger.info(
        f"[AutoshotTask] Complete | {len(all_produced_artifacts)} artifact(s) | "
        f"total_time={context.total_time_ms:.0f}ms"
    )

    return all_produced_artifacts[0] #intentional: ignore
