from __future__ import annotations
import re
import tempfile
import numpy as np
from prefect import get_run_logger, task
from prefect.artifacts import acreate_markdown_artifact, acreate_table_artifact

from video_pipeline.task.base.base_task import BaseTask, TaskConfig
from video_pipeline.core.client.progress import StageRegistry
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

@StageRegistry.register
class AutoshotTask(BaseTask[VideoArtifact, AutoshotArtifact]):

    async def preprocess(
        self, input_data: VideoArtifact
    ) -> tuple[np.ndarray, VideoArtifact]:
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

        return (frames, input_data)

    async def execute(
        self,
        preprocessed: tuple[np.ndarray, VideoArtifact],
        client: AutoShotClient,
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
        frames, artifact = preprocessed
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

    @staticmethod
    async def summary_artifact(final_result: AutoshotArtifact) -> None:
        """Create a Prefect markdown artifact summarising autoshot detection results."""
        segments: list = (
            final_result.metadata.get("segments", []) if final_result.metadata else []
        )
        fps = final_result.related_video_fps

        raw_key = f"autoshot-{final_result.related_video_id}".lower()
        key = re.sub(r"[^a-z0-9-]", "-", raw_key)

        shot_rows = ""
        for i, seg in enumerate(segments):
            start_frame, end_frame = seg[0], seg[1]
            start_ts = f"{start_frame / fps:.2f}s" if fps else "N/A"
            end_ts = f"{end_frame / fps:.2f}s" if fps else "N/A"
            shot_rows += f"| {i + 1} | {start_frame} | {end_frame} | {start_ts} | {end_ts} |\n"

        shots_table = (
f"## Detected Shots ({len(segments)})\n\n"
f"| # | Start Frame | End Frame | Start Time | End Time |\n"
f"|---|-------------|-----------|------------|----------|\n"
f"{shot_rows}"
        ) if segments else "## Detected Shots\n\n_No shots detected._\n"

        markdown = (
f"# Autoshot Detection Summary\n\n"
f"| Field | Value |\n"
f"|-------|-------|\n"
f"| **Artifact ID** | `{final_result.artifact_id}` |\n"
f"| **Related Video ID** | `{final_result.related_video_id}` |\n"
f"| **User ID** | `{final_result.user_id}` |\n"
f"| **Extension** | `{final_result.related_video_extension}` |\n"
f"| **FPS** | `{fps}` |\n"
f"| **Total Shots** | `{len(segments)}` |\n\n"
f"{shots_table}"
        )

        await acreate_markdown_artifact(
            key=key,
            markdown=markdown,
            description=f"Autoshot detection summary for video {final_result.related_video_id}",
        )

        await acreate_table_artifact(
            table=[
                {"Field": "Artifact ID", "Value": str(final_result.artifact_id)},
                {"Field": "Related Video ID", "Value": str(final_result.related_video_id)},
                {"Field": "User ID", "Value": str(final_result.user_id)},
                {"Field": "Extension", "Value": str(final_result.related_video_extension)},
                {"Field": "FPS", "Value": str(fps)},
                {"Field": "Total Shots", "Value": str(len(segments))},
            ],
            key=f"{key}-summary-table",
            description=f"Autoshot summary table for video {final_result.related_video_id}",
        )

        if segments:
            shots_table = [
                {
                    "#": i + 1,
                    "Start Frame": seg[0],
                    "End Frame": seg[1],
                    "Start Time": f"{seg[0] / fps:.2f}s" if fps else "N/A",
                    "End Time": f"{seg[1] / fps:.2f}s" if fps else "N/A",
                }
                for i, seg in enumerate(segments)
            ]
            await acreate_table_artifact(
                table=shots_table,
                key=f"{key}-shots-table",
                description=f"Detected shots for video {final_result.related_video_id}",
            )


@task(**AUTOSHOT_CONFIG.to_task_kwargs())
async def autoshot_task(
    video_artifact: VideoArtifact,
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
        minio_client=minio_client
    )
    async with AutoShotClient(url=settings.triton.url, config=autoshot_config) as client:
        artifact = await task_impl.execute_template(video_artifact, client)

    return artifact
