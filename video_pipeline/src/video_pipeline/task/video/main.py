from __future__ import annotations

import re

from pydantic import BaseModel

from prefect import get_run_logger, task
from prefect.artifacts import acreate_markdown_artifact, acreate_table_artifact

from video_pipeline.task.base.base_task import (
    BaseTask,
    TaskConfig,
)
from video_pipeline.core.client.progress import StageRegistry
from video_pipeline.core.artifact import VideoArtifact
from video_pipeline.core.storage.pg_tracker import ArtifactPersistentVisitor
from video_pipeline.core.client.storage.minio import MinioStorageClient
from video_pipeline.core.client.storage.pg.runtime import get_postgres_client, shutdown_postgres_client
from video_pipeline.config import get_settings

from .helper import (
    parse_s3_url,
    extract_extension,
    get_video_fps,
    get_video_duration_ffprobe,
)


class VideoInput(BaseModel):
    user_id: str
    video_s3_url: str
    video_id: str


VIDEO_CONFIG = TaskConfig.from_yaml("video_registration")
print(VIDEO_CONFIG.to_task_kwargs())


@StageRegistry.register
class VideoRegistryTask(BaseTask[VideoInput, VideoArtifact]):
    async def preprocess(self, input_data: VideoInput) -> VideoInput:
        """Wrap single video input into a list for execution."""
        return input_data

    async def execute(
        self,
        preprocessed: VideoInput,
        client: None,
    ) -> VideoArtifact:
        """Extract video metadata and create artifact.

        Args:
            item: Video input data
            client: Not used for this task
            context: Task execution context

        Returns:
            VideoArtifact with extracted metadata
        """
        logger = get_run_logger()
        logger.info(f"Processing video: {preprocessed.video_id} from {preprocessed.video_s3_url}")

        bucket, object_name = parse_s3_url(preprocessed.video_s3_url)
        video_extension = extract_extension(preprocessed.video_s3_url)

        async with self.minio_client.fetch_object_from_s3(
            s3_url=preprocessed.video_s3_url,
            suffix=f".{video_extension}",
        ) as video_tmp_path:
            fps = get_video_fps(video_tmp_path)
            duration = get_video_duration_ffprobe(path=video_tmp_path)

        metadata = {"fps": fps, "extension": video_extension, "duration": duration}

        video_artifact = VideoArtifact(
            artifact_id=preprocessed.video_id,
            user_id=preprocessed.user_id,
            video_id=preprocessed.video_id,
            video_extension=video_extension,
            video_minio_url=f"s3://{bucket}/{object_name}",
            fps=fps,
            object_name=object_name,
            metadata=metadata,
        )
        return video_artifact

    async def postprocess(self, result: VideoArtifact) -> VideoArtifact:
        """Persist artifact to database.

        Args:
            result: Video artifact to persist

        Returns:
            The persisted video artifact
        """
        await self.artifact_visitor.visit_artifact(result)
        return result

    @staticmethod
    async def summary_artifact(final_result: VideoArtifact) -> None:
        """Create a Prefect markdown artifact summarising video registration results."""
        duration = (
            final_result.metadata.get("duration", "N/A")
            if final_result.metadata
            else "N/A"
        )

        raw_key = f"video-reg-{final_result.video_id}".lower()
        key = re.sub(r"[^a-z0-9-]", "-", raw_key)

        markdown = (
f"# Video Registration Summary\n\n"
f"| Field | Value |\n"
f"|-------|-------|\n"
f"| **Video ID** | `{final_result.video_id}` |\n"
f"| **User ID** | `{final_result.user_id}` |\n"
f"| **Extension** | `{final_result.video_extension}` |\n"
f"| **FPS** | `{final_result.fps}` |\n"
f"| **Duration** | `{duration}s` |\n"
f"| **MinIO URL** | `{final_result.video_minio_url}` |\n"
f"| **Artifact ID** | `{final_result.artifact_id}` |\n"
        )

        await acreate_markdown_artifact(
            key=key,
            markdown=markdown,
            description=f"Video registration summary for video {final_result.video_id}",
        )

        await acreate_table_artifact(
            table=[
                {"Field": "Video ID", "Value": str(final_result.video_id)},
                {"Field": "User ID", "Value": str(final_result.user_id)},
                {"Field": "Extension", "Value": str(final_result.video_extension)},
                {"Field": "FPS", "Value": str(final_result.fps)},
                {"Field": "Duration", "Value": f"{duration}s"},
                {"Field": "MinIO URL", "Value": str(final_result.video_minio_url)},
                {"Field": "Artifact ID", "Value": str(final_result.artifact_id)},
            ],
            key=f"{key}-table",
            description=f"Video registration table for video {final_result.video_id}",
        )


@task(**VIDEO_CONFIG.to_task_kwargs())
async def video_reg_task(
    video_input: VideoInput,
) -> VideoArtifact:
    """Prefect task for video registration.

    Args:
        video_input: Video input data
        context: Task execution context

    Returns:
        Registered video artifact
    """
    settings = get_settings()

    logger = get_run_logger()

    logger.info(
        f"[VideoRegTask] Starting | video_id={video_input.video_id} "
        f"user_id={video_input.user_id} s3={video_input.video_s3_url}"
    )

    logger.debug(f"[VideoRegTask] Minio config: {settings.minio=}")
    minio_client = MinioStorageClient(
        endpoint=settings.minio.endpoint,
        access_key=settings.minio.access_key,
        secret_key=settings.minio.secret_key,
        secure=settings.minio.secure,
    )
    logger.info(f"[VideoRegTask] MinIO client initialized at {settings.minio.endpoint}")

    postgres_client = await get_postgres_client()
    logger.info("[VideoRegTask] Postgres client initialized")

    task_impl = VideoRegistryTask(
        artifact_visitor=ArtifactPersistentVisitor(
            minio_client,
            postgres_client
        ),
        minio_client=minio_client
    )

    logger.info("[VideoRegTask] Preprocessing input...")

    try:
        preprocessed = await task_impl.preprocess(input_data=video_input)
        result = await task_impl.execute(preprocessed, client=None)
        artifact = await task_impl.postprocess(result)
        return artifact
    finally:
        await shutdown_postgres_client(postgres_client)
