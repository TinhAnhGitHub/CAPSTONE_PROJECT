from __future__ import annotations

from pydantic import BaseModel

from prefect import get_run_logger, task

from video_pipeline.task.base.base_task import (
    TaskExecutionContext,
    BaseTask,
    TaskConfig,
)
from video_pipeline.core.artifact import VideoArtifact
from video_pipeline.core.storage.pg_tracker import ArtifactPersistentVisitor
from video_pipeline.core.client.storage.minio import MinioStorageClient
from video_pipeline.core.client.storage.pg import PostgresClient, PgConfig
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


class VideoRegistryTask(BaseTask[VideoInput, VideoArtifact]):
    """Task implementation for video registration and metadata extraction."""

    config = VIDEO_CONFIG

    async def preprocess(self, input_data: VideoInput) -> list[VideoInput]:
        """Wrap single video input into a list for execution."""
        return [input_data]

    async def execute_single(
        self,
        item: VideoInput,
        client: None,
        context: TaskExecutionContext,
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
        logger.info(f"Processing video: {item.video_id} from {item.video_s3_url}")

        _, object_name = parse_s3_url(item.video_s3_url)
        video_extension = extract_extension(item.video_s3_url)

        async with self.minio_client.fetch_object_from_s3(
            s3_url=item.video_s3_url,
            suffix=f".{video_extension}",
        ) as video_tmp_path:
            fps = get_video_fps(video_tmp_path)
            duration = get_video_duration_ffprobe(path=video_tmp_path)

        metadata = {"fps": fps, "extension": video_extension, "duration": duration}

        video_artifact = VideoArtifact(
            artifact_id=item.video_id,
            user_id=item.user_id,
            video_id=item.video_id,
            video_extension=video_extension,
            video_minio_url=item.video_s3_url,
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

    def format_result(self, result: VideoArtifact) -> str:
        """Format VideoArtifact into Markdown for summary artifact.

        Args:
            result: Video artifact to format

        Returns:
            Markdown formatted string
        """
        duration = result.metadata.get("duration") #type:ignore
        fps = result.fps

        return f"""### Video `{result.video_id}`

- **User ID:** `{result.user_id}`
- **Object Name:** `{result.object_name}`
- **Extension:** `{result.video_extension}`
- **FPS:** `{fps}`
- **Duration:** `{duration}` seconds
- **MinIO URL:** `{result.video_minio_url}`
"""


@task(**VIDEO_CONFIG.to_task_kwargs())
async def video_reg_task(
    video_input: VideoInput,
    context: TaskExecutionContext,
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

    logger.debug(f"[VideoRegTask] Postgres config: {settings.postgres=}")
    postgres_client = PostgresClient(
        config=PgConfig( #type:ignore
            database_url=settings.postgres.connection_string,
        )
    )
    await postgres_client.initialize()
    logger.info("[VideoRegTask] Postgres client initialized")

    task_impl = VideoRegistryTask(
        artifact_visitor=ArtifactPersistentVisitor(
            minio_client,
            postgres_client
        ),
        minio_client=minio_client
    )

    all_produced_artifacts: list[VideoArtifact] = []

    logger.info("[VideoRegTask] Preprocessing input...")
    preprocessed = await task_impl.preprocess(input_data=video_input)
    logger.info(f"[VideoRegTask] Preprocessing done — {len(preprocessed)} item(s) to process")

    async for result in task_impl.execute(
        preprocessed=preprocessed,
        client=None,
        context=context,
    ):
        artifact = await task_impl.postprocess(result)
        all_produced_artifacts.append(artifact)
        logger.info(
            f"[VideoRegTask] Artifact persisted | "
            f"video_id={artifact.video_id} fps={artifact.fps} "
            f"ext={artifact.video_extension} duration={artifact.metadata.get('duration')}s"
        )

    await task_impl.create_summary_artifact(all_produced_artifacts, context)
    logger.info(
        f"[VideoRegTask] Complete | {len(all_produced_artifacts)} artifact(s) produced | "
        f"total_time={context.total_time_ms:.0f}ms"
    )

    return all_produced_artifacts[0]
