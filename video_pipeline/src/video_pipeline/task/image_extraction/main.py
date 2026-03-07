from __future__ import annotations

import io
import re

from prefect import get_run_logger, task
from prefect.artifacts import acreate_markdown_artifact, acreate_table_artifact

from video_pipeline.task.base.base_task import TaskConfig, BaseTask
from video_pipeline.core.client.progress import StageRegistry
from video_pipeline.core.artifact import AutoshotArtifact, ImageArtifact
from video_pipeline.core.storage.pg_tracker import ArtifactPersistentVisitor
from video_pipeline.core.client.storage.minio import MinioStorageClient
from video_pipeline.core.client.storage.pg import PostgresClient, PgConfig
from video_pipeline.config import get_settings

from .helper import FastFrameReader, frames_to_timestamp


IMAGE_CONFIG = TaskConfig.from_yaml("image_extraction")
_base_kwargs = IMAGE_CONFIG.to_task_kwargs()

ImageItem = tuple[AutoshotArtifact, int]

_PreprocessedImageItem = tuple[AutoshotArtifact, int, bytes]


@StageRegistry.register
class ImageExtractionTask(BaseTask[list[ImageItem], list[ImageArtifact]]):
    """Extract representative frames from a batch of segments and persist as ImageArtifacts.

    preprocess() downloads the video once and decodes all requested frames into
    bytes, so execute_single() works purely in memory with no further I/O.
    """

    config = IMAGE_CONFIG

    async def preprocess(self, input_data: list[ImageItem]) -> list[_PreprocessedImageItem]:
        """Download the video once and extract frame bytes for every item in the batch."""
        logger = get_run_logger()
        autoshot_artifact = input_data[0][0]
        video_url = autoshot_artifact.related_video_minio_url
        video_extension = autoshot_artifact.related_video_extension

        logger.info(
            f"[ImageExtractionTask] Preprocessing {len(input_data)} frame(s) | "
            f"video={video_url}"
        )

        preprocessed: list[_PreprocessedImageItem] = []
        async with self.minio_client.fetch_object_from_s3(
            s3_url=video_url,
            suffix=f".{video_extension}",
        ) as video_path:
            with open(str(video_path), "rb") as f:
                video_bytes = f.read()

            reader = FastFrameReader(video_bytes)
            for artifact, frame_index in input_data:
                frame_bytes = reader.get_frame(frame_index)
                preprocessed.append((artifact, frame_index, frame_bytes))

        logger.info(
            f"[ImageExtractionTask] Preprocessing done — {len(preprocessed)} frame(s) extracted"
        )
        return preprocessed

    async def execute(
        self,
        preprocessed: _PreprocessedImageItem,
        client: None,
    ) -> tuple[ImageArtifact, bytes]:
        """Build an ImageArtifact for one frame.

        Args:
            item: (AutoshotArtifact, frame_index, frame_bytes)
            client: Unused — no inference server needed for image extraction.
            context: Task execution context.

        Returns:
            (ImageArtifact, frame_bytes) ready for postprocess to upload and persist.
        """
        logger = get_run_logger()
        autoshot_artifact, frame_index, frame_bytes = preprocessed
        fps = autoshot_artifact.related_video_fps
        timestamp = frames_to_timestamp(frame_index, fps)
        artifact = ImageArtifact(
            frame_index=frame_index,
            extension=".webp",
            related_video_id=autoshot_artifact.related_video_id,
            related_video_minio_url=autoshot_artifact.related_video_minio_url,
            related_video_extension=autoshot_artifact.related_video_extension,
            related_video_fps=fps,
            timestamp=timestamp,
            autoshot_artifact_id=str(autoshot_artifact.artifact_id),
            user_id=autoshot_artifact.user_id,
            content_type="image/webp",
            object_name=f"images/{autoshot_artifact.related_video_id}/{frame_index:08d}_{timestamp}.webp",
        )
        return artifact, frame_bytes

    async def postprocess(self, result: tuple[ImageArtifact, bytes]) -> ImageArtifact:  # type: ignore[override]
        """Upload frame bytes to MinIO and persist artifact metadata to Postgres."""
        artifact, frame_bytes = result
        await self.artifact_visitor.visit_artifact(
            artifact, upload_to_minio=io.BytesIO(frame_bytes)
        )
        return artifact

    def format_result(self, result: ImageArtifact) -> str:
        return f"""### Image Frame

- **Frame Index:** {result.frame_index}
- **Timestamp:** {result.timestamp}
- **MinIO URL:** `{result.minio_url_path}`
- **Autoshot Artifact ID:** `{result.autoshot_artifact_id}`
"""

    @staticmethod
    async def summary_artifact(final_result: list[ImageArtifact]) -> None:
        """Create a Prefect markdown artifact summarising an image extraction batch."""
        if not final_result:
            return

        first = final_result[0]
        raw_key = f"image-extraction-{first.related_video_id}".lower()
        key = re.sub(r"[^a-z0-9-]", "-", raw_key)

        frame_rows = ""
        for artifact in final_result:
            frame_rows += (
                f"| {artifact.frame_index} | {artifact.timestamp} "
                f"| `{artifact.minio_url_path}` | `{artifact.artifact_id}` |\n"
            )

        markdown = (
f"# Image Extraction Summary\n\n"
f"| Field | Value |\n"
f"|-------|-------|\n"
f"| **Related Video ID** | `{first.related_video_id}` |\n"
f"| **Autoshot Artifact ID** | `{first.autoshot_artifact_id}` |\n"
f"| **User ID** | `{first.user_id}` |\n"
f"| **FPS** | `{first.related_video_fps}` |\n"
f"| **Frames Extracted** | `{len(final_result)}` |\n\n"
f"## Extracted Frames\n\n"
f"| Frame Index | Timestamp | MinIO URL | Artifact ID |\n"
f"|-------------|-----------|-----------|-------------|\n"
f"{frame_rows}"
        )

        await acreate_markdown_artifact(
            key=key,
            markdown=markdown,
            description=f"Image extraction summary for video {first.related_video_id}",
        )

        await acreate_table_artifact(
            table=[
                {"Field": "Related Video ID", "Value": str(first.related_video_id)},
                {"Field": "Autoshot Artifact ID", "Value": str(first.autoshot_artifact_id)},
                {"Field": "User ID", "Value": str(first.user_id)},
                {"Field": "FPS", "Value": str(first.related_video_fps)},
                {"Field": "Frames Extracted", "Value": str(len(final_result))},
            ],
            key=f"{key}-summary-table",
            description=f"Image extraction stats table for video {first.related_video_id}",
        )

        await acreate_table_artifact(
            table=[
                {
                    "Frame Index": artifact.frame_index,
                    "Timestamp": artifact.timestamp,
                    "MinIO URL": str(artifact.minio_url_path),
                    "Artifact ID": str(artifact.artifact_id),
                }
                for artifact in final_result
            ],
            key=f"{key}-frames-table",
            description=f"Extracted frames for video {first.related_video_id}",
        )


@task(**{**_base_kwargs, "name": "Image Chunk"})  # type: ignore
async def image_chunk_task(
    items: list[ImageItem],
) -> list[ImageArtifact]:
    """Extract and persist a batch of frames from the video.

    Downloads the video once in preprocess(), then for each frame builds an
    ImageArtifact, uploads it to MinIO, and persists metadata to Postgres.

    Args:
        items: Batch of (AutoshotArtifact, frame_index) tuples.

    Returns:
        List of ImageArtifacts, one per frame in the batch.
    """
    logger = get_run_logger()
    settings = get_settings()

    logger.info(f"[ImageChunk] Starting | {len(items)} frame(s) in batch")

    minio_client = MinioStorageClient(
        endpoint=settings.minio.endpoint,
        access_key=settings.minio.access_key,
        secret_key=settings.minio.secret_key,
        secure=settings.minio.secure,
    )
    postgres_client = PostgresClient(
        config=PgConfig(database_url=settings.postgres.connection_string)  # type: ignore
    )
    task_impl = ImageExtractionTask(
        artifact_visitor=ArtifactPersistentVisitor(minio_client, postgres_client),
        minio_client=minio_client,
    )

    artifacts = []
    preprocessed = await task_impl.preprocess(items)
    for preproc in preprocessed:
        result = await task_impl.execute(preproc, None)
        artifact = await task_impl.postprocess(result)
        artifacts.append(artifact)    
   
    logger.info(f"[ImageChunk] Done | {len(artifacts)} artifact(s) produced")
    return artifacts
