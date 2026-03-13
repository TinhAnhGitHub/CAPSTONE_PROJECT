from __future__ import annotations

import io
import json
import os
import tempfile
from pathlib import Path

import re

from prefect import get_run_logger, task
from prefect.artifacts import acreate_markdown_artifact, acreate_table_artifact

from video_pipeline.task.base.base_task import TaskConfig, BaseTask
from video_pipeline.core.client.progress import StageRegistry
from video_pipeline.core.artifact import ImageArtifact, ImageOCRArtifact
from video_pipeline.core.storage.pg_tracker import ArtifactPersistentVisitor
from video_pipeline.core.client.storage.minio import MinioStorageClient
from video_pipeline.core.client.storage.pg.runtime import get_postgres_client, shutdown_postgres_client
from video_pipeline.core.client.inference.ocr_client import LightONOCRClient, LightONOCRConfig
from video_pipeline.config import get_settings

from .helper import create_image_tmp_file

IMAGE_OCR_CONFIG = TaskConfig.from_yaml("image_ocr")
_base_kwargs = IMAGE_OCR_CONFIG.to_task_kwargs()

_PreprocessedItem = tuple[ImageArtifact, bytes]


@StageRegistry.register
class ImageOCRTask(BaseTask[list[ImageArtifact], list[ImageOCRArtifact]]):
    """Run OCR on extracted video frames using LightON OCR in batch.

    preprocess() downloads WEBP image bytes from MinIO for the whole batch.
    execute_single() writes frames to temp files, calls client.ainfer() concurrently,
    and builds ImageOCRArtifacts.
    postprocess() uploads OCR JSON to MinIO and persists to Postgres.
    """

    config = IMAGE_OCR_CONFIG

    async def preprocess(self, input_data: list[ImageArtifact]) -> list[_PreprocessedItem]:
        """Download image bytes for every artifact in the batch."""
        logger = get_run_logger()
        logger.info(f"[ImageOCRTask] Downloading {len(input_data)} image(s) from MinIO")

        preprocessed: list[_PreprocessedItem] = []
        for artifact in input_data:
            assert artifact.object_name is not None, (
                f"ImageArtifact {artifact.artifact_id} has no object_name"
            )
            image_bytes = self.minio_client.get_object_bytes(
                bucket=artifact.user_id,
                object_name=artifact.object_name,
            )
            preprocessed.append((artifact, image_bytes))

        logger.info(f"[ImageOCRTask] Preprocessing done — {len(preprocessed)} image(s) ready")
        return preprocessed

    async def execute(
        self,
        preprocessed: list[_PreprocessedItem],
        client: LightONOCRClient,
    ) -> list[tuple[ImageOCRArtifact, bytes]]:
        """OCR a batch of image frames concurrently via LightON.

        Writes each WEBP frame to a NamedTemporaryFile so the OCR client can
        read and encode it. Temp files are cleaned up in a finally block.

        Args:
            item: list of (ImageArtifact, frame_bytes) — the full batch.
            client: LightON OCR client.
            context: Task execution context.

        Returns:
            list of (ImageOCRArtifact, json_bytes) ready for postprocess.
        """
        logger = get_run_logger()
        logger.info(f"[ImageOCRTask] Batch OCR on {len(preprocessed)} frame(s)")

        temp_paths: list[Path] = []
        try:
            for _, image_bytes in preprocessed:
                tmp_path = create_image_tmp_file(image_bytes)
                temp_paths.append(tmp_path)
            ocr_results: list[str] = await client.ainfer(temp_paths)  #type:ignore
        finally:
            for p in temp_paths:
                p.unlink(missing_ok=True)

        output: list[tuple[ImageOCRArtifact, bytes]] = []
        for (image_artifact, _), ocr_text in zip(preprocessed, ocr_results):
            text = ocr_text or ""
            logger.info(
                f"[ImageOCRTask] OCR done | frame={image_artifact.frame_index} "
                f"chars={len(text)} preview={text[:60]!r}"
            )

            artifact = ImageOCRArtifact(
                frame_index=image_artifact.frame_index,
                time_stamp=image_artifact.timestamp,
                related_video_id=image_artifact.related_video_id,
                related_video_fps=image_artifact.related_video_fps,
                extension=".json",
                image_minio_url=image_artifact.minio_url_path,
                image_id=image_artifact.artifact_id,
                user_id=image_artifact.user_id,
                object_name=(
                    f"ocr/image/{image_artifact.related_video_id}/"
                    f"{image_artifact.frame_index:08d}_{image_artifact.timestamp}.json"
                ),
                metadata={"ocr_text": text},
            )

            ocr_payload = json.dumps({
                "ocr_text": text,
                "frame_index": image_artifact.frame_index,
                "timestamp": image_artifact.timestamp,
                "image_minio_url": image_artifact.minio_url_path,
            })
            output.append((artifact, ocr_payload.encode()))

        logger.info(f"[ImageOCRTask] Batch done — {len(output)} OCR result(s) produced")
        return output

    async def postprocess(self, result: list[tuple[ImageOCRArtifact, bytes]]) -> list[ImageOCRArtifact]:  # type: ignore[override]
        """Upload OCR JSONs to MinIO and persist artifact metadata to Postgres."""
        artifacts: list[ImageOCRArtifact] = []
        for artifact, json_bytes in result:
            await self.artifact_visitor.visit_artifact(
                artifact, upload_to_minio=io.BytesIO(json_bytes)
            )
            artifacts.append(artifact)
        return artifacts

    def format_result(self, result: ImageOCRArtifact) -> str:
        meta = result.metadata or {}
        text = meta.get("ocr_text", "")
        return (
            f"### Frame {result.frame_index} — {result.time_stamp}\n\n"
            f"- **Image URL:** `{result.image_minio_url}`\n"
            f"- **OCR Text ({len(text)} chars):** {text[:300]!r}\n"
        )

    @staticmethod
    async def summary_artifact(final_result: list[ImageOCRArtifact]) -> None:
        """Create a Prefect markdown artifact summarising an OCR batch."""
        if not final_result:
            return

        first = final_result[0]
        video_id = first.related_video_id
        key = re.sub(r"[^a-z0-9-]", "-", f"image-ocr-{video_id}".lower())

        frames_with_text = sum(
            1 for a in final_result
            if (a.metadata or {}).get("ocr_text", "").strip()
        )
        total_chars = sum(
            len((a.metadata or {}).get("ocr_text", ""))
            for a in final_result
        )

        ocr_rows = ""
        for artifact in final_result:
            text = (artifact.metadata or {}).get("ocr_text", "")
            display = (text[:100] + "…") if len(text) > 100 else (text or "_empty_")
            ocr_rows += (
                f"| {artifact.frame_index} | {artifact.time_stamp} "
                f"| {len(text)} | {display} |\n"
            )

        markdown = (
f"# Image OCR Summary\n\n"
f"| Field | Value |\n"
f"|-------|-------|\n"
f"| **Video ID** | `{video_id}` |\n"
f"| **User ID** | `{first.user_id}` |\n"
f"| **FPS** | `{first.related_video_fps}` |\n"
f"| **Frames Processed** | `{len(final_result)}` |\n"
f"| **Frames with Text** | `{frames_with_text}` |\n"
f"| **Total Characters** | `{total_chars:,}` |\n\n"
f"## OCR Results\n\n"
f"| Frame | Timestamp | Chars | OCR Text |\n"
f"|-------|-----------|-------|----------|\n"
f"{ocr_rows}"
        )

        await acreate_markdown_artifact(
            key=key,
            markdown=markdown,
            description=f"OCR summary for video {video_id}",
        )

        await acreate_table_artifact(
            table=[
                {"Field": "Video ID", "Value": str(video_id)},
                {"Field": "User ID", "Value": str(first.user_id)},
                {"Field": "FPS", "Value": str(first.related_video_fps)},
                {"Field": "Frames Processed", "Value": str(len(final_result))},
                {"Field": "Frames with Text", "Value": str(frames_with_text)},
                {"Field": "Total Characters", "Value": f"{total_chars:,}"},
            ],
            key=f"{key}-summary-table",
            description=f"OCR stats table for video {video_id}",
        )

        await acreate_table_artifact(
            table=[
                {
                    "Frame": artifact.frame_index,
                    "Timestamp": artifact.time_stamp,
                    "Chars": len((artifact.metadata or {}).get("ocr_text", "")),
                    "OCR Text": (
                        lambda t: (t[:100] + "…") if len(t) > 100 else (t or "_empty_")
                    )((artifact.metadata or {}).get("ocr_text", "")),
                }
                for artifact in final_result
            ],
            key=f"{key}-results-table",
            description=f"OCR results for video {video_id}",
        )


@task(**{**_base_kwargs, "name": "Image OCR Chunk"})  # type: ignore
async def image_ocr_chunk_task(
    items: list[ImageArtifact],
) -> list[ImageOCRArtifact]:
    """OCR a batch of image frames using ImageOCRTask.execute().

    Downloads image bytes in preprocess(), writes to temp files, then calls
    execute_single() once with the whole batch via LightON OCR concurrently.

    Args:
        items: Batch of ImageArtifacts to OCR.
        context: Task execution context.

    Returns:
        List of ImageOCRArtifacts, one per frame in the batch.
    """
    logger = get_run_logger()
    settings = get_settings()

    logger.info(f"[ImageOCRChunk] Starting | {len(items)} frame(s) in batch")

    minio_client = MinioStorageClient(
        endpoint=settings.minio.endpoint,
        access_key=settings.minio.access_key,
        secret_key=settings.minio.secret_key,
        secure=settings.minio.secure,
    )
    postgres_client = await get_postgres_client()
    logger.info(f"[ImageOCRChunk] Clients initialized | minio={settings.minio.endpoint}")

    ocr_config = LightONOCRConfig(
        model_name=IMAGE_OCR_CONFIG.additional_kwargs.get("model_name", "ocr_lighton"),
        base_url=IMAGE_OCR_CONFIG.additional_kwargs.get("base_url", "http://ocr_lighton:8000"),
    )
    logger.info(f"[ImageOCRChunk] OCR config | model={ocr_config.model_name} url={ocr_config.base_url}")

    task_impl = ImageOCRTask(
        artifact_visitor=ArtifactPersistentVisitor(minio_client, postgres_client),
        minio_client=minio_client,
    )
    client = LightONOCRClient(config=ocr_config)

    try:
        all_artifacts = await task_impl.execute_template(items, client)
    finally:
        await client.close()
        await shutdown_postgres_client(postgres_client)

    logger.info(f"[ImageOCRChunk] Done | {len(all_artifacts)} artifact(s) produced")
    return all_artifacts
