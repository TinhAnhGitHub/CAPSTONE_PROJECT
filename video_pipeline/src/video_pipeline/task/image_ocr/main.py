from __future__ import annotations

import io
import json
import os
import re
from pathlib import Path

from prefect import get_run_logger, task
from prefect.artifacts import acreate_markdown_artifact, acreate_table_artifact

from video_pipeline.task.base.base_task import TaskConfig, BaseTask
from video_pipeline.core.client.progress import StageRegistry
from video_pipeline.core.artifact import ImageArtifact, ImageOCRArtifact
from video_pipeline.core.storage.pg_tracker import ArtifactPersistentVisitor
from video_pipeline.core.client.storage.minio import MinioStorageClient
from video_pipeline.core.client.storage.pg.runtime import (
    get_postgres_client,
    shutdown_postgres_client,
)
from video_pipeline.core.client.llm_provider.openrouter import OpenRouterClient, OpenRouterConfig
from video_pipeline.config import get_settings

from .prompt import OCR_PROMPT

IMAGE_OCR_CONFIG = TaskConfig.from_yaml("image_ocr")
_base_kwargs = IMAGE_OCR_CONFIG.to_task_kwargs()

_PreprocessedItem = tuple[ImageArtifact, bytes]


def normalize_ocr_text(text: str) -> str:
    t = text.strip()
    return t


@StageRegistry.register
class ImageOCRTask(BaseTask[list[ImageArtifact], list[ImageOCRArtifact]]):
    """Run OCR on extracted video frames using OpenRouter VLM in batch.

    preprocess() downloads image bytes from MinIO for the whole batch.
    execute() sends images to OpenRouter concurrently for OCR.
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
        client: OpenRouterClient,
    ) -> list[tuple[ImageOCRArtifact, bytes]]:
        """OCR a batch of image frames concurrently via OpenRouter VLM.

        Args:
            preprocessed: list of (ImageArtifact, frame_bytes) — the full batch.
            client: OpenRouter inference client.

        Returns:
            list of (ImageOCRArtifact, json_bytes) ready for postprocess.
        """
        logger = get_run_logger()
        max_concurrent: int = IMAGE_OCR_CONFIG.additional_kwargs.get("max_concurrent", 10)

        logger.info(
            f"[ImageOCRTask] Batch OCR on {len(preprocessed)} frame(s) | max_concurrent={max_concurrent}"
        )

        # Prepare image data URIs
        data_uris = []
        mime_map = {
            ".webp": "image/webp",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
        }
        for artifact, frame_bytes in preprocessed:
            ext = Path(artifact.object_name or "").suffix.lower() if artifact.object_name else ".webp"
            mime = mime_map.get(ext, "image/webp")
            data_uris.append(OpenRouterClient.encode_image_bytes(frame_bytes, mime=mime))

        results = await client.batch_ainfer_image(
            data_uris,
            prompts=OCR_PROMPT,
            max_concurrent=max_concurrent,
        )

        if results is None:
            raise RuntimeError("OpenRouter batch_ainfer_image returned None — API call failed")

        output: list[tuple[ImageOCRArtifact, bytes]] = []
        total_usage: dict[str, float] = {
            "prompt_tokens": 0.0,
            "completion_tokens": 0.0,
            "total_tokens": 0.0,
            "cost": 0.0,
        }

        for (image_artifact, _), result in zip(preprocessed, results):
            if result is None:
                raise RuntimeError(
                    f"OpenRouter returned None for frame {image_artifact.frame_index} — API call failed"
                )

            raw_text = normalize_ocr_text(result.content or "")
            usage: dict = result.usage or {} if result else {}

            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                total_usage[key] += usage.get(key, 0)
            total_usage["cost"] += usage.get("cost", 0.0)

            logger.info(
                f"[ImageOCRTask] OCR done | frame={image_artifact.frame_index} "
                f"chars={len(raw_text)} preview={raw_text[:60]!r} | "
                f"tokens={usage.get('total_tokens', '?')} cost=${usage.get('cost', 0.0):.6f}"
            )

            artifact = ImageOCRArtifact(
                frame_index=image_artifact.frame_index,
                timestamp=image_artifact.timestamp,
                timestamp_sec=image_artifact.timestamp_sec,
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
                metadata={"ocr_text": raw_text, "usage": usage},
            )

            ocr_payload = json.dumps(
                {
                    "ocr_text": raw_text,
                    "frame_index": image_artifact.frame_index,
                    "timestamp": image_artifact.timestamp,
                    "image_minio_url": image_artifact.minio_url_path,
                    "usage": usage,
                }
            )
            output.append((artifact, ocr_payload.encode()))

        logger.info(
            f"[ImageOCRTask] Batch done — {len(output)} OCR result(s) | "
            f"total_prompt={int(total_usage['prompt_tokens'])} "
            f"total_completion={int(total_usage['completion_tokens'])} "
            f"total_tokens={int(total_usage['total_tokens'])} "
            f"total_cost=${total_usage['cost']:.6f}"
        )
        return output

    async def postprocess(
        self, result: list[tuple[ImageOCRArtifact, bytes]]
    ) -> list[ImageOCRArtifact]:  # type: ignore[override]
        """Upload OCR JSONs to MinIO and persist artifact metadata to Postgres."""
        artifacts: list[ImageOCRArtifact] = []
        for artifact, json_bytes in result:
            await self.artifact_visitor.visit_artifact(
                artifact, upload_to_minio=io.BytesIO(json_bytes)
            )
            artifacts.append(artifact)
        return artifacts

    @staticmethod
    async def summary_artifact(final_result: list[ImageOCRArtifact]) -> None:
        """Create a Prefect markdown artifact summarising an OCR batch."""
        if not final_result:
            return

        first = final_result[0]
        video_id = first.related_video_id
        key = re.sub(r"[^a-z0-9-]", "-", f"image-ocr-{video_id}".lower())

        frames_with_text = sum(
            1 for a in final_result if (a.metadata or {}).get("ocr_text", "").strip()
        )
        total_chars = sum(len((a.metadata or {}).get("ocr_text", "")) for a in final_result)

        total_prompt = total_completion = total_tokens = 0
        total_cost = 0.0
        for a in final_result:
            u = (a.metadata or {}).get("usage") or {}
            total_prompt += u.get("prompt_tokens", 0)
            total_completion += u.get("completion_tokens", 0)
            total_tokens += u.get("total_tokens", 0)
            total_cost += u.get("cost", 0.0)

        ocr_rows = ""
        for artifact in final_result:
            text = (artifact.metadata or {}).get("ocr_text", "")
            display = (text[:100] + "…") if len(text) > 100 else (text or "_empty_")
            ocr_rows += (
                f"| {artifact.frame_index} | {artifact.timestamp} | {len(text)} | {display} |\n"
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
            f"| **Total Characters** | `{total_chars:,}` |\n"
            f"| **Total Prompt Tokens** | `{total_prompt:,}` |\n"
            f"| **Total Completion Tokens** | `{total_completion:,}` |\n"
            f"| **Total Tokens** | `{total_tokens:,}` |\n"
            f"| **Total Cost** | `${total_cost:.6f}` |\n\n"
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
                {"Field": "Total Cost", "Value": f"${total_cost:.6f}"},
            ],
            key=f"{key}-summary-table",
            description=f"OCR stats table for video {video_id}",
        )


@task(**{**_base_kwargs, "name": "Image OCR Chunk"})  # type: ignore
async def image_ocr_chunk_task(
    items: list[ImageArtifact],
) -> list[ImageOCRArtifact]:
    """OCR a batch of image frames using ImageOCRTask.execute().

    Downloads image bytes in preprocess(), sends to OpenRouter concurrently for OCR.

    Args:
        items: Batch of ImageArtifacts to OCR.

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

    openrouter_config = OpenRouterConfig(
        api_key=os.environ.get("OPENROUTER_API_KEY", ""),
        model=IMAGE_OCR_CONFIG.additional_kwargs["model"],
        base_url=IMAGE_OCR_CONFIG.additional_kwargs["base_url"],
    )
    logger.info(f"[ImageOCRChunk] OpenRouter config | model={openrouter_config.model}")

    task_impl = ImageOCRTask(
        artifact_visitor=ArtifactPersistentVisitor(minio_client, postgres_client),
        minio_client=minio_client,
    )
    client = OpenRouterClient(config=openrouter_config)

    try:
        all_artifacts = await task_impl.execute_template(items, client)
    finally:
        await client.close()
        await shutdown_postgres_client(postgres_client)

    logger.info(f"[ImageOCRChunk] Done | {len(all_artifacts)} artifact(s) produced")
    return all_artifacts