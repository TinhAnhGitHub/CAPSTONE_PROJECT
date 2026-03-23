"""Combined Image Caption + OCR Task.

A single task that extracts both caption and OCR from images using structured output,
outputting both ImageCaptionArtifact and ImageOCRArtifact.
"""

from __future__ import annotations

import io
import json
import os
import re
from pathlib import Path
from pydantic import SecretStr
from prefect import get_run_logger, task
from prefect.artifacts import acreate_markdown_artifact

from video_pipeline.task.base.base_task import TaskConfig, BaseTask
from video_pipeline.core.client.progress import StageRegistry
from video_pipeline.core.artifact import ImageArtifact, ImageCaptionArtifact, ImageOCRArtifact
from video_pipeline.core.storage.pg_tracker import ArtifactPersistentVisitor
from video_pipeline.core.client.storage.minio import MinioStorageClient
from video_pipeline.core.client.storage.pg.runtime import (
    get_postgres_client,
    shutdown_postgres_client,
)
from video_pipeline.core.client.llm_provider.openrouter import OpenRouterClient, OpenRouterConfig
from video_pipeline.task.kg_graph.models import CostTracker
from video_pipeline.config import get_settings

from .prompt import ImageCaptionOCR, CAPTION_OCR_PROMPT


IMAGE_CAPTION_OCR_CONFIG = TaskConfig.from_yaml("image_caption_ocr")
_base_kwargs = IMAGE_CAPTION_OCR_CONFIG.to_task_kwargs()


_PreprocessedItem = tuple[ImageArtifact, bytes]

_MergedOutput = tuple[ImageCaptionArtifact, bytes, ImageOCRArtifact, bytes]


@StageRegistry.register
class ImageCaptionOCRTask(BaseTask[list[ImageArtifact], tuple[list[ImageCaptionArtifact], list[ImageOCRArtifact], CostTracker]]):
    """Combined Caption + OCR task using structured output.

    preprocess() downloads image bytes from MinIO.
    execute() sends images to OpenRouter with structured output for caption + OCR.
    postprocess() uploads both caption and OCR JSONs to MinIO.
    """

    config = IMAGE_CAPTION_OCR_CONFIG

    async def preprocess(self, input_data: list[ImageArtifact]) -> list[_PreprocessedItem]:
        """Download image bytes for every artifact in the batch."""
        logger = get_run_logger()
        logger.info(f"[ImageCaptionOCR] Downloading {len(input_data)} image(s) from MinIO")

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

        logger.info(f"[ImageCaptionOCR] Preprocessing done — {len(preprocessed)} image(s) ready")
        return preprocessed

    async def execute(
        self,
        preprocessed: list[_PreprocessedItem],
        client: OpenRouterClient,
    ) -> tuple[list[_MergedOutput], CostTracker]:
        """Process images with structured output for caption + OCR.

        Args:
            preprocessed: list of (ImageArtifact, frame_bytes)
            client: OpenRouter inference client.

        Returns:
            Tuple of (list of (ImageCaptionArtifact, caption_json, ImageOCRArtifact, ocr_json), CostTracker)
        """
        logger = get_run_logger()
        max_concurrent: int = IMAGE_CAPTION_OCR_CONFIG.additional_kwargs.get("max_concurrent", 10)
        model = IMAGE_CAPTION_OCR_CONFIG.additional_kwargs.get("model", "google/gemini-2.5-flash-lite")

        logger.info(
            f"[ImageCaptionOCR] Processing {len(preprocessed)} frame(s) | max_concurrent={max_concurrent}"
        )

        data_bytes = [ frame_bytes for _, frame_bytes in preprocessed ]


        structured_llm = client.as_structured_llm(ImageCaptionOCR)
        cost_tracker = CostTracker(model=model)

        import asyncio
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_single(
            artifact: ImageArtifact, image_byte: bytes
        ) -> tuple[ImageCaptionOCR, dict] | None:
            async with semaphore:
                try:
                    from langchain_core.messages import HumanMessage

                    data_url = client.encode_image_bytes(image_byte)
                    msg = HumanMessage(
                        content=[
                            {"type": "image_url", "image_url": {"url": data_url}},
                            {"type": "text", "text": CAPTION_OCR_PROMPT},
                        ]
                    )
                    result, usage = await structured_llm([msg])

                    return result, usage
                except Exception as e:
                    logger.error(f"[ImageCaptionOCR] Failed for frame {artifact.frame_index}: {e}")
                    return None

        tasks = [
            process_single(artifact, image_byte)
            for (artifact, _), image_byte in zip(preprocessed, data_bytes)
        ]
        results = await asyncio.gather(*tasks)

        output: list[_MergedOutput] = []

        for (image_artifact, _), result_data in zip(preprocessed, results):
            if result_data is None:
                caption_result = ImageCaptionOCR(caption="", ocr_texts=[])
                usage = {}
            else:
                caption_result, usage = result_data

            # Track cost
            cost_tracker.add_usage(
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                cost=usage.get("cost", 0.0),
            )

            logger.info(
                f"[ImageCaptionOCR] Done | frame={image_artifact.frame_index} "
                f"caption_len={len(caption_result.caption)} ocr_items={len(caption_result.ocr_texts)} "
                f"| tokens={usage.get('total_tokens', '?')} cost=${usage.get('cost', 0.0):.6f}"
            )

            caption_payload = {
                "caption": caption_result.caption,
                "frame_index": image_artifact.frame_index,
                "timestamp": image_artifact.timestamp,
                "image_minio_url": image_artifact.minio_url_path,
                "usage": usage,
            }
            caption_artifact = ImageCaptionArtifact(
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
                    f"caption/image/{image_artifact.related_video_id}/"
                    f"{image_artifact.frame_index:08d}_{image_artifact.timestamp}.json"
                ),
                metadata={**caption_payload, "usage": usage},
            )

            ocr_text = "\n".join(caption_result.ocr_texts) if caption_result.ocr_texts else ""
            ocr_payload = {
                "ocr_text": ocr_text,
                "ocr_texts": caption_result.ocr_texts,
                "frame_index": image_artifact.frame_index,
                "timestamp": image_artifact.timestamp,
                "image_minio_url": image_artifact.minio_url_path,
                "usage": usage,
            }
            ocr_artifact = ImageOCRArtifact(
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
                metadata={"ocr_text": ocr_text, "ocr_texts": caption_result.ocr_texts, "usage": usage},
            )

            output.append((
                caption_artifact,
                json.dumps(caption_payload).encode(),
                ocr_artifact,
                json.dumps(ocr_payload).encode(),
            ))

        logger.info(
            f"[ImageCaptionOCR] Batch done — {len(output)} result(s) | "
            f"total_prompt={cost_tracker.total_prompt_tokens} "
            f"total_completion={cost_tracker.total_completion_tokens} "
            f"total_cost=${cost_tracker.total_cost:.6f}"
        )
        return output, cost_tracker

    async def postprocess(
        self, result: tuple[list[_MergedOutput], CostTracker]
    ) -> tuple[list[ImageCaptionArtifact], list[ImageOCRArtifact], CostTracker]:
        """Upload caption and OCR JSONs to MinIO and persist to Postgres."""
        merged_outputs, cost_tracker = result
        caption_artifacts: list[ImageCaptionArtifact] = []
        ocr_artifacts: list[ImageOCRArtifact] = []

        for caption_artifact, caption_bytes, ocr_artifact, ocr_bytes in merged_outputs:
            await self.artifact_visitor.visit_artifact(
                caption_artifact, upload_to_minio=io.BytesIO(caption_bytes)
            )
            await self.artifact_visitor.visit_artifact(
                ocr_artifact, upload_to_minio=io.BytesIO(ocr_bytes)
            )
            caption_artifacts.append(caption_artifact)
            ocr_artifacts.append(ocr_artifact)

        return caption_artifacts, ocr_artifacts, cost_tracker

    @staticmethod
    async def summary_artifact(
        final_result: tuple[list[ImageCaptionArtifact], list[ImageOCRArtifact], CostTracker],
    ) -> None:
        """Create Prefect artifacts summarizing the batch."""
        caption_results, ocr_results, cost_tracker = final_result
        if not caption_results:
            return

        first = caption_results[0]
        video_id = first.related_video_id
        raw_key = f"image-caption-ocr-{video_id}".lower()
        key = re.sub(r"[^a-z0-9-]", "-", raw_key)

        avg_cost = cost_tracker.total_cost / max(len(caption_results), 1)


        frames_with_text = sum(
            1 for a in ocr_results if (a.metadata or {}).get("ocr_text", "").strip()
        )
        total_ocr_chars = sum(
            len((a.metadata or {}).get("ocr_text", "")) for a in ocr_results
        )

        markdown = (
            f"# Image Caption + OCR Summary\n\n"
            f"| Field | Value |\n"
            f"|-------|-------|\n"
            f"| **Video ID** | `{video_id}` |\n"
            f"| **Frames Processed** | `{len(caption_results)}` |\n"
            f"| **Frames with OCR Text** | `{frames_with_text}` |\n"
            f"| **Total OCR Characters** | `{total_ocr_chars:,}` |\n"
            f"| **Model** | `{cost_tracker.model}` |\n"
            f"| **LLM Calls** | `{cost_tracker.llm_calls}` |\n"
            f"| **Prompt Tokens** | `{cost_tracker.total_prompt_tokens:,}` |\n"
            f"| **Completion Tokens** | `{cost_tracker.total_completion_tokens:,}` |\n"
            f"| **Total Cost** | `${cost_tracker.total_cost:.6f}` |\n"
            f"| **Avg Cost / Frame** | `${avg_cost:.6f}` |\n"
        )

        await acreate_markdown_artifact(
            key=key,
            markdown=markdown,
            description=f"Image Caption + OCR summary for video {video_id}",
        )


@task(**{**_base_kwargs, "name": "Image Caption OCR Chunk"})  # type: ignore
async def image_caption_ocr_chunk_task(
    items: list[ImageArtifact],
) -> tuple[list[ImageCaptionArtifact], list[ImageOCRArtifact], CostTracker]:
    """Process a batch of image frames for caption + OCR.

    Args:
        items: Batch of ImageArtifacts to process.

    Returns:
        Tuple of (caption_artifacts, ocr_artifacts, cost_tracker).
    """
    logger = get_run_logger()
    settings = get_settings()

    logger.info(f"[ImageCaptionOCRChunk] Starting | {len(items)} frame(s) in batch")

    minio_client = MinioStorageClient(
        endpoint=settings.minio.endpoint,
        access_key=settings.minio.access_key,
        secret_key=settings.minio.secret_key,
        secure=settings.minio.secure,
    )
    postgres_client = await get_postgres_client()
    logger.info(f"[ImageCaptionOCRChunk] Clients initialized | minio={settings.minio.endpoint}")

    openrouter_config = OpenRouterConfig(
        api_key=SecretStr(os.environ.get("OPENROUTER_API_KEY", "")),
        model=IMAGE_CAPTION_OCR_CONFIG.additional_kwargs["model"],
        base_url=IMAGE_CAPTION_OCR_CONFIG.additional_kwargs["base_url"],
    )
    logger.info(f"[ImageCaptionOCRChunk] OpenRouter config | model={openrouter_config.model}")

    task_impl = ImageCaptionOCRTask(
        artifact_visitor=ArtifactPersistentVisitor(minio_client, postgres_client),
        minio_client=minio_client,
    )
    client = OpenRouterClient(config=openrouter_config)

    try:
        caption_artifacts, ocr_artifacts, cost_tracker = await task_impl.execute_template(items, client)
    finally:
        await client.close()
        await shutdown_postgres_client(postgres_client)

    logger.info(
        f"[ImageCaptionOCRChunk] Done | {len(caption_artifacts)} caption(s), {len(ocr_artifacts)} OCR(s) | "
        f"total_cost=${cost_tracker.total_cost:.6f}"
    )
    return caption_artifacts, ocr_artifacts, cost_tracker