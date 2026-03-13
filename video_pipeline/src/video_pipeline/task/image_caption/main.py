from __future__ import annotations

import io
import re
from llm_json import json
import os

from prefect import get_run_logger, task
from prefect.artifacts import acreate_markdown_artifact, acreate_image_artifact, acreate_table_artifact

from video_pipeline.task.base.base_task import TaskConfig, BaseTask
from video_pipeline.core.client.progress import StageRegistry
from video_pipeline.core.artifact import ImageArtifact, ImageCaptionArtifact
from video_pipeline.core.storage.pg_tracker import ArtifactPersistentVisitor
from video_pipeline.core.client.storage.minio import MinioStorageClient
from video_pipeline.core.client.storage.pg.runtime import get_postgres_client, shutdown_postgres_client
from video_pipeline.core.client.llm_provider.openrouter import OpenRouterClient, OpenRouterConfig
from video_pipeline.config import get_settings

from video_pipeline.task.image_caption.prompt import PROMPT as DEFAULT_PROMPT


IMAGE_CAPTION_CONFIG = TaskConfig.from_yaml("image_caption")
_base_kwargs = IMAGE_CAPTION_CONFIG.to_task_kwargs()



_PreprocessedItem = tuple[ImageArtifact, bytes]


@StageRegistry.register
class ImageCaptionTask(BaseTask[list[ImageArtifact], list[ImageCaptionArtifact]]):
    """Caption extracted frames using a VLM via OpenRouter in batch.

    preprocess() downloads image bytes for the batch from MinIO once.
    execute_single() sends the whole batch to OpenRouter concurrently and builds caption artifacts.
    postprocess() uploads caption JSONs to MinIO and persists to Postgres.
    """

    config = IMAGE_CAPTION_CONFIG

    async def preprocess(self, input_data: list[ImageArtifact]) -> list[_PreprocessedItem]:
        """Download image bytes for every artifact in the batch."""
        logger = get_run_logger()
        logger.info(f"[ImageCaptionTask] Downloading {len(input_data)} image(s) from MinIO")

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

        logger.info(f"[ImageCaptionTask] Preprocessing done — {len(preprocessed)} image(s) ready")
        return preprocessed

    async def execute(
        self,
        preprocessed: list[_PreprocessedItem],
        client: OpenRouterClient,
    ) -> list[tuple[ImageCaptionArtifact, bytes]]:
        """Caption a batch of image frames via OpenRouter VLM concurrently.

        Args:
            item: list of (ImageArtifact, frame_bytes) — the full batch.
            client: OpenRouter inference client.
            context: Task execution context.

        Returns:
            list of (ImageCaptionArtifact, json_bytes) ready for postprocess.
        """
        logger = get_run_logger()
        max_concurrent: int = IMAGE_CAPTION_CONFIG.additional_kwargs.get("batch_size", 10)

        logger.info(f"[ImageCaptionTask] Batch captioning {len(preprocessed)} frame(s) | max_concurrent={max_concurrent}")

        data_uris = [
            OpenRouterClient.encode_image_bytes(frame_bytes, mime="image/webp")
            for _, frame_bytes in preprocessed
        ]
        results = await client.batch_ainfer_image(
            data_uris, prompts=DEFAULT_PROMPT, max_concurrent=max_concurrent #type:ignore
        )

        if results is None:
            raise RuntimeError("OpenRouter batch_ainfer_image returned None — API call failed")

        output: list[tuple[ImageCaptionArtifact, bytes]] = []
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

            dense_caption = result.content or ""
            usage: dict = result.usage or {} if result else {}

            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                total_usage[key] += usage.get(key, 0)
            total_usage["cost"] += usage.get("cost", 0.0)

        
            logger.info(
                f"[ImageCaptionTask] Caption done | frame={image_artifact.frame_index} "
                f"caption={dense_caption[:10]} | tokens={usage.get('total_tokens', '?')} "
                f"cost=${usage.get('cost', 0.0):.6f}"
            )

            caption_payload = {
                "caption": dense_caption,
                "frame_index": image_artifact.frame_index,
                "timestamp": image_artifact.timestamp,
                "image_minio_url": image_artifact.minio_url_path,
                "usage": usage,
            }
            artifact = ImageCaptionArtifact(
                frame_index=image_artifact.frame_index,
                time_stamp=image_artifact.timestamp,
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

            output.append((artifact, json.dumps(caption_payload).encode()))

        logger.info(
            f"[ImageCaptionTask] Batch done — {len(output)} caption(s) | "
            f"total_prompt={int(total_usage['prompt_tokens'])} "
            f"total_completion={int(total_usage['completion_tokens'])} "
            f"total_tokens={int(total_usage['total_tokens'])} "
            f"total_cost=${total_usage['cost']:.6f}"
        )
        return output

    async def postprocess(self, result: list[tuple[ImageCaptionArtifact, bytes]]) -> list[ImageCaptionArtifact]:  # type: ignore[override]
        """Upload caption JSONs to MinIO and persist artifact metadata to Postgres."""
        artifacts: list[ImageCaptionArtifact] = []
        for artifact, json_bytes in result:
            await self.artifact_visitor.visit_artifact(
                artifact, upload_to_minio=io.BytesIO(json_bytes)
            )
            artifacts.append(artifact)
        return artifacts

    def format_result(self, result: ImageCaptionArtifact) -> str:
        meta = result.metadata or {}
        usage = meta.get("usage") or {}
        caption_preview = {k: v for k, v in meta.items() if k != "usage"}
        caption_str = json.dumps(caption_preview, ensure_ascii=False)[:300]
        return (
            f"### Frame {result.frame_index} — {result.time_stamp}\n\n"
            f"- **Image URL:** `{result.image_minio_url}`\n"
            f"- **Tokens:** {usage.get('total_tokens', '?')} "
            f"(prompt={usage.get('prompt_tokens', '?')} / completion={usage.get('completion_tokens', '?')})\n"
            f"- **Cost:** ${usage.get('cost', 0.0):.6f}\n"
            f"- **Data:** {caption_str}\n"
        )

    @staticmethod
    async def summary_artifact(
        final_result: list[ImageCaptionArtifact],
    ) -> None:
        """Create Prefect artifacts: a markdown stats summary + image artifacts per sample frame."""
        if not final_result:
            return

        first = final_result[0]
        video_id = first.related_video_id
        raw_key = f"image-caption-{video_id}".lower()
        key = re.sub(r"[^a-z0-9-]", "-", raw_key)

        settings = get_settings()
        minio_endpoint = settings.minio.endpoint.rstrip("/")
        scheme = "https" if settings.minio.secure else "http"

        def s3_to_http(s3_url: str) -> str:
            """Convert s3://bucket/key  ->  http(s)://endpoint/bucket/key"""
            path = s3_url.removeprefix("s3://")
            return f"{scheme}://{minio_endpoint}/{path}"

        total_prompt = total_completion = total_tokens = 0
        total_cost = 0.0
        for a in final_result:
            u = (a.metadata or {}).get("usage") or {}
            total_prompt += u.get("prompt_tokens", 0)
            total_completion += u.get("completion_tokens", 0)
            total_tokens += u.get("total_tokens", 0)
            total_cost += u.get("cost", 0.0)

        avg_cost = total_cost / max(len(final_result), 1)

        sample_size = min(5, len(final_result))
        step = max(len(final_result) // sample_size, 1)
        samples = final_result[::step][:sample_size]


        gallery_rows = ""
        for artifact in samples:
            meta = artifact.metadata or {}
            caption = meta.get("caption", "")
            display_caption = (caption[:120] + "…") if len(caption) > 120 else caption
            gallery_rows += (
                f"| {artifact.frame_index} | {artifact.time_stamp} "
                f"| {display_caption} |\n"
            )

        model = ImageCaptionTask.config.additional_kwargs["model"]
        markdown = (
f"# Image Caption Summary\n\n"
f"| Field | Value |\n"
f"|-------|-------|\n"
f"| **Video ID** | `{video_id}` |\n"
f"| **Model** | `{model}` |\n"
f"| **Frames Captioned** | `{len(final_result)}` |\n"
f"| **Prompt Tokens** | `{total_prompt:,}` |\n"
f"| **Completion Tokens** | `{total_completion:,}` |\n"
f"| **Total Tokens** | `{total_tokens:,}` |\n"
f"| **Total Cost** | `${total_cost:.6f}` |\n"
f"| **Avg Cost / Frame** | `${avg_cost:.6f}` |\n\n"
f"## Caption Gallery (sample {len(samples)} of {len(final_result)} frames)\n\n"
f"| Frame | Timestamp | Caption |\n"
f"|-------|-----------|--------|\n"
f"{gallery_rows}"
        )

        await acreate_markdown_artifact(
            key=key,
            markdown=markdown,
            description=f"Image caption summary for video {video_id}",
        )

        await acreate_table_artifact(
            table=[
                {"Field": "Video ID", "Value": str(video_id)},
                {"Field": "Model", "Value": str(model)},
                {"Field": "Frames Captioned", "Value": str(len(final_result))},
                {"Field": "Prompt Tokens", "Value": f"{total_prompt:,}"},
                {"Field": "Completion Tokens", "Value": f"{total_completion:,}"},
                {"Field": "Total Tokens", "Value": f"{total_tokens:,}"},
                {"Field": "Total Cost", "Value": f"${total_cost:.6f}"},
                {"Field": "Avg Cost / Frame", "Value": f"${avg_cost:.6f}"},
            ],
            key=f"{key}-summary-table",
            description=f"Image caption stats table for video {video_id}",
        )

        gallery_table = []
        for artifact in samples:
            meta = artifact.metadata or {}
            caption = meta.get("caption", "")
            display_caption = (caption[:120] + "…") if len(caption) > 120 else caption
            gallery_table.append({
                "Frame": artifact.frame_index,
                "Timestamp": artifact.time_stamp,
                "Caption": display_caption,
            })
        await acreate_table_artifact(
            table=gallery_table,
            key=f"{key}-gallery-table",
            description=f"Caption gallery sample for video {video_id}",
        )


@task(**{**_base_kwargs, "name": "Image Caption Chunk"})  # type: ignore
async def image_caption_chunk_task(
    items: list[ImageArtifact],
) -> list[ImageCaptionArtifact]:
    """Caption a batch of image frames using ImageCaptionTask.execute().

    Downloads image bytes in preprocess(), then calls execute_single() once with
    the whole batch, captioning all frames concurrently via OpenRouter.

    Args:
        items: Batch of ImageArtifacts to caption.

    Returns:
        List of ImageCaptionArtifacts, one per frame in the batch.
    """
    logger = get_run_logger()
    settings = get_settings()

    logger.info(f"[ImageCaptionChunk] Starting | {len(items)} frame(s) in batch")

    minio_client = MinioStorageClient(
        endpoint=settings.minio.endpoint,
        access_key=settings.minio.access_key,
        secret_key=settings.minio.secret_key,
        secure=settings.minio.secure,
    )
    postgres_client = await get_postgres_client()
    logger.info(f"[ImageCaptionChunk] Clients initialized | minio={settings.minio.endpoint}")

    openrouter_config = OpenRouterConfig(
        api_key=os.environ.get("OPENROUTER_API_KEY", ""),
        model=IMAGE_CAPTION_CONFIG.additional_kwargs["model"],
        base_url=IMAGE_CAPTION_CONFIG.additional_kwargs["base_url"]
    )
    logger.info(f"[ImageCaptionChunk] OpenRouter config | model={openrouter_config.model}")

    task_impl = ImageCaptionTask(
        artifact_visitor=ArtifactPersistentVisitor(minio_client, postgres_client),
        minio_client=minio_client,
    )
    client = OpenRouterClient(config=openrouter_config)

    try:
        all_artifacts = await task_impl.execute_template(items, client)
    finally:
        await client.close()
        await shutdown_postgres_client(postgres_client)

    logger.info(f"[ImageCaptionChunk] Done | {len(all_artifacts)} artifact(s) produced")
    return all_artifacts
