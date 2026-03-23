from __future__ import annotations

import os

from prefect import get_run_logger, task
from prefect.artifacts import acreate_markdown_artifact
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, SecretStr

from video_pipeline.task.base.base_task import TaskConfig, BaseTask
from video_pipeline.core.client.progress import StageRegistry
from video_pipeline.core.artifact import AudioSegmentArtifact, SegmentCaptionArtifact
from video_pipeline.core.storage.pg_tracker import ArtifactPersistentVisitor
from video_pipeline.core.client.storage.minio import MinioStorageClient
from video_pipeline.core.client.storage.pg.runtime import get_postgres_client, shutdown_postgres_client
from video_pipeline.core.client.llm_provider.openrouter import OpenRouterClient, OpenRouterConfig
from video_pipeline.task.kg_graph.models import CostTracker
from video_pipeline.config import get_settings
from video_pipeline.task.image_extraction.helper import FastFrameReader, get_segment_frame_indices


SEGMENT_CAPTION_CONFIG = TaskConfig.from_yaml("segment_caption")


class OutputCaptionSegment(BaseModel):
    summary_caption: str = Field(
        ...,
        description=(
            "A concise overall summary describing what is happening in the segment. "
            "It should capture the main activities, visual elements, and relevant audio cues "
            "(such as speech, music, or sound effects) that define the scene. "
            "The summary should represent the segment as a whole rather than focusing on small details."
        ),
    )

    event_captions: list[str] = Field(
        default_factory=list,
        description=(
            "A list of short, atomic event descriptions occurring within the segment. "
            "Each caption should describe a single, distinct action or event that can be clearly "
            "observed or inferred from the scene (e.g., a person speaking, a door opening, a dog running). "
            "Think of these as multiple micro-observations from different viewers describing what they notice. "
            "Each item should focus on only one event and avoid combining multiple actions into a single caption."
        ),
    )


CAPTION_SYSTEM_PROMPT = """
You are an expert in multimodal video understanding.

Your task is to generate a structured caption following the schema `OutputCaptionSegment`.

The output must contain two fields:
1. summary_caption
2. event_captions

---------------------
summary_caption
---------------------
Write a concise summary describing the segment as a whole.

Requirements:
- Maximum length: 125 words
- Write as ONE coherent paragraph.
- Describe the overall situation, key entities, and the main progression of events.
- Use both visual information and audio transcript when available.
- Focus on:
  • important people, objects, locations, and on-screen elements
  • major actions and how they unfold
  • context or setting when relevant
  • the central event or narrative of the segment

Avoid:
- frame-by-frame narration
- bullet points or lists
- repeating transcript verbatim
- speculation beyond visible or audible evidence


---------------------
event_captions
---------------------
Provide a list of short event descriptions representing atomic observations within the segment.

Requirements:
- Each item must describe ONE distinct action or event.
- Each caption should be short (5–15 words recommended).
- Focus on clearly observable actions or audio events.
- Write them as independent statements.

Examples of good event captions:
- "A man opens a car door."
- "Two people shake hands."
- "A dog runs across the street."
- "The narrator mentions the Apollo mission."

Avoid:
- combining multiple events in one caption
- long sentences
- repeating the summary
- vague descriptions like "something happens"

The event list should represent multiple small observations that together reflect what happens during the segment.
"""


@StageRegistry.register
class SegmentCaptionTask(BaseTask[list[AudioSegmentArtifact], tuple[list[SegmentCaptionArtifact], CostTracker]]):
    """Caption audio segments using VLM with visual frames and audio transcript."""

    config = SEGMENT_CAPTION_CONFIG

    async def preprocess(
        self,
        input_data: list[AudioSegmentArtifact],
    ) -> list[AudioSegmentArtifact]:
        """Return segments unchanged - video will be loaded in execute().

        Note: We don't load video bytes here to avoid memory bloat.
        Video is streamed to disk in execute() and processed within
        the context manager to ensure proper cleanup.
        """
        logger = get_run_logger()
        logger.info(f"[SegmentCaptionTask] Preparing {len(input_data)} segment(s)")
        return input_data

    async def execute(
        self,
        preprocessed: list[AudioSegmentArtifact],
        client: OpenRouterClient,
    ) -> tuple[list[SegmentCaptionArtifact], CostTracker]:
        """Generate captions for each segment using VLM.

        Downloads video via streaming (no memory bloat) and processes
        all segments within the context manager.

        Returns:
            Tuple of (artifacts, cost_tracker) for cost monitoring.
        """
        logger = get_run_logger()
        n_frames = self.config.additional_kwargs.get("n_frames", 6)
        max_concurrent = self.config.additional_kwargs.get("max_concurrent", 5)
        model = self.config.additional_kwargs.get("model", "qwen/qwen3-vl-32b-instruct")

        if not preprocessed:
            return [], CostTracker(model=model)

        logger.info(
            f"[SegmentCaptionTask] Captioning {len(preprocessed)} segment(s) | max_concurrent={max_concurrent}"
        )

        video_s3_url = preprocessed[0].related_video_minio_url
        artifacts: list[SegmentCaptionArtifact] = []
        cost_tracker = CostTracker(model=model)

        async with self.minio_client.fetch_object_streaming(
            s3_url=video_s3_url, suffix=".mp4"
        ) as video_path:
            import asyncio

            semaphore = asyncio.Semaphore(max_concurrent)

            async def caption_segment(
                seg: AudioSegmentArtifact
            ) -> tuple[SegmentCaptionArtifact | None, dict]:
                async with semaphore:
                    reader = FastFrameReader(video_path)
                    fps = reader.fps

                    start_frame = int(seg.start_sec * fps)
                    end_frame = int(seg.end_sec * fps)

                    frame_indices = get_segment_frame_indices(start_frame, end_frame, n_frames)

                    image_blocks = []
                    for fi in frame_indices:
                        try:
                            frame_bytes = reader.get_frame(fi)
                            data_url = client.encode_image_bytes(frame_bytes, mime="image/webp")
                            image_blocks.append(
                                {"type": "image_url", "image_url": {"url": data_url}}
                            )
                        except Exception as e:
                            logger.warning(f"Failed to extract frame {fi}: {e}")

                    if not image_blocks:
                        logger.warning(f"No frames extracted for segment {seg.segment_index}, skipping")
                        return None, {}

                    user_text = f"Audio Transcript:\n{seg.audio_text}"

                    messages = [
                        SystemMessage(content=CAPTION_SYSTEM_PROMPT),
                        HumanMessage(
                            content=[
                                *image_blocks,
                                {"type": "text", "text": user_text},
                            ]
                        ),
                    ]

                    try:
                        allm = client.as_structured_llm(OutputCaptionSegment)
                        caption_result, usage = await allm(messages) #type:ignore
                    except Exception as e:
                        logger.warning(f"Failed to caption segment {seg.segment_index}: {e}")
                        return None, {}

                    artifact = SegmentCaptionArtifact(
                        related_audio_segment_artifact_id=seg.artifact_id,
                        related_video_id=seg.related_video_id,
                        related_video_minio_url=seg.related_video_minio_url,
                        related_video_extension=seg.related_video_extension,
                        related_video_fps=seg.related_video_fps,
                        start_frame=start_frame,
                        end_frame=end_frame,
                        start_timestamp=seg.start_timestamp,
                        end_timestamp=seg.end_timestamp,
                        start_sec=seg.start_sec,
                        end_sec=seg.end_sec,
                        audio_text=seg.audio_text,
                        summary_caption=caption_result.summary_caption if caption_result else "",
                        event_captions=caption_result.event_captions if caption_result else [],
                        user_id=seg.user_id,
                    )
                    return artifact, usage

            tasks = [caption_segment(seg) for seg in preprocessed]
            results = await asyncio.gather(*tasks)

            for artifact, usage in results:
                if artifact is not None:
                    artifacts.append(artifact)
                if usage:
                    cost_tracker.add_usage(
                        prompt_tokens=usage.get("prompt_tokens", 0),
                        completion_tokens=usage.get("completion_tokens", 0),
                        cost=usage.get("cost", 0.0),
                    )

        logger.info(
            f"[SegmentCaptionTask] Created {len(artifacts)} caption artifact(s) | "
            f"total_cost=${cost_tracker.total_cost:.6f} | "
            f"prompt_tokens={cost_tracker.total_prompt_tokens} | "
            f"completion_tokens={cost_tracker.total_completion_tokens}"
        )
        return artifacts, cost_tracker

    async def postprocess(
        self, result: tuple[list[SegmentCaptionArtifact], CostTracker]
    ) -> tuple[list[SegmentCaptionArtifact], CostTracker]:
        """Persist segment captions to database."""
        artifacts, cost_tracker = result
        for res in artifacts:
            await self.artifact_visitor.visit_artifact(res)
        return artifacts, cost_tracker

    @staticmethod
    async def summary_artifact(final_result: tuple[list[SegmentCaptionArtifact], CostTracker]) -> None:
        """Create a Prefect artifact summarizing segment captions."""
        if not final_result:
            return

        artifacts, cost_tracker = final_result
        if not artifacts:
            return

        first = artifacts[0]

        segment_rows = ""
        for i, seg in enumerate(artifacts):
            caption_preview = (
                (seg.summary_caption[:60] + "...")
                if len(seg.summary_caption) > 60
                else seg.summary_caption
            )
            n_events = len(seg.event_captions)
            segment_rows += (
                f"| {i + 1} | {seg.start_timestamp} | {seg.end_timestamp} | "
                f"{n_events} | {caption_preview} |\n"
            )

        markdown = (
            f"# Segment Caption Summary\n\n"
            f"| Field | Value |\n"
            f"|-------|-------|\n"
            f"| **Video ID** | `{first.related_video_id}` |\n"
            f"| **User ID** | `{first.user_id}` |\n"
            f"| **Segments Captioned** | `{len(artifacts)}` |\n"
            f"| **Model** | `{cost_tracker.model}` |\n"
            f"| **Prompt Tokens** | `{cost_tracker.total_prompt_tokens:,}` |\n"
            f"| **Completion Tokens** | `{cost_tracker.total_completion_tokens:,}` |\n"
            f"| **Total Cost** | `${cost_tracker.total_cost:.6f}` |\n\n"
            f"## Segment Captions\n\n"
            f"| # | Start | End | Events | Summary |\n"
            f"|---|-------|-----|--------|---------|\n"
            f"{segment_rows}"
        )

        await acreate_markdown_artifact(
            key=f"segment-caption-{first.related_video_id}".lower(),
            markdown=markdown,
            description=f"Segment caption summary for video {first.related_video_id}",
        )


@task(**{**SEGMENT_CAPTION_CONFIG.to_task_kwargs(), "name": "Segment Caption"})  # type: ignore
async def segment_caption_chunk_task(
    segments: list[AudioSegmentArtifact],
) -> tuple[list[SegmentCaptionArtifact], CostTracker]:
    """Process audio segments into captions.

    Args:
        segments: List of AudioSegmentArtifact

    Returns:
        Tuple of (list of SegmentCaptionArtifact, CostTracker with usage stats)
    """
    logger = get_run_logger()
    settings = get_settings()

    logger.info(f"[SegmentCaption] Starting | {len(segments)} segment(s)")

    minio_client = MinioStorageClient(
        endpoint=settings.minio.endpoint,
        access_key=settings.minio.access_key,
        secret_key=settings.minio.secret_key,
        secure=settings.minio.secure,
    )
    postgres_client = await get_postgres_client()

    model = SEGMENT_CAPTION_CONFIG.additional_kwargs.get("model", "qwen/qwen3-vl-32b-instruct")
    base_url = SEGMENT_CAPTION_CONFIG.additional_kwargs.get(
        "base_url", "https://openrouter.ai/api/v1"
    )

    caption_config = OpenRouterConfig(
        model=model,
        base_url=base_url,
        api_key=SecretStr(os.environ.get("OPENROUTER_API_KEY", "")),
    )

    task_impl = SegmentCaptionTask(
        artifact_visitor=ArtifactPersistentVisitor(minio_client, postgres_client),
        minio_client=minio_client,
    )
    client = OpenRouterClient(config=caption_config)

    try:
        artifacts, cost_tracker = await task_impl.execute_template(segments, client)
    finally:
        await client.close()
        await shutdown_postgres_client(postgres_client)

    logger.info(
        f"[SegmentCaption] Done | {len(artifacts)} artifact(s) produced | "
        f"total_cost=${cost_tracker.total_cost:.6f}"
    )
    return artifacts, cost_tracker
