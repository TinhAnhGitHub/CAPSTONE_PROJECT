from __future__ import annotations

import os
from pydantic import SecretStr

from prefect import get_run_logger, task
from prefect.artifacts import acreate_markdown_artifact
from langchain_core.messages import ChatMessage

from video_pipeline.task.base.base_task import TaskConfig, BaseTask
from video_pipeline.core.client.progress import StageRegistry
from video_pipeline.core.artifact import ASRArtifact, AudioSegmentArtifact
from video_pipeline.core.storage.pg_tracker import ArtifactPersistentVisitor
from video_pipeline.core.client.storage.minio import MinioStorageClient
from video_pipeline.core.client.storage.pg.runtime import get_postgres_client, shutdown_postgres_client
from video_pipeline.core.client.llm_provider.openrouter import OpenRouterClient, OpenRouterConfig
from video_pipeline.task.kg_graph.models import CostTracker
from video_pipeline.config import get_settings

from .util import (
    MergeList,
    create_segment_from_group,
    format_segment_for_llm,
    passthrough_segments,
)
from .prompt import SYSTEM_PROMPT

AUDIO_SEGMENT_CONFIG = TaskConfig.from_yaml("audio_segmentation")

MAX_RAW_SEGMENTS_WITHOUT_MERGE = 10
MIN_SEGMENTS_AFTER_MERGE = 8

def build_segments_from_merge_rules(
    asr_artifacts: list[ASRArtifact],
    video_id: str,
    merge_rules: list,
) -> list[AudioSegmentArtifact]:
    """Apply conservative raw-segment merge rules while preserving untouched segments."""
    last_index = len(asr_artifacts) - 1
    normalized_rules = []
    for rule in merge_rules:
        start_idx = min(max(0, rule.from_segment), last_index)
        end_idx = min(max(0, rule.to_segment), last_index)
        if start_idx <= end_idx:
            normalized_rules.append((start_idx, end_idx))
    normalized_rules.sort()

    segments: list[AudioSegmentArtifact] = []
    current_idx = 0
    for start_idx, end_idx in normalized_rules:
        if start_idx < current_idx:
            continue

        while current_idx < start_idx:
            segments.append(
                create_segment_from_group(
                    [asr_artifacts[current_idx]], len(segments), video_id
                )
            )
            current_idx += 1

        group = asr_artifacts[start_idx : end_idx + 1]
        if group:
            segments.append(create_segment_from_group(group, len(segments), video_id))
            current_idx = end_idx + 1

    while current_idx < len(asr_artifacts):
        segments.append(
            create_segment_from_group([asr_artifacts[current_idx]], len(segments), video_id)
        )
        current_idx += 1

    return segments


@StageRegistry.register
class AudioSegmentTask(BaseTask[list[ASRArtifact], tuple[list[AudioSegmentArtifact], CostTracker]]):
    """Audio merge task that converts raw ASR artifacts into audio segments.

    Small inputs are kept as-is. Larger inputs may be conservatively merged by
    the LLM, but raw segments are preserved whenever the result is too coarse.
    """

    config = AUDIO_SEGMENT_CONFIG

    async def preprocess(self, input_data: list[ASRArtifact]) -> list[ASRArtifact]:
        logger = get_run_logger()
        logger.info(f"[AudioSegmentTask] Preprocessing {len(input_data)} ASR artifact(s)")
        return input_data

    async def execute(
        self,
        preprocessed: list[ASRArtifact],
        client: OpenRouterClient,
    ) -> tuple[list[AudioSegmentArtifact], CostTracker]:
        """Segment ASR artifacts using LLM or rule-based fallback.

        Returns:
            Tuple of (segments, cost_tracker) for cost monitoring.
        """
        logger = get_run_logger()
        model = self.config.additional_kwargs.get("model", "google/gemini-2.5-flash-lite")
        cost_tracker = CostTracker(model=model)

        if not preprocessed:
            logger.info("[AudioSegmentTask] No ASR artifacts to segment")
            return [], cost_tracker

        has_content = any(a.metadata and a.metadata.get("text", "").strip() for a in preprocessed)

        video_id = preprocessed[0].related_video_id if preprocessed else ""

        if len(preprocessed) <= MAX_RAW_SEGMENTS_WITHOUT_MERGE:
            logger.info(
                "[AudioSegmentTask] Skipping merge because raw ASR segment count "
                f"is <= {MAX_RAW_SEGMENTS_WITHOUT_MERGE}"
            )
            return passthrough_segments(preprocessed, video_id), cost_tracker

        if not has_content:
            logger.info(
                "[AudioSegmentTask] ASR has no meaningful content, preserving raw segments"
            )
            return passthrough_segments(preprocessed, video_id), cost_tracker

        all_segments_text = "\n\n\n\n\n\n\n\n".join(
            format_segment_for_llm(segment, i) for i, segment in enumerate(preprocessed)
        )

        user_prompt = f"""Here are numbered raw ASR segments.

        Please merge only neighboring raw segments that clearly belong to the same local event.
        Keep the result fine-grained and conservative.
        Do not be aggressive: preserve boundaries whenever the relationship is weak or uncertain.
        Because this input has more than {MAX_RAW_SEGMENTS_WITHOUT_MERGE} raw ASR segments,
        the final merged result must contain at least {MIN_SEGMENTS_AFTER_MERGE} segments.

        Return structured JSON only.

        Raw ASR Segments:

        {all_segments_text}
        """

        messages = [
            ChatMessage(role="system", content=SYSTEM_PROMPT),
            ChatMessage(role="user", content=user_prompt),
        ]

        try:
            structured_llm = client.as_structured_llm(MergeList)
            llm_result, usage = await structured_llm(messages) #type:ignore
            cost_tracker.add_usage(
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                cost=usage.get("cost", 0.0),
            )
            logger.info(
                f"[AudioSegmentTask] LLM call completed | "
                f"prompt_tokens={usage.get('prompt_tokens', 0)} | "
                f"completion_tokens={usage.get('completion_tokens', 0)} | "
                f"cost=${usage.get('cost', 0.0):.6f}"
            )
        except Exception as e:
            logger.warning(
                f"[AudioSegmentTask] LLM call failed: {e}, preserving raw segments"
            )
            return passthrough_segments(preprocessed, video_id), cost_tracker

        if not llm_result.merge_rules:
            logger.info("[AudioSegmentTask] LLM returned no merge rules, preserving raw segments")
            return passthrough_segments(preprocessed, video_id), cost_tracker

        segments = build_segments_from_merge_rules(
            preprocessed,
            video_id,
            llm_result.merge_rules,
        )

        logger.info(
            f"[AudioSegmentTask] Created {len(segments)} segment(s) from LLM | "
            f"total_cost=${cost_tracker.total_cost:.6f}"
        )
        if len(segments) < MIN_SEGMENTS_AFTER_MERGE:
            logger.info(
                "[AudioSegmentTask] LLM merge was too aggressive "
                f"({len(segments)} < {MIN_SEGMENTS_AFTER_MERGE}); preserving raw segments"
            )
            return passthrough_segments(preprocessed, video_id), cost_tracker
    
        return segments, cost_tracker

    async def postprocess(
        self, result: tuple[list[AudioSegmentArtifact], CostTracker]
    ) -> tuple[list[AudioSegmentArtifact], CostTracker]:
        """Persist audio segment artifacts to database."""
        segments, cost_tracker = result
        for res in segments:
            await self.artifact_visitor.visit_artifact(res)
        return segments, cost_tracker

    @staticmethod
    async def summary_artifact(
        final_result: tuple[list[AudioSegmentArtifact], CostTracker]
    ) -> None:
        """Create a Prefect artifact summarizing audio segments with cost."""
        if not final_result:
            return

        segments, cost_tracker = final_result
        if not segments:
            return

        first = segments[0]

        segment_rows = ""
        for i, seg in enumerate(segments):
            audio_preview = seg.audio_text
            segment_rows += (
                f"| {i + 1} | {seg.start_timestamp} | {seg.end_timestamp} | "
                f"{seg.end_sec - seg.start_sec:.1f}s | {audio_preview} |\n"
            )

        total_duration = sum(s.end_sec - s.start_sec for s in segments)

        markdown = (
            f"# Audio Segmentation Summary\n\n"
            f"| Field | Value |\n"
            f"|-------|-------|\n"
            f"| **Video ID** | `{first.related_video_id}` |\n"
            f"| **User ID** | `{first.user_id}` |\n"
            f"| **Segments Created** | `{len(segments)}` |\n"
            f"| **Total Duration** | `{total_duration:.2f}s` |\n"
            f"| **Model** | `{cost_tracker.model}` |\n"
            f"| **Prompt Tokens** | `{cost_tracker.total_prompt_tokens:,}` |\n"
            f"| **Completion Tokens** | `{cost_tracker.total_completion_tokens:,}` |\n"
            f"| **Total Cost** | `${cost_tracker.total_cost:.6f}` |\n\n"
            f"## Audio Segments\n\n"
            f"| # | Start | End | Duration | Audio Text |\n"
            f"|---|-------|-----|----------|------------|\n"
            f"{segment_rows}"
        )

        await acreate_markdown_artifact(
            key=f"audio-segment-{first.related_video_id}".lower(),
            markdown=markdown,
            description=f"Audio segmentation summary for video {first.related_video_id}",
        )

@task(**{**AUDIO_SEGMENT_CONFIG.to_task_kwargs(), "name": "Audio Segment"})  # type: ignore
async def audio_segment_task(
    asr_artifacts: list[ASRArtifact],
) -> tuple[list[AudioSegmentArtifact], CostTracker]:
    """Process ASR artifacts into audio segments using LLM or rule-based fallback.

    Args:
        asr_artifacts: List of ASRArtifact from ASR transcription

    Returns:
        Tuple of (list of AudioSegmentArtifact, CostTracker with usage stats)
    """
    logger = get_run_logger()
    settings = get_settings()

    video_id = asr_artifacts[0].related_video_id if asr_artifacts else "unknown"
    logger.info(
        f"[AudioSegment] Starting | {len(asr_artifacts)} ASR artifact(s) | video_id={video_id}"
    )

    minio_client = MinioStorageClient(
        endpoint=settings.minio.endpoint,
        access_key=settings.minio.access_key,
        secret_key=settings.minio.secret_key,
        secure=settings.minio.secure,
    )
    postgres_client = await get_postgres_client()

    model = AUDIO_SEGMENT_CONFIG.additional_kwargs.get("model", "google/gemini-2.5-flash-lite")
    base_url = AUDIO_SEGMENT_CONFIG.additional_kwargs.get(
        "base_url", "https://openrouter.ai/api/v1"
    )

    openrouter_config = OpenRouterConfig(
        api_key=SecretStr(os.environ.get("OPENROUTER_API_KEY", "")),
        model=model,
        base_url=base_url,
    )

    task_impl = AudioSegmentTask(
        artifact_visitor=ArtifactPersistentVisitor(minio_client, postgres_client),
        minio_client=minio_client,
    )
    client = OpenRouterClient(config=openrouter_config)

    try:
        artifacts, cost_tracker = await task_impl.execute_template(asr_artifacts, client)
    finally:
        await client.close()
        await shutdown_postgres_client(postgres_client)

    logger.info(
        f"[AudioSegment] Done | {len(artifacts)} segment(s) produced | "
        f"total_cost=${cost_tracker.total_cost:.6f} | "
        f"prompt_tokens={cost_tracker.total_prompt_tokens} | "
        f"completion_tokens={cost_tracker.total_completion_tokens}"
    )
    return artifacts, cost_tracker
