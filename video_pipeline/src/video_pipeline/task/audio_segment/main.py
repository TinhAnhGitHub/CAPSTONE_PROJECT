from __future__ import annotations

import os
from typing import cast

from prefect import get_run_logger, task
from prefect.artifacts import acreate_markdown_artifact, acreate_table_artifact
from llama_index.llms.openrouter import OpenRouter
from llama_index.core.llms import ChatMessage

from pydantic import BaseModel, Field
from video_pipeline.task.base.base_task import TaskConfig, BaseTask
from video_pipeline.core.client.progress import StageRegistry
from video_pipeline.core.artifact import ASRArtifact, AudioSegmentArtifact
from video_pipeline.core.storage.pg_tracker import ArtifactPersistentVisitor
from video_pipeline.core.client.storage.minio import MinioStorageClient
from video_pipeline.core.client.storage.pg.runtime import get_postgres_client, shutdown_postgres_client
from video_pipeline.config import get_settings

from .util import AudioSegment, AudioSegments, build_audio_batches, format_audio_time, format_batch_for_llm

AUDIO_SEGMENT_CONFIG = TaskConfig.from_yaml("audio_segmentation")




def _create_segment_from_group(
    asr_group: list[ASRArtifact],
    segment_index: int,
    video_id: str,
) -> AudioSegmentArtifact:
    first_asr = asr_group[0]
    last_asr = asr_group[-1]

    fps = first_asr.related_video_fps
    first_frame_num = first_asr.metadata.get("frame_num", [0, 0]) if first_asr.metadata else [0, 0]
    last_frame_num = last_asr.metadata.get("frame_num", [0, 0]) if last_asr.metadata else [0, 0]

    start_frame = first_frame_num[0]
    end_frame = last_frame_num[1]

    start_sec = start_frame / fps
    end_sec = end_frame / fps

    start_timestamp = first_asr.metadata.get("timestamp", ["", ""])[0] if first_asr.metadata else ""
    end_timestamp = last_asr.metadata.get("timestamp", ["", ""])[1] if last_asr.metadata else ""

    audio_text = " ".join(a.metadata.get("text", "") for a in asr_group if a.metadata)

    asr_artifact_ids = [a.artifact_id for a in asr_group]

    return AudioSegmentArtifact(
        asr_artifact_ids=asr_artifact_ids,
        related_video_id=video_id,
        related_video_minio_url=first_asr.related_video_minio_url,
        related_video_extension=first_asr.related_video_extension,
        related_video_fps=fps,
        segment_index=segment_index,
        start_sec=start_sec,
        end_sec=end_sec,
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
        audio_text=audio_text,
        start_frame=start_frame,
        end_frame=end_frame,
        user_id=first_asr.user_id,
    )


def rule_based_segment(
    asr_artifacts: list[ASRArtifact],
    video_id: str,
    min_duration_sec: float = 30.0,
) -> list[AudioSegmentArtifact]:
    if not asr_artifacts:
        return []

    segments = []
    current_group = []
    current_duration = 0.0
    segment_index = 0

    for asr in asr_artifacts:
        duration = asr.metadata.get("duration", 0.0) if asr.metadata else 0.0
        current_group.append(asr)
        current_duration += duration

        if current_duration >= min_duration_sec:
            segment = _create_segment_from_group(current_group, segment_index, video_id)
            segments.append(segment)
            segment_index += 1
            current_group = []
            current_duration = 0.0

    if current_group:
        segment = _create_segment_from_group(current_group, segment_index, video_id)
        segments.append(segment)

    return segments


@StageRegistry.register
class AudioSegmentTask(BaseTask[list[ASRArtifact], list[AudioSegmentArtifact]]):
    """Audio segmentation task that converts ASR artifacts into audio segments.

    Uses LLM for semantic segmentation, with rule-based fallback if:
    - ASR is empty
    - LLM returns empty segments
    """

    config = AUDIO_SEGMENT_CONFIG

    async def preprocess(self, input_data: list[ASRArtifact]) -> list[ASRArtifact]:
        logger = get_run_logger()
        logger.info(f"[AudioSegmentTask] Preprocessing {len(input_data)} ASR artifact(s)")
        return input_data

    async def execute(
        self,
        preprocessed: list[ASRArtifact],
        client: OpenRouter,
    ) -> list[AudioSegmentArtifact]:
        """Segment ASR artifacts using LLM or rule-based fallback."""
        logger = get_run_logger()

        if not preprocessed:
            logger.info("[AudioSegmentTask] No ASR artifacts to segment")
            return []

        batch_size = self.config.additional_kwargs.get("batch_size", 10)
        min_duration_sec = self.config.additional_kwargs.get("min_duration_sec", 30)

        has_content = any(a.metadata and a.metadata.get("text", "").strip() for a in preprocessed)

        video_id = preprocessed[0].related_video_id if preprocessed else ""

        if not has_content:
            logger.info(
                "[AudioSegmentTask] ASR has no meaningful content, using rule-based segmentation"
            )
            return rule_based_segment(preprocessed, video_id, min_duration_sec)

        batches = build_audio_batches(preprocessed, batch_size)
        all_batches_text = "\n\n\n\n\n\n\n\n".join(
            format_batch_for_llm(batch, i) for i, batch in enumerate(batches)
        )

        system_prompt = """
        You are an expert audio semantic segmenter.
        Your task is to divide numbered audio batches into semantically meaningful segments.
        STRICT RULES:
        1. Output ONLY valid JSON.
        2. Do NOT include markdown.
        3. Do NOT include explanations outside the JSON.
        4. Do NOT add extra fields.
        5. Do NOT rename fields.
        6. Segments must:
            - Merge or split batches based on semantic continuity.
            - Use approximately 15-second granularity.
            - Avoid overly small segments unless there is a strong topic shift.
            - Avoid overly large segments that mix unrelated themes.
        7. from_batch and to_batch must refer to existing batch numbers.
        8. Segments must be contiguous and non-overlapping.
        9. If valid audio content is provided:
            - Populate new_au_seg.
            - Set reason why do segment that way.
        10. If the input does NOT contain usable audio batches:
            - Leave new_au_seg as an empty list.
            - Fill reason with an explanation.
        """

        user_prompt = f"""Here are numbered audio batches.

        Please segment them into semantically meaningful audio chapters.

        Follow ~15 second granularity.
        Merge related batches.
        Split when topic meaningfully changes.

        Return structured JSON only.

        Audio Batches:

        {all_batches_text}
        """

        messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user_prompt),
        ]

        try:
            structured_llm = client.as_structured_llm(AudioSegments)
            response = await structured_llm.achat(messages)
            llm_result = cast(AudioSegments, response.raw)
        except Exception as e:
            logger.warning(f"[AudioSegmentTask] LLM call failed: {e}, using rule-based fallback")
            return rule_based_segment(preprocessed, video_id, min_duration_sec)

        if not llm_result.new_au_seg:
            logger.info("[AudioSegmentTask] LLM returned empty segments, using rule-based fallback")
            return rule_based_segment(preprocessed, video_id, min_duration_sec)

        segments = []
        for seg_info in llm_result.new_au_seg:
            from_idx = seg_info.from_batch
            to_idx = seg_info.to_batch

            start_batch_idx = from_idx * batch_size
            end_batch_idx = min((to_idx + 1) * batch_size, len(preprocessed))

            if start_batch_idx >= len(preprocessed):
                continue

            group = preprocessed[start_batch_idx:end_batch_idx]
            if not group:
                continue

            segment = _create_segment_from_group(group, len(segments), video_id)
            segments.append(segment)

        logger.info(f"[AudioSegmentTask] Created {len(segments)} segment(s) from LLM")
        return segments

    async def postprocess(self, result: list[AudioSegmentArtifact]) -> list[AudioSegmentArtifact]:
        """Persist audio segment artifacts to database."""
        for res in result:
            await self.artifact_visitor.visit_artifact(res)
        return result

    @staticmethod
    async def summary_artifact(final_result: list[AudioSegmentArtifact]) -> None:
        """Create a Prefect artifact summarizing audio segments."""
        if not final_result:
            return

        first = final_result[0]

        segment_rows = ""
        for i, seg in enumerate(final_result):
            audio_preview = (
                (seg.audio_text[:80] + "...") if len(seg.audio_text) > 80 else seg.audio_text
            )
            segment_rows += (
                f"| {i + 1} | {seg.start_timestamp} | {seg.end_timestamp} | "
                f"{seg.end_sec - seg.start_sec:.1f}s | {audio_preview} |\n"
            )

        total_duration = sum(s.end_sec - s.start_sec for s in final_result)

        markdown = (
            f"# Audio Segmentation Summary\n\n"
            f"| Field | Value |\n"
            f"|-------|-------|\n"
            f"| **Video ID** | `{first.related_video_id}` |\n"
            f"| **User ID** | `{first.user_id}` |\n"
            f"| **Segments Created** | `{len(final_result)}` |\n"
            f"| **Total Duration** | `{total_duration:.2f}s` |\n\n"
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
) -> list[AudioSegmentArtifact]:
    """Process ASR artifacts into audio segments using LLM or rule-based fallback.

    Args:
        asr_artifacts: List of ASRArtifact from ASR transcription

    Returns:
        List of AudioSegmentArtifact
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
    llm = OpenRouter(
        api_key=os.environ["OPENROUTER_API_KEY"],
        model=model,
        max_tokens=4096,
        context_window=8000,
    )

    task_impl = AudioSegmentTask(
        artifact_visitor=ArtifactPersistentVisitor(minio_client, postgres_client),
        minio_client=minio_client,
    )

    try:
        artifacts = await task_impl.execute_template(asr_artifacts, llm)
    finally:
        await shutdown_postgres_client(postgres_client)

    logger.info(f"[AudioSegment] Done | {len(artifacts)} segment(s) produced")
    return artifacts
