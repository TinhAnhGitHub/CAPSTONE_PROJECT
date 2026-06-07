from pydantic import BaseModel, Field

from video_pipeline.core.artifact import ASRArtifact, AudioSegmentArtifact

class MergeRule(BaseModel):
    from_segment: int = Field(..., description="Start raw ASR segment index")
    to_segment: int = Field(..., description="End raw ASR segment index")

class MergeList(BaseModel):
    merge_rules: list[MergeRule] | None = Field(
        None,
        description=(
            "List of conservative merge rules over raw ASR segments. "
            "Return None when no merge is necessary."
        ),
    )
    



def format_audio_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:04.1f}"

def format_segment_for_llm(asr_artifact: ASRArtifact, segment_idx: int) -> str:
    """Format one raw ASR segment for LLM input."""
    metadata = asr_artifact.metadata or {}
    timestamp = metadata.get("timestamp", ["", ""])
    audio_text = metadata.get("text", "")
    return (
        f"{'=' * 60}\n"
        f"Segment {segment_idx}\n"
        f"Start Time : {timestamp[0] if len(timestamp) > 0 else ''}\n"
        f"End Time   : {timestamp[1] if len(timestamp) > 1 else ''}\n"
        f"Audio Text:\n"
        f"{audio_text}\n"
    )


def create_segment_from_group(asr_group: list[ASRArtifact],
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


def passthrough_segments(
    asr_artifacts: list[ASRArtifact],
    video_id: str,
) -> list[AudioSegmentArtifact]:
    """Keep each raw ASR segment as its own audio segment."""
    return [
        create_segment_from_group([asr], segment_index, video_id)
        for segment_index, asr in enumerate(asr_artifacts)
    ]
