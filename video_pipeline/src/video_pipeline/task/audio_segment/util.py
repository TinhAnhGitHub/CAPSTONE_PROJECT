from pydantic import BaseModel, Field

from video_pipeline.core.artifact import ASRArtifact, AudioSegmentArtifact

class AudioSegment(BaseModel):
    from_batch: int
    to_batch: int
    summary_audio: str


class AudioSegments(BaseModel):
    new_au_seg: list[AudioSegment] = Field(default_factory=list)
    reason: str


def format_audio_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:04.1f}"

def build_audio_batches(asr_artifacts: list[ASRArtifact], batch_size: int):
    """Group ASR artifacts into batches for LLM segmentation."""
    batches = []
    for i in range(0, len(asr_artifacts), batch_size):
        batch = asr_artifacts[i : i + batch_size]
        batches.append(batch)
    return batches


def format_batch_for_llm(batch: list[ASRArtifact], batch_idx: int) -> str:
    """Format a batch of ASR artifacts for LLM input."""
    start_sec = batch[0].metadata.get("start_sec", 0) if batch[0].metadata else 0
    end_sec = batch[-1].metadata.get("end_sec", 0) if batch[-1].metadata else 0
    audio_texts = " ".join(a.metadata.get("text", "") for a in batch if a.metadata)
    return (
        f"{'=' * 60}\n"
        f"Batch {batch_idx}\n"
        f"Start Time : {format_audio_time(start_sec)}\n"
        f"End Time   : {format_audio_time(end_sec)}\n"
        f"Audio Text:\n"
        f"{audio_texts}\n"
    )



