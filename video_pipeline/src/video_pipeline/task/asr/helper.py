import os
import re
import tempfile
from urllib.parse import urlparse

import ffmpeg

from video_pipeline.task.video_utils import frames_to_timestamp


def split_minio_url(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme == "s3":
        return parsed.netloc, parsed.path.lstrip("/")
    path_parts = parsed.path.lstrip("/").split("/", 1)
    bucket = path_parts[0]
    object_name = path_parts[1] if len(path_parts) > 1 else ""
    return bucket, object_name


def extract_single_audio_segment(
    video_path: str,
    start_frame: int,
    end_frame: int,
    fps: float,
) -> str:
    """Extract audio segment from video as temp WAV file.

    The caller is responsible for deleting the returned file after use.
    """
    start_sec = start_frame / fps
    duration_sec = (end_frame - start_frame) / fps

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()

    (
        ffmpeg.input(video_path, ss=start_sec, t=max(duration_sec, 0.01))
        .output(
            tmp.name,
            acodec="pcm_s16le",
            ac=1,
            ar="16000",
            vn=None,
        )
        .overwrite_output()
        .run(quiet=True)
    )
    return tmp.name


def delete_audio_file(path: str) -> None:
    """Delete a temp audio file, silently ignoring missing files."""
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def parse_asr_response(raw: str) -> str:
    """Parse a Qwen3-ASR response string into clean transcription text.

    Qwen3-ASR prefixes each language segment with:
        ``language <lang><asr_text><transcription>``

    This strips all such markers and returns the joined text.

    Example raw input::
        "language English<asr_text>Hello world.language Chinese<asr_text>你好。"

    Returns::
        "Hello world.你好。"
    """
    text = re.sub(r"language\s+[^<]+<asr_text>", "", raw)
    return text.strip()
