import base64
import av
import asyncio
from PIL import Image
import io
import numpy as np
from typing import List
from urllib.parse import urlparse
from pathlib import Path
import decord
import imageio.v3 as iio


def return_related_asr_with_shot(
    asr_tokens: list[dict],
    start_frame: int,
    end_frame: int,
    *,
    overlap_threshold: float = 0.8,
)-> str:
    """
    Return the concatenated related asr moment, based on the segments
    """

    result = []

    for token in asr_tokens:
        token_start = int(token.get("start_frame", 0))
        token_end = int(token.get("end_frame", 0))
        token_text = token.get("text", "").strip()

        if not token_text or token_end <= token_start:
            continue
        
        intersection = max(
            0, min(
                token_end, end_frame
            ) - max(token_start, start_frame)
        )

        token_length = token_end - token_start
        overlap_ratio = intersection / token_length

        fully_inside = token_start >= start_frame and token_end <= end_frame

        if fully_inside or overlap_ratio >= overlap_threshold:
            result.append(token_text)
    
    return "\n\n".join(result).strip()

def get_segment_frame_indices(start: int, end: int, n: int) -> list[int]:
    """Return n evenly spaced frame indices between start and end."""
    if n <= 0 or end <= start:
        return []
    total = end - start
    return [start + (i + 1) * total // (n + 1) for i in range(n)]


class FastFrameReader:
    """High-performance random-access frame extractor using PyAV FFmpeg bindings."""

    def __init__(self, video_bytes: bytes):
        self.video_bytes = video_bytes
        self.buffer = io.BytesIO(video_bytes)
        self.container = av.open(self.buffer)
        self.stream = self.container.streams.video[0]
        self.fps = float(self.stream.average_rate)  # type: ignore
        if self.fps <= 0:
            raise RuntimeError("Invalid FPS detected in video.")


    def get_frame(self, frame_index: int) -> bytes:
        ts_seconds = frame_index / self.fps
        seek_ts = int(ts_seconds * av.time_base)
        self.buffer.seek(0)
        self.container.seek(seek_ts, stream=self.stream)#type:ignore
        for frame in self.container.decode(video=0):#type:ignore
            if frame.pts is None:
                continue
            pts_frame = int(frame.pts * self.fps * float(self.stream.time_base))#type:ignore

            if pts_frame >= frame_index:
                return self._encode_webp(frame)

        raise RuntimeError(f"Could not decode requested frame {frame_index}.")

    def _encode_webp(self, frame: av.VideoFrame) -> bytes:
        rgb = frame.to_ndarray(format="rgb24")
        img = Image.fromarray(rgb)

        buf = io.BytesIO()
        img.save(buf, format="WEBP", quality=80)  
        return buf.getvalue()
    


def extract_frames(video_bytes: bytes, indices: List[int]) -> List[tuple[int, bytes]]:
    reader = FastFrameReader(video_bytes)
    results = []
    for idx in sorted(indices):
        img_bytes = reader.get_frame(idx)
        results.append((idx, img_bytes))
    return results


def extract_segment(
    video_bytes: bytes,
    segments: list[tuple[int,int]], # segment in order
    n_per_segments: int
) -> list[list[str]]:
    
    sorted_segments = sorted(segments, key=lambda x: x[0])

    total_indices = []
    for segment in sorted_segments:
        start_frame, end_frame = segment
        segment_indices = get_segment_frame_indices(start_frame, end_frame, n_per_segments)
        total_indices.extend(segment_indices)
    
    indices_frames = extract_frames(video_bytes, total_indices)

    results = []
    for segment in sorted_segments:
        start_frame, end_frame = segment
        filter_frames = list(
            filter(lambda x: x[0] >= start_frame and x[0] <= end_frame, indices_frames )
        )

        frames = [base64.b64encode(f[1]).decode('utf-8') for f in filter_frames]
        results.append(frames)
    return results




def parse_s3_url(s3_url: str) -> tuple[str, str]:
    parsed = urlparse(s3_url)
    return parsed.netloc, parsed.path.lstrip("/")
