import io
from pathlib import Path

import av
from PIL import Image

from video_pipeline.task.video_utils import frames_to_timestamp


class FastFrameReader:
    """High-performance random-access frame extractor using PyAV FFmpeg bindings."""

    def __init__(self, video_source: bytes | Path | str):
        """Initialize reader from bytes or file path.

        Args:
            video_source: Either video bytes or a path to a video file.
                          Using a file path is more memory-efficient for large videos.
        """
        if isinstance(video_source, bytes):
            self.video_bytes = video_source
            self.buffer = io.BytesIO(video_source)
            self.container = av.open(self.buffer)
            self._is_bytes = True
        else:
            self.video_bytes = None
            self.buffer = None
            self.container = av.open(str(video_source))
            self._is_bytes = False

        self.stream = self.container.streams.video[0]
        self.fps = float(self.stream.average_rate)  # type: ignore
        if self.fps <= 0:
            raise RuntimeError("Invalid FPS detected in video.")

    def get_frame(self, frame_index: int) -> bytes:
        ts_seconds = frame_index / self.fps
        seek_ts = int(ts_seconds / self.stream.time_base)  # type: ignore

        self.container.seek(seek_ts, stream=self.stream)  # type: ignore

        for frame in self.container.decode(video=0):  # type: ignore
            if frame.pts is None:
                continue
            ts_sec = frame.pts * self.stream.time_base  # type: ignore
            frame_number = int(ts_sec * self.fps)
            if frame_number >= frame_index:
                return self._encode_webp(frame)

        raise RuntimeError(f"Could not decode requested frame {frame_index}.")

    def _encode_webp(self, frame: av.VideoFrame) -> bytes:
        rgb = frame.to_ndarray(format="rgb24")
        img = Image.fromarray(rgb)
        buf = io.BytesIO()
        img.save(buf, format="WEBP", quality=80)
        return buf.getvalue()

    def close(self):
        """Release resources."""
        if self.container:
            self.container.close()
        if self.buffer:
            self.buffer.close()
        self.video_bytes = None
        self.buffer = None
        self.container = None


def get_segment_frame_indices(start: int, end: int, n: int) -> list[int]:
    """Return n evenly-spaced frame indices within [start, end)."""
    if n <= 0 or end <= start:
        return []
    total = end - start
    return [start + (i + 1) * total // (n + 1) for i in range(n)]
