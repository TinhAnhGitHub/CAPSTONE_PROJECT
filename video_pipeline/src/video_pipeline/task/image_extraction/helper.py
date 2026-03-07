import io
import av
from PIL import Image


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


def get_segment_frame_indices(start: int, end: int, n: int) -> list[int]:
    """Return n evenly-spaced frame indices within [start, end)."""
    if n <= 0 or end <= start:
        return []
    total = end - start
    return [start + (i + 1) * total // (n + 1) for i in range(n)]


def frames_to_timestamp(frame: int, fps: float) -> str:
    total_seconds = frame / fps
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"
