import asyncio
import av
import io
from urllib.parse import urlparse
from PIL import Image


def parse_s3_url(s3_url: str) -> tuple[str, str]:
    parsed = urlparse(s3_url)
    return parsed.netloc, parsed.path.lstrip("/")

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
        seek_ts = int(ts_seconds / self.stream.time_base)


        self.container.seek(seek_ts, stream=self.stream)#type:ignore

        for frame in self.container.decode(video=0):
            if frame.pts is None:
                continue

            ts_sec = frame.pts * self.stream.time_base
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
    

def extract_frames(video_bytes: bytes, indices: list[int]) -> list[bytes]:
    reader = FastFrameReader(video_bytes)
    results = []
    for idx in sorted(indices):
        img_bytes = reader.get_frame(idx)
        results.append(img_bytes)
    return results

async def extract_frames_async(video_bytes: bytes, indices: list[int]) -> list[bytes]:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, extract_frames, video_bytes, indices)



def get_segment_frame_indices(start: int, end: int, n: int) -> list[int]:
    if n <= 0 or end <= start:
        return []

    total = end - start
    return [start + (i + 1) * total // (n + 1) for i in range(n)]