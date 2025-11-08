from urllib.parse import urlparse
from datetime import datetime
import tempfile
import os
import re

def extract_s3_minio_url(s3_link:str) -> tuple[str,str]:
    parsed = urlparse(s3_link)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    return bucket, key

def time_to_seconds(time_str: str) -> float:
        t = datetime.strptime(time_str, "%H:%M:%S.%f")
        return t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1e6

def time_range_overlap(
        start_input: float,
        end_input:float,
        start_range: float,
        end_range:float,
        iou:float,
    ):
        s1, e1 = start_input, end_input
        s2, e2 = start_range, end_range
        inter = max(0, min(e1, e2) - max(s1, s2))
        union = max(e1, e2) - min(s1, s2)
        return (inter / union) >= iou or (s1 >= s2 and e1 <= e2)


def timecode_to_frame(time_str: str, fps: float):
    match = re.match(r"^(\d{2}):(\d{2}):(\d{2}(?:\.\d+)?)$", time_str)
    if not match:
        raise ValueError(
            f"Invalid timecode format: '{time_str}'. Expected format: 'HH:MM:SS.sss'"
        )
    hours, minutes, seconds = match.groups()
    total_seconds = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    frame_index = round(total_seconds * fps)

    return frame_index      


def create_tmp_file_from_minio_object(
    file_bytes: bytes,
    extension:str
):
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=extension)
    os.close(tmp_fd)

    with open(tmp_path, 'wb') as f:
        f.write(file_bytes)
    return tmp_path

def parse_time_safe(time_str: str):
    """Parse both HH:MM:SS and HH:MM:SS.sss safely."""
    try:
        return datetime.strptime(time_str, "%H:%M:%S.%f")
    except ValueError:
        return datetime.strptime(time_str, "%H:%M:%S")





