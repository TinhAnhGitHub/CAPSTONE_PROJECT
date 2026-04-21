from __future__ import annotations

import json
import re
from datetime import datetime
from urllib.parse import urlparse
from loguru import logger

RESULT_TYPE_TO_SOCKET = {
    "image": "image_search",
    "segment": "segment_caption_search",
}

def parse_json_list(value: list[str] | str | None) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed
            logger.warning(f"parse_json_list: Expected list, got {type(parsed).__name__}")
            return None
        except json.JSONDecodeError:
            logger.warning(f"parse_json_list: Failed to parse JSON string: {value}")
            return None
    return None

def extract_s3_minio_url(s3_link: str) -> tuple[str, str]:
    if s3_link.startswith('s3://'):
        s3_link = s3_link.replace('s3://', '', 1)
        bucket, object_name = s3_link.split('/', 1)
        return bucket, object_name
    parsed = urlparse(s3_link)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    return bucket, key

def time_to_seconds(time_str: str) -> float:
    """Convert time string to seconds. Handles both 'HH:MM:SS' and 'HH:MM:SS.ffffff' formats."""
    try:
        t = datetime.strptime(time_str, "%H:%M:%S.%f")
    except ValueError:
        t = datetime.strptime(time_str, "%H:%M:%S")
    return t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1e6

def parse_time_safe(time_str: str) -> datetime:
    """Parse time string, handling both 'HH:MM:SS' and 'HH:MM:SS.ffffff' formats."""
    try:
        return datetime.strptime(time_str, "%H:%M:%S.%f")
    except ValueError:
        return datetime.strptime(time_str, "%H:%M:%S")

def time_range_overlap(start_input: float, end_input: float, start_range: float, end_range: float) -> bool:
    s1, e1 = start_input, end_input
    s2, e2 = start_range, end_range
    inter = max(0, min(e1, e2) - max(s1, s2))
    if inter > 0:
        return True
    return (s1 >= s2 and e1 <= e2) or (s2 >= s1 and e2 <= e1)

def convert_time_to_frame(time: str, fps: float) -> int:
    seconds = time_to_seconds(time)
    return int(seconds * fps)

def timecode_to_frame(time_str: str, fps: float) -> int:
    match = re.match(r"^(\d{2}):(\d{2}):(\d{2}(?:\.\d+)?)$", time_str)
    if not match:
        raise ValueError(f"Invalid timecode format: '{time_str}'. Expected: 'HH:MM:SS.sss'")
    hours, minutes, seconds = match.groups()
    total_seconds = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    return round(total_seconds * fps)

def format_duration(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

