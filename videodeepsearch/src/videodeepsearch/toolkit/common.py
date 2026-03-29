from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from tempfile import gettempdir
from typing import Any, Literal
from urllib.parse import urlparse

from loguru import logger
from pydantic import BaseModel

from videodeepsearch.schemas import ImageInterface, SegmentInterface, AudioInterface

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
    t = datetime.strptime(time_str, "%H:%M:%S.%f")
    return t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1e6


def parse_time_safe(time_str: str) -> datetime:
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

class CacheManager:
    def __init__(self, cache_dir: str | None = None):
        self.cache_dir = Path(cache_dir) if cache_dir else Path(gettempdir()) / "agno_cache" / "functions"

    def _generate_cache_key(self, function_name: str, args: dict[str, Any]) -> str:
        args_str = json.dumps(args, sort_keys=True, default=str)
        return hashlib.md5(f"{function_name}:{args_str}".encode()).hexdigest()

    def _get_cache_file_path(self, function_name: str, cache_key: str) -> Path:
        return self.cache_dir / function_name / f"{cache_key}.json"

    def get_cached_result(self, function_name: str, args: dict[str, Any]) -> tuple[Any | None, bool]:
        try:
            cache_key = self._generate_cache_key(function_name, args)
            cache_file = self._get_cache_file_path(function_name, cache_key)

            if not cache_file.exists():
                return None, False

            with open(cache_file, "r") as f:
                cache_data = json.load(f)

            return cache_data.get("result"), True

        except Exception as e:
            logger.warning(f"Failed to read cache for {function_name}: {e}")
            return None, False


class SearchResultContainer(BaseModel):
    tool_name: str
    tool_kwargs: dict[str, Any]
    results: list[ImageInterface | SegmentInterface | AudioInterface]
    result_type: Literal["image", "segment", "audio"]

    def get_brief(self, top_n: int = 5) -> str:
        sorted_results = sorted(self.results, key=lambda x: x.score, reverse=True)[:top_n]

        lines = [
            f"Tool: {self.tool_name}",
            f"Args: {self.tool_kwargs}",
            f"Total: {len(self.results)} {self.result_type}(s)",
            f"Top {min(top_n, len(sorted_results))}:",
        ]

        for i, item in enumerate(sorted_results):
            lines.append(f"  {i}. {item.brief_representation()}")

        return "\n".join(lines)

    def get_detailed(self, top_n: int = 5) -> str:
        sorted_results = sorted(self.results, key=lambda x: x.score, reverse=True)[:top_n]

        lines = [
            "=== Detailed Results ===",
            f"Tool: {self.tool_name}",
            f"Args: {self.tool_kwargs}",
            f"Total: {len(self.results)} {self.result_type}(s)",
            f"Top {min(top_n, len(sorted_results))}:",
            "",
        ]

        for i, item in enumerate(sorted_results):
            lines.append(f"[{i}] {item.detailed_representation()}")
            lines.append("")

        return "\n".join(lines)

    def get_statistics(self, group_by: str = "video_id") -> str:
        if self.result_type == "image":
            return ImageInterface.statistic_format(
                tool_name=self.tool_name,
                tool_kwargs=self.tool_kwargs,
                handle_id="local",
                items=self.results,
                group_by=group_by,
            )
        elif self.result_type == "audio":
            return AudioInterface.statistic_format(
                tool_name=self.tool_name,
                tool_kwargs=self.tool_kwargs,
                handle_id="local",
                items=self.results,
                group_by=group_by,
            )
        else:
            return SegmentInterface.statistic_format(
                tool_name=self.tool_name,
                tool_kwargs=self.tool_kwargs,
                handle_id="local",
                items=self.results,
                group_by=group_by,
            )

    def get_full(self) -> str:
        lines = [
            f"=== Full Results ({len(self.results)} items) ===",
            f"Tool: {self.tool_name}",
            f"Args: {self.tool_kwargs}",
            "",
        ]

        for i, item in enumerate(self.results):
            lines.append(f"[{i}] {item.detailed_representation()}")
            lines.append("")

        return "\n".join(lines)


__all__ = [
    "extract_s3_minio_url",
    "time_to_seconds",
    "parse_time_safe",
    "time_range_overlap",
    "convert_time_to_frame",
    "timecode_to_frame",
    "format_duration",
    "CacheManager",
    "SearchResultContainer",
]