"""Artifact interfaces for video search results.

These interfaces represent search results returned from Qdrant.
They are decoupled from the video_pipeline artifact types.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from statistics import mean
from typing import Any, Callable, Sequence

from pydantic import BaseModel, Field


class ImageBytes(BaseModel):
    """Container for raw image bytes."""

    image_bytes: bytes


class BaseInterface(BaseModel, ABC):
    """Base class for all search result interfaces."""

    id: str
    related_video_id: str
    minio_path: str
    user_bucket: str

    @abstractmethod
    def accept_filter(self, filter_fn: Callable[[Any], bool]) -> bool:
        """Apply a filter function to this item."""
        ...

    @abstractmethod
    def brief_representation(self) -> str:
        """Return a brief string representation."""
        ...

    @abstractmethod
    def detailed_representation(self) -> str:
        """Return a detailed string representation."""
        ...

    @staticmethod
    @abstractmethod
    def quick_format(
        tool_name: str,
        tool_kwargs: dict,
        handle_id: str,
        items: Sequence[BaseInterface],
    ) -> str:
        """Format items for quick display."""
        ...

    @staticmethod
    @abstractmethod
    def statistic_format(
        tool_name: str,
        tool_kwargs: dict,
        handle_id: str,
        items: Sequence[BaseInterface],
        group_by: str,
    ) -> str:
        """Format items as statistics grouped by a field."""
        ...


class ImageInterface(BaseInterface):
    """Interface for image search results.

    Returned by:
    - ImageQdrantClient (visual search)
    - CaptionQdrantClient (caption search)
    """

    frame_index: int
    timestamp: str
    image_caption: str
    score: float
    
    timestamp_sec : float | None = Field(default=None, description="Timestamp in seconds")
    related_video_fps: float | None = Field(default=None, description="Fps of the related video")

    def accept_filter(self, filter_fn: Callable[[ImageInterface], bool]) -> bool:
        return filter_fn(self)

    def brief_representation(self) -> str:
        score_str = f"{self.score:.3f}" if self.score is not None else "No score"
        return (
            f"score={score_str} | {self.related_video_id} "
            f"@ Timestamp/FrameIndex: {self.timestamp}/{self.frame_index} "
            f"| minio_path: {self.minio_path}"
        )

    def detailed_representation(self) -> str:
        return (
            f"score={self.score:.3f} | {self.related_video_id} "
            f"@ Timestamp/FrameIndex: {self.timestamp}/{self.frame_index} "
            f"| minio_path: {self.minio_path} | image caption: {self.image_caption}"
        )

    @staticmethod
    def quick_format(
        tool_name: str,
        tool_kwargs: dict,
        handle_id: str,
        items: Sequence[BaseInterface],
    ) -> str:
        items = [i for i in items if isinstance(i, ImageInterface)]
        top_5 = sorted(items, key=lambda x: x.score, reverse=True)[:5]

        lines = [
            f"Handle id: {handle_id}",
            f"Tool name: {tool_name}",
            f"Tool input kwargs: {tool_kwargs}",
            f"Total: {len(items)} images",
            "Top 5:",
        ]

        for i, item in enumerate(top_5):
            lines.append(f"   {i}. {item.detailed_representation()}")

        return "\n".join(lines)

    @staticmethod
    def statistic_format(
        tool_name: str,
        tool_kwargs: dict,
        handle_id: str,
        items: Sequence[BaseInterface],
        group_by: str,
    ) -> str:
        current_items = [i for i in items if isinstance(i, ImageInterface)]
        grouped_data = defaultdict(list)

        for item in current_items:
            if group_by == "video_id":
                key = item.related_video_id
            elif group_by == "score_bucket":
                bucket = int(item.score * 10) / 10.0
                key = f"Score range [{bucket:.1f} - {bucket + 0.1:.1f}]"
            else:
                key = "Unknown"
            grouped_data[key].append(item)

        lines = [
            f"=== Statistic Report (Images) ===",
            f"Handle ID: {handle_id}",
            f"Tool: {tool_name} | Strategy: {group_by}",
            f"Total Items: {len(current_items)}",
            "-" * 40,
        ]

        sorted_keys = sorted(grouped_data.keys(), reverse=(group_by == "score_bucket"))

        for key in sorted_keys:
            group_items = grouped_data[key]
            count = len(group_items)
            avg_score = mean([x.score for x in group_items]) if group_items else 0.0
            best_item = max(group_items, key=lambda x: x.score)

            lines.append(f"Group: {key}")
            lines.append(f"  • Count: {count}")
            lines.append(f"  • Avg Score: {avg_score:.3f}")
            lines.append(f"  • Best Match: {best_item.brief_representation()}")
            lines.append("")

        return "\n".join(lines)


class AudioInterface(BaseInterface):
    """Interface for audio transcript search results.

    Returned by:
    - AudioQdrantClient (audio transcript search)
    """

    segment_index: int
    start_frame: int
    end_frame: int
    start_time: str
    end_time: str
    audio_text: str
    score: float
    start_sec: float | None = Field(default=None)
    end_sec: float | None = Field(default=None)
    related_audio_segment_artifact_id: str | None = Field(default=None)

    def accept_filter(self, filter_fn: Callable[[AudioInterface], bool]) -> bool:
        return filter_fn(self)

    def brief_representation(self) -> str:
        return (
            f"[{self.related_video_id}] "
            f"Segment {self.segment_index} "
            f"Frames {self.start_frame}-{self.end_frame} "
            f"({self.start_time} → {self.end_time}) | "
            f"score={self.score:.3f}"
        )

    def detailed_representation(self) -> str:
        return (
            f"Audio Segment {self.id} | Video: {self.related_video_id}\n"
            f"- Segment Index: {self.segment_index}\n"
            f"- Frames: {self.start_frame} → {self.end_frame}\n"
            f"- Time: {self.start_time} → {self.end_time}\n"
            f"- Score: {self.score:.3f}\n"
            f"- Transcript: {self.audio_text}\n"
            f"- MinIO URL: {self.minio_path}"
        )

    @staticmethod
    def quick_format(
        tool_name: str,
        tool_kwargs: dict,
        handle_id: str,
        items: Sequence[BaseInterface],
    ) -> str:
        items = [i for i in items if isinstance(i, AudioInterface)]
        top_5 = sorted(items, key=lambda x: x.score, reverse=True)[:5]

        lines = [
            f"Handle id: {handle_id}",
            f"Tool name: {tool_name}",
            f"Tool input kwargs: {tool_kwargs}",
            f"Total: {len(items)} audio segments",
            "Top 5:",
        ]

        for i, item in enumerate(top_5):
            lines.append(f"   {i}. {item.detailed_representation()}")

        return "\n".join(lines)

    @staticmethod
    def statistic_format(
        tool_name: str,
        tool_kwargs: dict,
        handle_id: str,
        items: Sequence[BaseInterface],
        group_by: str,
    ) -> str:
        current_items = [i for i in items if isinstance(i, AudioInterface)]
        grouped_data = defaultdict(list)

        for item in current_items:
            if group_by == "video_id":
                key = item.related_video_id
            elif group_by == "score_bucket":
                bucket = int(item.score * 10) / 10.0
                key = f"Score range [{bucket:.1f} - {bucket + 0.1:.1f}]"
            else:
                key = "Unknown"
            grouped_data[key].append(item)

        lines = [
            f"=== Statistic Report (Audio Segments) ===",
            f"Handle ID: {handle_id}",
            f"Tool: {tool_name} | Strategy: {group_by}",
            f"Total Items: {len(current_items)}",
            "-" * 40,
        ]

        sorted_keys = sorted(grouped_data.keys(), reverse=(group_by == "score_bucket"))

        for key in sorted_keys:
            group_items = grouped_data[key]
            count = len(group_items)
            avg_score = mean([x.score for x in group_items]) if group_items else 0.0
            best_item = max(group_items, key=lambda x: x.score)

            lines.append(f"Group: {key}")
            lines.append(f"  • Count: {count}")
            lines.append(f"  • Avg Score: {avg_score:.3f}")
            lines.append(f"  • Best Match: {best_item.brief_representation()}")
            lines.append("")

        return "\n".join(lines)


class SegmentInterface(BaseInterface):
    """Interface for segment search results.

    Returned by:
    - SegmentQdrantClient (segment search)
    """

    start_frame: int
    end_frame: int
    start_time: str
    end_time: str
    segment_caption: str
    score: float
    start_sec: float | None = Field(default=None)
    end_sec: float | None = Field(default=None)
    frame_indices: list[int] | None = Field(default=None)

    def accept_filter(self, filter_fn: Callable[[SegmentInterface], bool]) -> bool:
        return filter_fn(self)

    def brief_representation(self) -> str:
        return (
            f"[{self.related_video_id}] "
            f"Frames {self.start_frame}-{self.end_frame} "
            f"({self.start_time} → {self.end_time}) | "
            f"score={self.score:.3f} | "
            f"Minio path: {self.minio_path}"
        )

    def detailed_representation(self) -> str:
        return (
            f"Segment {self.id} | Video: {self.related_video_id}\n"
            f"- Frames: {self.start_frame} → {self.end_frame}\n"
            f"- Time: {self.start_time} → {self.end_time}\n"
            f"- Score: {self.score:.3f}\n"
            f"- Caption: {self.segment_caption}\n"
            f"- Caption MinIO URL: {self.minio_path}"
        )

    @staticmethod
    def quick_format(
        tool_name: str,
        tool_kwargs: dict,
        handle_id: str,
        items: Sequence[BaseInterface],
    ) -> str:
        items = [i for i in items if isinstance(i, SegmentInterface)]
        top_5 = sorted(items, key=lambda x: x.score, reverse=True)[:5]

        lines = [
            f"Handle id: {handle_id}",
            f"Tool name: {tool_name}",
            f"Tool input kwargs: {tool_kwargs}",
            f"Total: {len(items)} segments",
            "Top 5:",
        ]

        for i, item in enumerate(top_5):
            lines.append(f"   {i}. {item.detailed_representation()}")

        return "\n".join(lines)

    @staticmethod
    def statistic_format(
        tool_name: str,
        tool_kwargs: dict,
        handle_id: str,
        items: Sequence[BaseInterface],
        group_by: str,
    ) -> str:
        current_items = [i for i in items if isinstance(i, SegmentInterface)]
        grouped_data = defaultdict(list)

        for item in current_items:
            if group_by == "video_id":
                key = item.related_video_id
            elif group_by == "score_bucket":
                bucket = int(item.score * 10) / 10.0
                key = f"Score range [{bucket:.1f} - {bucket + 0.1:.1f}]"
            else:
                key = "Unknown"
            grouped_data[key].append(item)

        lines = [
            f"=== Statistic Report (Segments) ===",
            f"Handle ID: {handle_id}",
            f"Tool: {tool_name} | Strategy: {group_by}",
            f"Total Items: {len(current_items)}",
            "-" * 40,
        ]

        sorted_keys = sorted(grouped_data.keys(), reverse=(group_by == "score_bucket"))

        for key in sorted_keys:
            group_items = grouped_data[key]
            count = len(group_items)
            avg_score = mean([x.score for x in group_items]) if group_items else 0.0
            best_item = max(group_items, key=lambda x: x.score)

            lines.append(f"Group: {key}")
            lines.append(f"  • Count: {count}")
            lines.append(f"  • Avg Score: {avg_score:.3f}")
            lines.append(f"  • Best Match: {best_item.brief_representation()}")
            lines.append("")

        return "\n".join(lines)