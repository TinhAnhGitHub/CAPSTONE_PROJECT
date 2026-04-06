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

    @abstractmethod
    def to_socket_format(self) -> dict[str, Any]:
        """Convert to socket-compatible format for frontend display."""
        ...

    @staticmethod
    @abstractmethod
    def quick_format(
        tool_name: str,
        tool_kwargs: dict,
        handle_id: str,
        items: Sequence[BaseInterface],
    ) -> dict[str, Any]:
        """Format items for quick display. Returns JSON dict."""
        ...

    @staticmethod
    @abstractmethod
    def statistic_format(
        tool_name: str,
        tool_kwargs: dict,
        handle_id: str,
        items: Sequence[BaseInterface],
        group_by: str,
    ) -> dict[str, Any]:
        """Format items as statistics grouped by a field. Returns JSON dict."""
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
    
    timestamp_sec: float | None = Field(default=None, description="Timestamp in seconds")
    related_video_fps: float | None = Field(default=None, description="Fps of the related video")
    minio_path: str | None = Field(default=None, description="S3 path to image")

    def accept_filter(self, filter_fn: Callable[[ImageInterface], bool]) -> bool:
        return filter_fn(self)

    def brief_representation(self) -> str:
        score_str = f"{self.score:.3f}" if self.score is not None else "No score"
        return (
            f"score={score_str} | {self.related_video_id} "
            f"@ Timestamp/FrameIndex: {self.timestamp}/{self.frame_index} "
        )

    def detailed_representation(self) -> str:
        return (
            f"score={self.score:.3f} | {self.related_video_id} "
            f"@ Timestamp/FrameIndex: {self.timestamp}/{self.frame_index} "
            f"| image caption: {self.image_caption}"
        )

    def to_socket_format(self) -> dict[str, Any]:
        """Convert to socket-compatible format for frontend display."""
        return {
            "id": self.id,
            "video_id": self.related_video_id,
            "frame_index": self.frame_index,
            "timestamp": self.timestamp,
            "timestamp_sec": self.timestamp_sec,
            "fps": self.related_video_fps,
            "caption": self.image_caption,
            "score": self.score,
            "minio_path": self.minio_path or "",
        }

    @staticmethod
    def quick_format(
        tool_name: str,
        tool_kwargs: dict,
        handle_id: str,
        items: Sequence[BaseInterface],
    ) -> dict[str, Any]:
        """Format items for quick display. Returns JSON dict."""
        items = [i for i in items if isinstance(i, ImageInterface)]
        top_5 = sorted(items, key=lambda x: x.score, reverse=True)[:5]

        return {
            "view_mode": "quick",
            "handle_id": handle_id,
            "tool_name": tool_name,
            "tool_kwargs": tool_kwargs,
            "result_type": "image",
            "total": len(items),
            "top_results": [item.to_socket_format() for item in top_5],
        }

    @staticmethod
    def statistic_format(
        tool_name: str,
        tool_kwargs: dict,
        handle_id: str,
        items: Sequence[BaseInterface],
        group_by: str,
    ) -> dict[str, Any]:
        """Format items as statistics grouped by a field. Returns JSON dict."""
        current_items = [i for i in items if isinstance(i, ImageInterface)]
        grouped_data = defaultdict(list)

        for item in current_items:
            if group_by == "video_id":
                key = item.related_video_id
            elif group_by == "score_bucket":
                bucket = int(item.score * 10) / 10.0
                key = f"{bucket:.1f}-{bucket + 0.1:.1f}"
            else:
                key = "Unknown"
            grouped_data[key].append(item)

        groups = []
        sorted_keys = sorted(grouped_data.keys(), reverse=(group_by == "score_bucket"))

        for key in sorted_keys:
            group_items = grouped_data[key]
            count = len(group_items)
            avg_score = mean([x.score for x in group_items]) if group_items else 0.0
            best_item = max(group_items, key=lambda x: x.score)

            groups.append({
                "key": key,
                "count": count,
                "avg_score": round(avg_score, 3),
                "best_match": best_item.to_socket_format(),
            })

        return {
            "view_mode": "statistics",
            "result_type": "image",
            "handle_id": handle_id,
            "tool_name": tool_name,
            "tool_kwargs": tool_kwargs,
            "group_by": group_by,
            "total_items": len(current_items),
            "groups": groups,
        }


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
        )

    def to_socket_format(self) -> dict[str, Any]:
        """Convert to socket-compatible format for frontend display."""
        return {
            "id": self.id,
            "video_id": self.related_video_id,
            "segment_index": self.segment_index,
            "frame_range": {
                "start": self.start_frame,
                "end": self.end_frame,
            },
            "time_range": {
                "start": self.start_time,
                "end": self.end_time,
            },
            "start_sec": self.start_sec,
            "end_sec": self.end_sec,
            "transcript": self.audio_text,
            "score": self.score,
        }

    @staticmethod
    def quick_format(
        tool_name: str,
        tool_kwargs: dict,
        handle_id: str,
        items: Sequence[BaseInterface],
    ) -> dict[str, Any]:
        """Format items for quick display. Returns JSON dict."""
        items = [i for i in items if isinstance(i, AudioInterface)]
        top_5 = sorted(items, key=lambda x: x.score, reverse=True)[:5]

        return {
            "view_mode": "quick",
            "handle_id": handle_id,
            "tool_name": tool_name,
            "tool_kwargs": tool_kwargs,
            "result_type": "audio",
            "total": len(items),
            "top_results": [item.to_socket_format() for item in top_5],
        }

    @staticmethod
    def statistic_format(
        tool_name: str,
        tool_kwargs: dict,
        handle_id: str,
        items: Sequence[BaseInterface],
        group_by: str,
    ) -> dict[str, Any]:
        """Format items as statistics grouped by a field. Returns JSON dict."""
        current_items = [i for i in items if isinstance(i, AudioInterface)]
        grouped_data = defaultdict(list)

        for item in current_items:
            if group_by == "video_id":
                key = item.related_video_id
            elif group_by == "score_bucket":
                bucket = int(item.score * 10) / 10.0
                key = f"{bucket:.1f}-{bucket + 0.1:.1f}"
            else:
                key = "Unknown"
            grouped_data[key].append(item)

        groups = []
        sorted_keys = sorted(grouped_data.keys(), reverse=(group_by == "score_bucket"))

        for key in sorted_keys:
            group_items = grouped_data[key]
            count = len(group_items)
            avg_score = mean([x.score for x in group_items]) if group_items else 0.0
            best_item = max(group_items, key=lambda x: x.score)

            groups.append({
                "key": key,
                "count": count,
                "avg_score": round(avg_score, 3),
                "best_match": best_item.to_socket_format(),
            })

        return {
            "view_mode": "statistics",
            "result_type": "audio",
            "handle_id": handle_id,
            "tool_name": tool_name,
            "tool_kwargs": tool_kwargs,
            "group_by": group_by,
            "total_items": len(current_items),
            "groups": groups,
        }


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
    fps: float | None = Field(default=None, description="Video fps")

    def accept_filter(self, filter_fn: Callable[[SegmentInterface], bool]) -> bool:
        return filter_fn(self)

    def brief_representation(self) -> str:
        return (
            f"[{self.related_video_id}] "
            f"Frames {self.start_frame}-{self.end_frame} "
            f"({self.start_time} → {self.end_time}) | "
            f"score={self.score:.3f} | "
        )

    def detailed_representation(self) -> str:
        return (
            f"Segment {self.id} | Video: {self.related_video_id}\n"
            f"- Frames: {self.start_frame} → {self.end_frame}\n"
            f"- Time: {self.start_time} → {self.end_time}\n"
            f"- Score: {self.score:.3f}\n"
            f"- Caption: {self.segment_caption}\n"
        )

    def to_socket_format(self) -> dict[str, Any]:
        """Convert to socket-compatible format for frontend display."""
        return {
            "id": self.id,
            "video_id": self.related_video_id,
            "frame_range": {
                "start": self.start_frame,
                "end": self.end_frame,
            },
            "time_range": {
                "start": self.start_time,
                "end": self.end_time,
            },
            "start_sec": self.start_sec,
            "end_sec": self.end_sec,
            "caption_preview": self.segment_caption,
            "fps": self.fps,
            "score": self.score,
        }

    @staticmethod
    def quick_format(
        tool_name: str,
        tool_kwargs: dict,
        handle_id: str,
        items: Sequence[BaseInterface],
    ) -> dict[str, Any]:
        """Format items for quick display. Returns JSON dict."""
        items = [i for i in items if isinstance(i, SegmentInterface)]
        top_5 = sorted(items, key=lambda x: x.score, reverse=True)[:5]

        return {
            "view_mode": "quick",
            "handle_id": handle_id,
            "tool_name": tool_name,
            "tool_kwargs": tool_kwargs,
            "result_type": "segment",
            "total": len(items),
            "top_results": [item.to_socket_format() for item in top_5],
        }

    @staticmethod
    def statistic_format(
        tool_name: str,
        tool_kwargs: dict,
        handle_id: str,
        items: Sequence[BaseInterface],
        group_by: str,
    ) -> dict[str, Any]:
        """Format items as statistics grouped by a field. Returns JSON dict."""
        current_items = [i for i in items if isinstance(i, SegmentInterface)]
        grouped_data = defaultdict(list)

        for item in current_items:
            if group_by == "video_id":
                key = item.related_video_id
            elif group_by == "score_bucket":
                bucket = int(item.score * 10) / 10.0
                key = f"{bucket:.1f}-{bucket + 0.1:.1f}"
            else:
                key = "Unknown"
            grouped_data[key].append(item)

        groups = []
        sorted_keys = sorted(grouped_data.keys(), reverse=(group_by == "score_bucket"))

        for key in sorted_keys:
            group_items = grouped_data[key]
            count = len(group_items)
            avg_score = mean([x.score for x in group_items]) if group_items else 0.0
            best_item = max(group_items, key=lambda x: x.score)

            groups.append({
                "key": key,
                "count": count,
                "avg_score": round(avg_score, 3),
                "best_match": best_item.to_socket_format(),
            })

        return {
            "view_mode": "statistics",
            "result_type": "segment",
            "handle_id": handle_id,
            "tool_name": tool_name,
            "tool_kwargs": tool_kwargs,
            "group_by": group_by,
            "total_items": len(current_items),
            "groups": groups,
        }
