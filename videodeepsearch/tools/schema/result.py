from __future__ import annotations
from typing import Protocol, TypeVar, Generic, Callable
from typing import Any
from pydantic import BaseModel, Field
from abc import ABC, abstractmethod

from videodeepsearch.tools.schema.artifact import VideoInterface, SegmentObjectInterface, ImageObjectInterface

T = TypeVar('T', bound=BaseModel)
ReturnType = TypeVar('ReturnType', covariant=True)


class ArtifactVisitor(Protocol[ReturnType]):
    
    def visit_video(self, video: VideoInterface) -> ReturnType:
        ...
    
    def visit_segment(self, segment: SegmentObjectInterface) -> ReturnType:
        ...
    
    def visit_image(self, image: ImageObjectInterface) -> ReturnType:
        ...


# Concrete Implementation for the Protocol
class ScoreFilter(BaseModel):
    min_score_filter: float = Field(description="The score to filter", le=1.0, ge=0.0)

    def visit_video(self, video: VideoInterface) -> bool:
        return True

    def visit_segment(self, segment: SegmentObjectInterface) -> bool:
        return segment.score is not None and segment.score >= self.min_score_filter

    def visit_image(self, image: ImageObjectInterface) -> bool:
        return image.score is not None and image.score >= self.min_score_filter
    

class SummaryBuilder:
    max_length_cap_review: int = Field(default=50, description="The maximum caption length to preview")
    
    def visit_video(self, video: VideoInterface) -> str:
        return (
            f"Type: {VideoInterface.__name__}"
            f"video id: {video.video_id}"
            f"Video FPS: {video.fps}"
            f"Duration of video: {video.duration}"
            f"Video minio path: {video.minio_path}"
        )
    
    def visit_segment(self, segment: SegmentObjectInterface) -> str:
        return (
            f"Type: {SegmentObjectInterface.__name__}"
            f"Belong to video id: {segment.related_video_id}"
            f"Time range: {segment.start_time} - {segment.end_time}"
            f"Caption preview: {segment.caption_info[:self.max_length_cap_review] + "..." if len(segment.caption_info) > self.max_length_cap_review else segment.caption_info}"
            f"The query used for the retrieval: [{segment.caption_info}]"
            f"The semantic score: {segment.score}"
            f"The minio path: {segment.minio_path}"
        )
    
    def visit_image(self, image: ImageObjectInterface) -> str:
        return (
            f"Type: {ImageObjectInterface.__name__} "
            f"Belong to video id: {image.related_video_id} "
            f"Timestamp: {image.timestamp} "
            f"Caption preview: "
            f"{(image.caption_info[:self.max_length_cap_review] + '...') if image.caption_info and len(image.caption_info) > self.max_length_cap_review else (image.caption_info or 'No caption')}"
            f"The query used for the retrieval: {', '.join(image.query) if image.query else 'No query'}"
            f"The semantic score: {image.score}"
            f"The minio path: {image.minio_path}"
        )
    



