from __future__ import annotations
from pydantic import Field, BaseModel
from abc import ABC, abstractmethod

from videodeepsearch.tools.schema.result import ArtifactVisitor, ReturnType

class BaseArtifact(BaseModel, ABC):
    """
    This is the base artifact.
    Every tool's output that related to the video's information must be from this artifact class
    """
    type_artifact: str
    minio_path: str

    @abstractmethod
    def accept(self, visitor: ArtifactVisitor[ReturnType]) -> ReturnType:
        """Accept a visitor for processing"""
        pass

class VideoInterface(BaseArtifact):
    """
    This will be the interface of a video
    """
    video_id: str
    type_artifact: str = "VideoInterface"
    fps: float
    duration: str
    def accept(self, visitor: ArtifactVisitor[ReturnType]) -> ReturnType:
        return visitor.visit_video(self)

    

class SegmentObjectInterface(BaseArtifact):
    """
    Represents a temporally localized **video segment** identified via semantic or event-level retrieval.
    """
    type_artifact: str = "SegmentObjectInterface"
    
    related_video_id: str = Field(..., description="Unique identifier of the video containing this segment.")
    start_frame_index: int = Field(..., description="Index of the first frame in the segment.")
    end_frame_index: int = Field(..., description="Index of the last frame in the segment.")
    start_time: str = Field(..., description="Start time of the segment in HH:MM:SS.sss format.")
    end_time: str = Field(..., description="End time of the segment in HH:MM:SS.sss format.")
    caption_info: str = Field(..., description="Semantic or descriptive caption summarizing the segment's content.")

    score: float | None = Field(
        None,
        description=(
            "Normalized similarity score (e.g., cosine similarity) between this segment and the query. "
            "Higher scores indicate stronger semantic alignment. "
            "If `None`, the segment was obtained outside of semantic retrieval (e.g., via procedural or scan-based tools)."
        ),
    )

    segment_caption_query: str | None = Field(
        None,
        description=(
            "The textual or multimodal query that retrieved this segment. "
            "If `None`, the segment was not produced by a semantic search process."
        ),
    )
    def accept(self, visitor: ArtifactVisitor[ReturnType]) -> ReturnType:
        return visitor.visit_segment(self)
        
class ImageObjectInterface(BaseArtifact):
    """
    Represents an **image** or single **video frame** retrieved through semantic or multimodal search.
    """
    type_artifact: str = "ImageObjectInterface"

    related_video_id: str = Field(..., description="Identifier of the video that this image belongs to.")
    frame_index: int = Field(..., description="Frame index within the source video.")
    timestamp: str = Field(..., description="Timestamp of this frame in the video (e.g., '00:00:12.87').")
    caption_info: str | None= Field(None, description="Descriptive caption summarizing the image content.")

    score: float | None = Field(
        None,
        description=(
            "Normalized similarity score between this image and the retrieval query. "
            "Higher scores imply stronger semantic correlation. "
            "If `None`, the image was generated or retrieved outside of a semantic search process."
        ),
    )

    query: list[str] | None = Field(
        None,
        description=(
            "The search query or queries that led to the retrieval of this image. "
            "Can be a single text query or a list for multimodal inputs (e.g., visual + textual). "
            "If `None`, the image was not semantically retrieved."
        ),
    )

    def accept(self, visitor: ArtifactVisitor[ReturnType]) -> ReturnType:
        return visitor.visit_image(self)
        

ARTIFACT_MODELS = (VideoInterface, SegmentObjectInterface, ImageObjectInterface)


