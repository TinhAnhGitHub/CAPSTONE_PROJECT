from __future__ import annotations
from pydantic import Field, BaseModel
from ingestion.core.artifact.schema import SegmentCaptionArtifact, ImageCaptionArtifact


class VideoInterface(BaseModel):
    """
    This will be the interface of a video
    """
    video_id: str
    fps: float
    duration: str

    def __str__(self) -> str:
        return super().__str__()


class SegmentObjectInterface(BaseModel):
    """
    Represents a temporally localized **video segment** identified via semantic or event-level retrieval.
    """

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


    def expr(self) -> str:
        """
        Structured, agent-readable summary of the segment object.
        Designed for reasoning agents or retrievers to parse easily.
        """
        return (
            f"SegmentObject("
            f"video_id='{self.related_video_id}', "
            f"frames={self.start_frame_index}-{self.end_frame_index}, "
            f"time='{self.start_time}->{self.end_time}', "
            f"caption='{self.caption_info}', "
            f"score={self.score if self.score is not None else 'N/A'}, "
            f"query='{self.segment_caption_query if self.segment_caption_query else 'N/A'}'"
            f")"
        )
    
    


class ImageObjectInterface(BaseModel):
    """
    Represents an **image** or single **video frame** retrieved through semantic or multimodal search.
    """

    related_video_id: str = Field(..., description="Identifier of the video that this image belongs to.")
    frame_index: int = Field(..., description="Frame index within the source video.")
    timestamp: str = Field(..., description="Timestamp of this frame in the video (e.g., '00:00:12.87').")
    caption_info: str | None= Field(None, description="Descriptive caption summarizing the image content.")
    minio_path: str = Field(..., description="Storage path or URL for this image (e.g., MinIO or S3).")

    score: float | None = Field(
        None,
        description=(
            "Normalized similarity score between this image and the retrieval query. "
            "Higher scores imply stronger semantic correlation. "
            "If `None`, the image was generated or retrieved outside of a semantic search process."
        ),
    )

    query: list[str] | str | None = Field(
        None,
        description=(
            "The search query or queries that led to the retrieval of this image. "
            "Can be a single text query or a list for multimodal inputs (e.g., visual + textual). "
            "If `None`, the image was not semantically retrieved."
        ),
    )
    reference_query_image: ImageObjectInterface | None = Field(
        None,
        description=(
            "If this image was retrieved using another image as the query, "
            "this field holds that reference image's metadata."
        )
    )

    def expr(self) -> str:
        """
        Structured, agent-readable representation of the image object.
        """
        query_repr = (
            f"{self.query}"
            if self.query
            else "N/A"
        )
        return (
            f"ImageObject("
            f"video_id='{self.related_video_id}', "
            f"frame={self.frame_index}, "
            f"timestamp='{self.timestamp}', "
            f"caption='{self.caption_info}', "
            f"path='{self.minio_path}', "
            f"score={self.score if self.score is not None else 'N/A'}, "
            f"query={query_repr}"
            f")"
        )





