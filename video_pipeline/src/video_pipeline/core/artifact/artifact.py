from pydantic import BaseModel, Field, computed_field
from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING, BinaryIO, Protocol
from uuid import uuid4

if TYPE_CHECKING:
    from video_pipeline.core.storage.pg_tracker import ArtifactPersistentVisitor


class BaseArtifact(ABC, BaseModel):
    artifact_id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    metadata: dict[str, Any] | None = Field(default=None)
    object_name: str | None = Field(default=None)

    @computed_field
    @property
    def artifact_type(self) -> str:
        return self.__class__.__name__

    @computed_field
    @property
    def lineage_parents(self) -> list[str]:
        return self._build_lineage_parents()

    @abstractmethod
    def _build_lineage_parents(self) -> list[str]:
        raise NotImplementedError

    @property
    def minio_url_path(self) -> str:
        return f"s3://{self.user_id}/{self.object_name}"

    def accept_upload(
        self, visitor: "ArtifactPersistentVisitor", upload_file: None | BinaryIO = None
    ):
        return visitor.visit_artifact(self, upload_to_minio=upload_file)

    async def accept_check_exist(self, visitor: "ArtifactPersistentVisitor") -> bool:
        return await visitor._check_exist(self)
    

class VideoArtifact(BaseArtifact):
    video_id: str
    video_minio_url: str
    video_extension: str
    fps: float

    def _build_lineage_parents(self) -> list[str]:
        return []

    @property
    def minio_url_path(self) -> str:
        return self.video_minio_url


class AutoshotArtifact(BaseArtifact):
    related_video_id: str = Field(
        ..., description="Which video id does this autoshot artifact belong to"
    )
    related_video_minio_url: str
    related_video_extension: str
    related_video_fps: float

    def _build_lineage_parents(self) -> list[str]:
        return [self.related_video_id]

    @property
    def minio_url_path(self) -> str:
        """
        This artifact does not store anything in minio
        """
        return ""


class ASRArtifact(BaseArtifact):
    related_autoshot_artifact_id: str = Field(
        ..., description="Which video id does this autoshot artifact belong to"
    )
    related_video_minio_url: str
    related_video_extension: str
    related_video_fps: float
    related_video_id: str = ""

    def _build_lineage_parents(self) -> list[str]:
        return [self.related_autoshot_artifact_id]

    @property
    def minio_url_path(self) -> str:
        """ASR artifact does not store anything in MinIO."""
        return ""


class AudioSegmentArtifact(BaseArtifact):
    asr_artifact_ids: list[str] = Field(default_factory=list)
    related_video_id: str
    related_video_minio_url: str = ""
    related_video_extension: str = ""
    related_video_fps: float = 0.0
    segment_index: int
    start_sec: float
    end_sec: float
    start_timestamp: str
    end_timestamp: str
    audio_text: str = ""
    start_frame: int = 0
    end_frame: int = 0

    def _build_lineage_parents(self) -> list[str]:
        return self.asr_artifact_ids

    @property
    def minio_url_path(self) -> str:
        """AudioSegment artifact does not store anything in MinIO."""
        return ""


class SegmentEmbeddingArtifact(BaseArtifact):
    related_audio_segment_artifact_id: str
    related_video_id: str
    related_video_minio_url: str
    related_video_extension: str
    related_video_fps: float
    start_frame: int
    end_frame: int
    start_timestamp: str
    end_timestamp: str
    frame_indices: list[int]
    embedding_dim: int = 1536

    def _build_lineage_parents(self) -> list[str]:
        return [self.related_audio_segment_artifact_id]


class SegmentCaptionArtifact(BaseArtifact):
    related_audio_segment_artifact_id: str
    related_video_id: str
    related_video_minio_url: str = ""
    related_video_extension: str
    related_video_fps: float
    start_frame: int
    end_frame: int
    start_timestamp: str
    end_timestamp: str
    audio_text: str
    summary_caption: str
    event_captions: list[str] = Field(default_factory=list)

    def _build_lineage_parents(self) -> list[str]:
        return [self.related_audio_segment_artifact_id]


class ImageArtifact(BaseArtifact):
    frame_index: int
    extension: str
    related_video_id: str
    related_video_minio_url: str
    related_video_extension: str
    related_video_fps: float
    timestamp: str
    autoshot_artifact_id: str
    content_type: str

    def _build_lineage_parents(self) -> list[str]:
        return [self.autoshot_artifact_id]


class ImageOCRArtifact(BaseArtifact):
    frame_index: int
    time_stamp: str
    related_video_id: str
    related_video_fps: float
    extension: str
    image_minio_url: str
    image_id: str

    def _build_lineage_parents(self) -> list[str]:
        return [self.image_id]


class ImageCaptionArtifact(BaseArtifact):
    frame_index: int
    time_stamp: str
    related_video_id: str
    related_video_fps: float
    extension: str
    image_minio_url: str
    image_id: str

    def _build_lineage_parents(self) -> list[str]:
        return [self.image_id]


class ImageEmbeddingArtifact(BaseArtifact):
    time_stamp: str
    frame_index: int
    related_video_id: str
    related_video_fps: float
    image_minio_url: str
    extension: str
    image_id: str

    def _build_lineage_parents(self) -> list[str]:
        return [self.image_id]


class TextCaptionEmbeddingArtifact(BaseArtifact):
    time_stamp: str
    related_frame_fps: float
    frame_index: int
    related_video_id: str
    image_caption_minio_url: str
    caption_id: str
    image_id: str
    image_minio_url: str

    def _build_lineage_parents(self) -> list[str]:
        return [self.caption_id]


class ImageCaptionMultimodalEmbeddingArtifact(BaseArtifact):
    time_stamp: str
    related_frame_fps: float
    frame_index: int
    related_video_id: str
    image_caption_minio_url: str
    caption_id: str
    image_id: str
    image_minio_url: str

    def _build_lineage_parents(self) -> list[str]:
        return [self.caption_id]


class TextCapSegmentEmbedArtifact(BaseArtifact):
    related_video_fps: float
    related_video_id: str
    start_frame: int
    end_frame: int
    start_time: str
    end_time: str
    related_segment_caption_url: str
    segment_cap_id: str

    def _build_lineage_parents(self) -> list[str]:
        return [self.segment_cap_id]


class SegmentCaptionMultimodalEmbedArtifact(BaseArtifact):
    """Multimodal embedding artifact for segment captions using QwenVL."""
    related_video_fps: float
    related_video_id: str
    start_frame: int
    end_frame: int
    start_timestamp: str
    end_timestamp: str
    related_segment_caption_url: str
    segment_cap_id: str

    def _build_lineage_parents(self) -> list[str]:
        return [self.segment_cap_id]


##### Graph entities
class EntityDoc(BaseModel):
    video_id: str
    entity_id: str
    entity_name: str
    entity_type: str
    desc: str


class MicroEventDoc(BaseModel):
    video_id: str
    event_id: str
    event_des: str

class RelationshipDoc(BaseModel):
    video_id: str
    subject_id: str
    relation_desc: str
    object_id: str

class GraphArtifact(BaseArtifact):
    """
    Graph Extraction
    """
    related_video_fps: float
    related_video_id: str
    start_frame: int
    end_frame: int
    start_timestamp: str
    end_timestamp: str
    related_segment_caption_id: str

    entities: list[EntityDoc] = Field(default_factory=list)
    events: list[MicroEventDoc] = Field(default_factory=list)
    relationships: list[RelationshipDoc] = Field(default_factory=list)


    def _build_lineage_parents(self) -> list[str]:
        return [self.related_segment_caption_id]
        










