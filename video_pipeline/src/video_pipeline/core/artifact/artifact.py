from pydantic import BaseModel, Field
from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING, BinaryIO
from uuid import uuid4

if TYPE_CHECKING:
    from video_pipeline.core.storage.pg_tracker import ArtifactPersistentVisitor


class BaseArtifact(ABC, BaseModel):
    artifact_id: str = Field(default_factory=lambda: str(uuid4()))
    artifact_type: str = Field(init=False)
    lineage_parents: list[str] = Field(init=False)
    user_id: str
    metadata: dict[str, Any] | None = Field(default=None)

    object_name: str | None = Field(default=None)

    def model_post_init(self, __context: Any) -> None:
        self.artifact_type = self.__class__.__name__
        self.lineage_parents = self._build_lineage_parents()
        self.object_name = self.construct_object_name()

    @abstractmethod
    def _build_lineage_parents(self) -> list[str]:
        raise NotImplementedError

    def construct_object_name(self) -> str | None:
        """
        Override this method if artifact has a file uploaded to minio
        """
        return None

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
    user_bucket: str

    def _build_lineage_parents(self) -> list[str]:
        return [self.related_video_id]

    @property
    def minio_url_path(self) -> str:
        """
        This artifact does not store anything in minio
        """
        return ""


class ASRArtifact(BaseArtifact):
    related_video_id: str = Field(
        ..., description="Which video id does this autoshot artifact belong to"
    )
    related_video_minio_url: str
    related_video_extension: str
    related_video_fps: float

    def _build_lineage_parents(self) -> list[str]:
        return [self.related_video_id]

    def construct_object_name(self) -> str:
        return f"asr/{self.related_video_id}.json"


class ImageArtifact(BaseArtifact):
    frame_index: int
    extension: str
    related_video_id: str
    related_video_minio_url: str
    related_video_extension: str
    related_video_fps: float
    timestamp: str
    autoshot_artifact_id: str
    user_bucket: str
    content_type: str

    def _build_lineage_parents(self) -> list[str]:
        return [self.autoshot_artifact_id]

    def construct_object_name(self) -> str:
        return f"images/{self.related_video_id}/{self.frame_index:08d}_{self.timestamp}{self.extension}"


class SegmentCaptionArtifact(BaseArtifact):
    artifact_type: str
    autoshot_artifact_id: str
    asr_artifact_id: str
    related_video_extension: str
    related_video_id: str
    related_video_fps: float
    start_frame: int
    end_frame: int
    start_timestamp: str
    end_timestamp: str
    related_asr: str
    related_video_minio_url: str

    def _build_lineage_parents(self) -> list[str]:
        return [pid for pid in (self.autoshot_artifact_id, self.asr_artifact_id) if pid]

    def construct_object_name(self) -> str:
        return f"caption/segment/{self.related_video_id}/{self.start_frame}_{self.end_frame}_{self.start_timestamp}_{self.end_timestamp}.json"


class ImageCaptionArtifact(BaseArtifact):
    frame_index: int
    time_stamp: str
    related_video_id: str
    related_video_fps: float
    extension: str
    image_minio_url: str
    image_id: str

    def construct_object_name(self) -> str:
        return f"caption/image/{self.related_video_id}/{self.frame_index:08d}_{self.time_stamp}.json"

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

    def construct_object_name(self) -> str:
        return f"embedding/image/{self.related_video_id}/{self.frame_index:08d}_{self.time_stamp}.npy"


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

    def construct_object_name(self) -> str | None:
        return f"embedding/image_caption/{self.related_video_id}/{self.frame_index:08d}_{self.time_stamp}.npy"


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

    def construct_object_name(self) -> str | None:
        return f"embedding/caption_segment/{self.related_video_id}/{self.start_frame}_{self.end_frame}_{self.start_time}_{self.end_time}.npy"
