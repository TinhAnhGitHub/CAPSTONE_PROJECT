from __future__ import annotations
from pydantic import BaseModel, Field
from typing import BinaryIO, Any, Literal
from datetime import datetime
import hashlib
from abc import ABC, abstractmethod
from uuid import uuid4
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .persist import ArtifactPersistentVisitor


class BaseArtifact(ABC, BaseModel):
    
    @abstractmethod 
    def accept_upload(self, visitor: Any, upload_file: Any):
        raise NotImplementedError

    @abstractmethod
    def accept_check_exist(self, visitor: Any):
        raise NotImplementedError

    @property
    @abstractmethod
    def object_key(self)->str:
        raise NotImplementedError

    @property
    @abstractmethod
    def minio_url_path(self) -> str:
        raise NotImplementedError

class VideoArtifact(BaseArtifact):
    artifact_type: str 
    video_id: str
    video_minio_url: str
    video_extension: str
    object_name:str
    user_bucket: str
    fps: float


    def __post_init__(self):
        self.artifact_type = self.__class__.__name__

    def accept_upload(self, visitor: "ArtifactPersistentVisitor", upload_file: dict):
        return visitor.visit_video(self, upload_file) 

    async def accept_check_exist(self, visitor: "ArtifactPersistentVisitor") -> bool:
        return await visitor._check_exist(self, self.user_bucket, check_minio=False)

    @property
    def artifact_id(self):
        return self.video_id
    @property
    def object_key(self):
        return self.object_name
    
    @property
    def minio_url_path(self):
        return self.video_minio_url

  

class AutoshotArtifact(BaseArtifact):
    """
    This will hold the artifact segments after the autoshot processing
    """
    artifact_type: str
    related_video_id: str = Field(..., description="Which video id does this autoshot artifact belong to")
    related_video_minio_url: str
    related_video_extension: str
    related_video_fps: float
    user_bucket: str

    artifact_id: str = Field(default_factory=lambda: str(uuid4()))
    def __post_init__(self):
        self.artifact_type = self.__class__.__name__


    def accept_upload(self, visitor: "ArtifactPersistentVisitor", upload_file: list):
        return visitor.visit_segments(self, upload_file)
    
    async def accept_check_exist(self, visitor: "ArtifactPersistentVisitor"):
        return await visitor._check_exist(self, self.user_bucket)
    
    @property
    def object_key(self) -> str:
        return f"autoshot/{self.related_video_id}.json"

 
    @property
    def minio_url_path(self)->str:
        return f"s3://{self.user_bucket}/{self.object_key}"

class ASRArtifact(BaseArtifact):
    artifact_type: str  

    related_video_id: str = Field(..., description="Which video id does this autoshot artifact belong to")
    related_video_minio_url: str
    related_video_extension: str
    related_video_fps: float

    user_bucket: str
    artifact_id: str = Field(default_factory=lambda: str(uuid4()))
    
    def __post_init__(self):
        self.artifact_type = self.__class__.__name__

    def accept_upload(self, visitor: "ArtifactPersistentVisitor", upload_file: dict):
        return visitor.visit_asr(self, upload_file)
    
    async def accept_check_exist(self, visitor: "ArtifactPersistentVisitor"):
        return await visitor._check_exist(self, self.user_bucket)

    @property
    def object_key(self) -> str:
        return f"asr/{self.related_video_id}.json"

    @property
    def minio_url_path(self)->str:
        return f"s3://{self.user_bucket}/{self.object_key}"

class ImageArtifact(BaseArtifact):
    artifact_type:str
    frame_index: int
    extension: str
    related_video_id: str
    related_video_minio_url: str
    related_video_extension: str
    related_video_fps: float
    timestamp: str
    autoshot_artifact_id: str
    user_bucket: str
    metadata: dict
    content_type: str

    artifact_id: str = Field(default_factory=lambda: str(uuid4()))


    def __post_init__(self):
        self.artifact_type = self.__class__.__name__
    

    def accept_upload(self, visitor: "ArtifactPersistentVisitor", upload_file: BinaryIO):
        return visitor.visit_image(self, upload_file)
    
    async def accept_check_exist(self, visitor: "ArtifactPersistentVisitor"):
        return await visitor._check_exist(self, self.user_bucket)

    @property
    def object_key(self) -> str:
        return f"images/{self.related_video_id}/{self.frame_index:08d}_{self.timestamp}{self.extension}"

    @property
    def minio_url_path(self)->str:
        return f"s3://{self.user_bucket}/{self.object_key}"


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
    user_bucket: str

    artifact_id: str = Field(default_factory=lambda: str(uuid4()))

    def __post_init__(self):
        self.artifact_type = self.__class__.__name__

    
    def accept_upload(self, visitor: "ArtifactPersistentVisitor", upload_file: str):
        return visitor.visit_segment_caption(self, upload_file)
    
    async def accept_check_exist(self, visitor: "ArtifactPersistentVisitor") -> bool:
        return await visitor._check_exist(self, self.user_bucket)

    
    @property
    def object_key(self) -> str:
        return f"caption/segment/{self.related_video_id}/{self.start_frame}_{self.end_frame}_{self.start_timestamp}_{self.end_timestamp}.json"

    @property
    def minio_url_path(self)->str:
        return f"s3://{self.user_bucket}/{self.object_key}"


    @property
    def lineage_parents(self) -> list[str]:
        return [pid for pid in (self.autoshot_artifact_id, self.asr_artifact_id) if pid]
    
class ImageCaptionArtifact(BaseArtifact):
    artifact_type: str
    
    frame_index: int
    time_stamp: str
    related_video_id: str
    related_video_fps: float
    extension: str
    user_bucket: str
    image_minio_url: str
    image_id: str

    artifact_id: str = Field(default_factory=lambda: str(uuid4()))


    def __post_init__(self):
        self.artifact_type = self.__class__.__name__

    def accept_upload(self, visitor: "ArtifactPersistentVisitor", upload_file: str):
        return visitor.visit_image_caption(self, upload_file)
    
    async def accept_check_exist(self, visitor: "ArtifactPersistentVisitor") -> bool:
        return await visitor._check_exist(self, self.user_bucket)
    

    @property
    def object_key(self) -> str:
        return f"caption/image/{self.related_video_id}/{self.frame_index:08d}_{self.time_stamp}.json"

    @property
    def minio_url_path(self)->str:
        return f"s3://{self.user_bucket}/{self.object_key}"

    
class ImageEmbeddingArtifact(BaseArtifact):
    artifact_type: str
    time_stamp: str
    frame_index: int
    related_video_id: str
    related_video_fps: float
    user_bucket: str
    image_minio_url: str

    extension: str
    image_id: str

    artifact_id: str = Field(default_factory=lambda: str(uuid4()))

    def __post_init__(self):
        self.artifact_type = self.__class__.__name__

    def accept_upload(self, visitor: "ArtifactPersistentVisitor", upload_file: BinaryIO):
        return visitor.visit_image_embedding(self, upload_file)
    
    async def accept_check_exist(self, visitor: "ArtifactPersistentVisitor") -> bool:
        return await visitor._check_exist(self, self.user_bucket)
    

    @property
    def object_key(self) -> str:
        return f"embedding/image/{self.related_video_id}/{self.frame_index:08d}_{self.time_stamp}.npy"

    @property
    def minio_url_path(self)->str:
        return f"s3://{self.user_bucket}/{self.object_key}"


class TextCaptionEmbeddingArtifact(BaseArtifact):
    artifact_type: str
    time_stamp: str
    related_frame_fps:float
    frame_index: int
    related_video_id: str
    image_caption_minio_url: str
    user_bucket:str
    caption_id: str
    image_id: str
    image_minio_url: str

    artifact_id: str = Field(default_factory=lambda: str(uuid4()))

    def __post_init__(self):
        self.artifact_type = self.__class__.__name__


    def accept_upload(self, visitor: "ArtifactPersistentVisitor", upload_file: BinaryIO):
        return visitor.visit_image_caption_embedding(self, upload_file)
    
    async def accept_check_exist(self, visitor: "ArtifactPersistentVisitor") -> bool:
        return await visitor._check_exist(self, self.user_bucket)
    

    @property
    def object_key(self) -> str:
        return f"embedding/image_caption/{self.related_video_id}/{self.frame_index:08d}_{self.time_stamp}.npy"

    @property
    def minio_url_path(self)->str:
        return f"s3://{self.user_bucket}/{self.object_key}"

class TextCapSegmentEmbedArtifact(BaseArtifact):
    artifact_type: str
    
    related_video_fps: float
    related_video_id: str

    start_frame: int
    end_frame: int
    start_time:str
    end_time:str

    related_segment_caption_url: str
    user_bucket:str
    segment_cap_id: str

    artifact_id: str = Field(default_factory=lambda: str(uuid4()))

    def __post_init__(self):
        self.artifact_type = self.__class__.__name__
        
    def accept_upload(self, visitor: "ArtifactPersistentVisitor", upload_file: BinaryIO):
        return visitor.visit_segment_caption_embedding(self, upload_file)
    
    async def accept_check_exist(self, visitor: "ArtifactPersistentVisitor") -> bool:
        return await visitor._check_exist(self, self.user_bucket)
    

    @property
    def object_key(self) -> str:
        return f"embedding/caption_segment/{self.related_video_id}/{self.start_frame}_{self.end_frame}_{self.start_time}_{self.end_time}.npy"

    @property
    def minio_url_path(self)->str:
        return f"s3://{self.user_bucket}/{self.object_key}"

