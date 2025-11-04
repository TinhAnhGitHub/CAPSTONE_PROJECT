from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Dict, Any

class MilvusIndexBaseConfig(BaseSettings):
    """Base schema shared across Milvus index types."""
    type_config: str
    dimension: int
    metric_type: str
    index_type: str
    description: str = ""
    index_params: Dict[str, Any] = Field(default_factory=dict)

    model_config = SettingsConfigDict(
        case_sensitive=False,
        validate_assignment=True,
        extra="ignore",
    )

class ImageVisualDenseIndexConfig(MilvusIndexBaseConfig):
    model_config = SettingsConfigDict(env_prefix="IMAGE_VISUAL_")

class ImageCaptionDenseIndexConfig(MilvusIndexBaseConfig):
    model_config = SettingsConfigDict(env_prefix="IMAGE_CAPTION_DENSE_")

class ImageCaptionSparseIndexConfig(MilvusIndexBaseConfig):
    model_config = SettingsConfigDict(env_prefix="IMAGE_CAPTION_SPARSE_")

class SegmentCaptionDenseIndexConfig(MilvusIndexBaseConfig):
    model_config = SettingsConfigDict(env_prefix="SEGMENT_CAPTION_DENSE_")

class SegmentCaptionSparseIndexConfig(MilvusIndexBaseConfig):
    model_config = SettingsConfigDict(env_prefix="SEGMENT_CAPTION_SPARSE_")


image_visual_dense_conf = ImageVisualDenseIndexConfig()#type:ignore
image_caption_dense_conf = ImageCaptionDenseIndexConfig()#type:ignore
image_caption_sparse_conf = ImageCaptionSparseIndexConfig()#type:ignore
segment_caption_dense_conf = SegmentCaptionDenseIndexConfig()#type:ignore
segment_caption_sparse_conf = SegmentCaptionSparseIndexConfig()#type:ignore

