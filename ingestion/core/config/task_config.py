from typing import Literal, Any
from pydantic import Field
from datetime import timedelta

from pydantic_settings import BaseSettings, SettingsConfigDict


class ConsulClientConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CONSUL_",
        case_sensitive=False,
        validate_assignment=True,
        extra="ignore",
    )
    timeout_seconds: float
    max_retries: int
    retry_min_wait: float
    retry_max_wait: float
    consul_host: str
    consul_port: int

class AutoshotTaskConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AUTOSHOT_",
        case_sensitive=False,
        validate_assignment=True,
        extra="ignore",
    )
    model_name: str
    device: str

class ASRTaskConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ASR_",
        case_sensitive=False,
        validate_assignment=True,
        extra="ignore",
    )
    model_name: str
    device: str

class ImageProcessingTaskConfig(BaseSettings):
    """Configuration for image extraction task."""
    num_img_per_segment: int = Field(default=3, ge=1)
    model_config = SettingsConfigDict(
        env_prefix="IMAGE_",
        case_sensitive=False,
        validate_assignment=True,
        extra="ignore",
    )

class LLMTaskConfig(BaseSettings):
    image_per_segments: int = Field(default=5, ge=1)
    model_name:str
    device: str
    model_config = SettingsConfigDict(
        env_prefix="LLM_",
        case_sensitive=False,
        validate_assignment=True,
        extra="ignore",
    )

class ImageEmbeddingTaskConfig(BaseSettings):
    batch_size: int = Field(default=32, ge=1)
    model_name:str
    device: str
    model_config = SettingsConfigDict(
        env_prefix="IMAGE_EMBEDDING_",
        case_sensitive=False,
        validate_assignment=True,
        extra="ignore",
    )



class TextEmbeddingTaskConfig(BaseSettings):
    batch_size: int = Field(default=32, ge=1)
    model_name:str
    device: str
    model_config = SettingsConfigDict(
        env_prefix="TEXT_EMBEDDING_",
        case_sensitive=False,
        validate_assignment=True,
        extra="ignore",
    )








tautoshot_conf = AutoshotTaskConfig()  # type: ignore
tasr_conf = ASRTaskConfig()            # type: ignore
timage_processing_conf = ImageProcessingTaskConfig()  # type: ignore
tllm_conf = LLMTaskConfig()            # type: ignore
t_i_embed_conf = ImageEmbeddingTaskConfig()  # type: ignore
t_t_embed_conf = TextEmbeddingTaskConfig()   # type: ignore
consule_conf = ConsulClientConfig() #type:ignore