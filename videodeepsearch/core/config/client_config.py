from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict

class ImageMilvusConfig(BaseSettings):
    model_config = SettingsConfigDict(
        case_sensitive=False,
        env_prefix="IMAGE_MILVUS_"
    )
    uri: str
    collection_name: str 
    visual_param: dict
    caption_param: dict
    sparse_param: dict

class SegmentCaptionImageMilvusConfig(BaseSettings):
    model_config = SettingsConfigDict(
        case_sensitive=False,
        env_prefix="SEGMENT_MILVUS_"
    )
    caption_uri: str
    collection_name: str
    dense_param: dict
    sparse_param: dict

class PostgresClientConfig(BaseSettings):
    model_config = SettingsConfigDict(
        case_sensitive=False,
        env_prefix="POSTGRES_CLIENT_"
    )
    database_url: str

class MinioStorageClientConfig(BaseSettings):
    model_config = SettingsConfigDict(
        case_sensitive=False,
        env_prefix='MINIO_STORAGE_CLIENT_'
    )
    host: str
    port: str
    access_key: str
    secret_key: str
    secure: bool

class ExternalImageEmbeddingConfig(BaseSettings):
    model_config = SettingsConfigDict(
        case_sensitive=False,
        env_prefix="EXTERNAL_IMAGE_EMBEDDING_CLIENT_"
    )
    base_url: str
    model_name: str
    device: Literal['cuda', 'cpu']
    batch_size: int = 32

class ExternaltTextEmbeddingConfig(BaseSettings):
    model_config = SettingsConfigDict(
        case_sensitive=False,
        env_prefix="EXTERNAL_TEXT_EMBEDDING_CLIENT_"
    )
    base_url: str
    model_name: str
    device: Literal['cuda', 'cpu']
    batch_size: int = 32
    
image_milvus_config = ImageMilvusConfig() #type:ignore
segment_caption_milvus_config = SegmentCaptionImageMilvusConfig() #type:ignore
postgres_client_config = PostgresClientConfig() #type:ignore
minio_storage_client_config = MinioStorageClientConfig() #type:ignore
external_image_embedding_config = ExternalImageEmbeddingConfig() #type:ignore
external_text_embedding_config = ExternaltTextEmbeddingConfig() #type:ignore