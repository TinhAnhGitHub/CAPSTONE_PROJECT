from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class ImageMilvusConfig(BaseSettings):
    model_config = SettingsConfigDict(
        case_sensitive=False,
        env_prefix="IMAGE_MILVUS_"
    )
    image_milvus_uri: str
    image_milvus_collection_name: str 
    image_milvus_visual_param: dict
    image_milvus_caption_param: dict
    image_milvus_sparse_param: dict

class SegmentCaptionImageMilvusConfig(BaseSettings):
    model_config = SettingsConfigDict(
        case_sensitive=False,
        env_prefix="SEGMENT_MILVUS_"
    )
    segment_milvus_caption_uri: str
    segment_milvus_collection_name: str
    segment_milvus_dense_param: dict
    segment_milvus_sparse_param: dict

class PostgresClientConfig(BaseSettings):
    model_config = SettingsConfigDict(
        case_sensitive=False,
        env_prefix="POSTGRES_CLIENT_"
    )
    postgres_client_database_url: str

class MinioStorageClientConfig(BaseSettings):
    model_config = SettingsConfigDict(
        case_sensitive=False,
        env_prefix='MINIO_STORAGE_CLIENT_'
    )
    minio_storage_client_host: str
    minio_storage_client_port: str
    minio_storage_client_access_key: str
    minio_storage_client_secret_key: str
    minio_storage_client_secure: bool

class ExternalImageEmbeddingConfig(BaseSettings):
    model_config = SettingsConfigDict(
        case_sensitive=False,
        env_prefix="EXTERNAL_IMAGE_EMBEDDING_CLIENT_"
    )
    external_image_embedding_client_base_url: str
    external_image_embedding_client_model_name: str
    external_image_embedding_client_device: Literal['cuda', 'cpu']
    external_image_embedding_client_batch_size: int = 32

class ExternaltTextEmbeddingConfig(BaseSettings):
    model_config = SettingsConfigDict(
        case_sensitive=False,
        env_prefix="EXTERNAL_TEXT_EMBEDDING_CLIENT_"
    )
    external_text_embedding_client_base_url: str
    external_text_embedding_client_model_name: str
    external_text_embedding_client_device: Literal['cuda', 'cpu']
    external_text_embedding_client_batch_size: int = 32

class GreetingAgentLLM(BaseSettings):
    

image_milvus_config = ImageMilvusConfig() #type:ignore
segment_caption_milvus_config = SegmentCaptionImageMilvusConfig() #type:ignore
postgres_client_config = PostgresClientConfig() #type:ignore
minio_storage_client_config = MinioStorageClientConfig() #type:ignore
external_image_embedding_config = ExternalImageEmbeddingConfig() #type:ignore
external_text_embedding_config = ExternaltTextEmbeddingConfig() #type:ignore