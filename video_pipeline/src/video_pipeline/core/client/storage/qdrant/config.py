from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, BaseModel
from qdrant_client.models import Distance, HnswConfigDiff, QuantizationConfig


class QdrantConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="QDRANT_", case_sensitive=False, extra="ignore"
    )

    port: int
    host: str
    collection_name: str
    timeout: int
    use_grpc: bool = True
    prefer_grpc: bool = True


class QdrantIndexConfig(BaseModel):
    vector_size: int
    distance: Distance
    on_disk: bool
    hnsw_config: HnswConfigDiff
    quantization_config: QuantizationConfig
    is_sparse: bool = Field(default=False)
