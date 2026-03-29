from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class AgentModelConfig(BaseModel):
    model_id: str
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None


class WorkerModelConfig(BaseModel):
    name: str
    model_id: str
    description: str = ""
    strengths: list[str] = []
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None


class AgentsConfig(BaseModel):
    greeter: AgentModelConfig
    orchestrator: AgentModelConfig
    planning: AgentModelConfig
    llm_tool: AgentModelConfig | None = None  # Separate model for LLM toolkit operations
    summarizer: AgentModelConfig | None = None  


class LLMProviderConfig(BaseModel):
    provider: str = "openrouter"
    base_url: str
    agents: AgentsConfig
    workers: list[WorkerModelConfig] = []

    @property
    def api_key(self) -> str | None:
        return os.environ.get("OPENROUTER_API_KEY")


class PostgresConfig(BaseModel):
    host: str = "localhost"
    port: int = 5432
    database: str = "video-pipeline"
    username: str = "admin123"
    password: str = "admin123"
    url: str | None = None

    @property
    def connection_url(self) -> str:
        if self.url:
            return self.url
        return f"postgresql+asyncpg://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"


class MinioBucketsConfig(BaseModel):
    videos: str = "videos"
    artifacts: str = "video-artifacts"
    asr: str = "asr-data"


class MinioConfig(BaseModel):
    host: str = "localhost"
    port: str = "9000"
    access_key: str = "minioadmin"
    secret_key: str = "minioadmin"
    secure: bool = False
    buckets: MinioBucketsConfig = Field(default_factory=MinioBucketsConfig)


class QdrantConfig(BaseModel):
    host: str = "localhost"
    port: int = 6333
    grpc_port: int = 6334
    prefer_grpc: bool = True
    collection_name: str = "video_embeddings"


class ElasticsearchConfig(BaseModel):
    host: str = "localhost"
    port: int = 9200
    user: str | None = None
    password: str | None = None
    use_ssl: bool = False
    verify_certs: bool = True
    request_timeout: int = 30
    index_name: str = "video_ocr_docs_dev"

    @property
    def url(self) -> str:
        scheme = "https" if self.use_ssl else "http"
        return f"{scheme}://{self.host}:{self.port}"


class ArangoCollectionsConfig(BaseModel):
    entities: str = "entities"
    events: str = "events"
    micro_events: str = "micro_events"
    communities: str = "communities"
    relations: str = "relations"


class ArangoIndexesConfig(BaseModel):
    semantic_dim: int = 768
    structural_dim: int = 128


class ArangoConfig(BaseModel):
    host: str = "http://localhost:8529"
    database: str = "video_kg"
    username: str = "root"
    password: str = ""
    graph_name: str = "video_knowledge_graph"
    view_name: str = "video_kg_search_view"
    collections: ArangoCollectionsConfig = Field(default_factory=ArangoCollectionsConfig)
    indexes: ArangoIndexesConfig = Field(default_factory=ArangoIndexesConfig)


class StorageConfig(BaseModel):
    postgres: PostgresConfig = Field(default_factory=PostgresConfig)
    minio: MinioConfig = Field(default_factory=MinioConfig)
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    elasticsearch: ElasticsearchConfig = Field(default_factory=ElasticsearchConfig)
    arangodb: ArangoConfig = Field(default_factory=ArangoConfig)


class QwenVLConfig(BaseModel):
    base_url: str = "http://localhost:8000"


class MMBertInfConfig(BaseModel):
    base_url: str = "http://localhost:8009"
    model_name: str = "mmbert"


class SpladeConfig(BaseModel):
    url: str = "http://localhost:8001"
    model_name: str = "splade"
    timeout: int = 30
    verbose: bool = False
    max_batch_size: int = 32


class InferenceConfig(BaseModel):
    qwenvl: QwenVLConfig = Field(default_factory=QwenVLConfig)
    mmbert: MMBertInfConfig = Field(default_factory=MMBertInfConfig)
    splade: SpladeConfig = Field(default_factory=SpladeConfig)


class CacheConfig(BaseModel):
    enabled: bool = True
    ttl: int = 1800
    dir: str | None = None


class MLflowConfig(BaseModel):
    """MLflow tracing configuration."""
    enabled: bool = False
    tracking_uri: str = "http://localhost:5000"
    experiment_name: str = "VideoDeepSearch"


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8080
    debug: bool = False
    cors_origins: list[str] = ["http://localhost:3000"]


class Settings(BaseModel):
    llm_provider: LLMProviderConfig
    storage: StorageConfig = Field(default_factory=StorageConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)


def load_settings(config_path: str | Path | None = None) -> Settings:
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent.parent / "config" / "settings.yaml"
    else:
        config_path = Path(config_path)

    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}
    else:
        config = {}

    return Settings(**config)


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = load_settings()
    return _settings


settings = get_settings()


__all__ = [
    "Settings",
    "settings",
    "get_settings",
    "load_settings",
]
