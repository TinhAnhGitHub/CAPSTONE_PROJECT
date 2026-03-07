"""Application settings using Pydantic Settings for type-safe configuration."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal


_ENV_FILE = Path(__file__).parent.parent.parent.parent / '.env'
print(_ENV_FILE)

from dotenv import load_dotenv
load_dotenv(_ENV_FILE)

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class MinioSettings(BaseSettings):
    endpoint: str = Field(default="localhost:9000")
    access_key: str = Field(default="minioadmin123")
    secret_key: str = Field(default="minioadmin123")
    secure: bool = Field(default=False)
    bucket_name: str = Field(default="video-pipeline")

    model_config = SettingsConfigDict(
        env_prefix="MINIO_",
        env_file=_ENV_FILE,
        extra="ignore",
        case_sensitive=False
    )


class PostgresSettings(BaseSettings):
    """PostgreSQL configuration."""

    host: str = Field(default="localhost")
    port: int = Field(default=5432)
    database: str = Field(default="video_pipeline")
    user: str = Field(default="postgres")
    password: str = Field(default="postgres")

    model_config = SettingsConfigDict(
        env_prefix="POSTGRES_",
        env_file=_ENV_FILE,
        extra="ignore",
        case_sensitive=False
    )

    @property
    def connection_string(self) -> str:
        """Get PostgreSQL connection string."""
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


class QdrantSettings(BaseSettings):
    """Qdrant vector database configuration."""

    host: str = Field(default="localhost")
    port: int = Field(default=6333)
    api_key: str | None = Field(default=None)
    collection_name: str = Field(default="video_embeddings")

    model_config = SettingsConfigDict(
        env_prefix="QDRANT_",
        env_file=_ENV_FILE,
        extra="ignore",
        case_sensitive=False
    )


class DaskSettings(BaseSettings):
    """Dask cluster configuration."""

    n_workers: int = Field(default=1, ge=1)
    threads_per_worker: int = Field(default=4, ge=1)
    processes: bool = Field(default=True)
    scheduler_address: str | None = Field(default=None)

    model_config = SettingsConfigDict(
        env_prefix="DASK_",
        env_file=_ENV_FILE,
        extra="ignore",
        case_sensitive=False
    )

    def to_cluster_kwargs(self) -> dict[str, int | bool]:
        """Convert to Dask cluster_kwargs."""
        return {
            "n_workers": self.n_workers,
            "threads_per_worker": self.threads_per_worker,
            "processes": self.processes,
        }


class TaskConfigSettings(BaseSettings):
    """Global task defaults."""

    default_retries: int = Field(default=2, ge=0)
    default_retry_delay: int = Field(default=5, ge=0)
    default_timeout: int = Field(default=300, ge=1)

    model_config = SettingsConfigDict(
        env_prefix="TASK_",
        env_file=_ENV_FILE,
        extra="ignore",
        case_sensitive=False
    )


class TritonSettings(BaseSettings):
    """Triton Inference Server configuration."""

    url: str = Field(default="localhost:8001")
    timeout: int = Field(default=30)

    model_config = SettingsConfigDict(
        env_prefix="TRITON_",
        env_file=_ENV_FILE,
        extra="ignore",
        case_sensitive=False
    )


class AppSettings(BaseSettings):
    """Main application settings."""

    environment: Literal["dev", "staging", "prod"] = Field(default="dev")
    log_level: str = Field(default="INFO")
    debug: bool = Field(default=False)

    # Nested settings
    minio: MinioSettings = Field(default_factory=MinioSettings)
    postgres: PostgresSettings = Field(default_factory=PostgresSettings)
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    dask: DaskSettings = Field(default_factory=DaskSettings)
    tasks: TaskConfigSettings = Field(default_factory=TaskConfigSettings)
    triton: TritonSettings = Field(default_factory=TritonSettings)

    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False
    )

    @classmethod
    def load_with_yaml(cls, yaml_path: Path | None = None) -> AppSettings:
        """Load settings from YAML + environment variables.

        Environment variables take precedence over YAML values.

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            AppSettings instance with merged configuration
        """
        yaml_data = {}

        if yaml_path and yaml_path.exists():
            with open(yaml_path) as f:
                yaml_data = yaml.safe_load(f) or {}

        return cls(**yaml_data)


_settings: AppSettings | None = None


def get_settings() -> AppSettings:
    """Get or create settings singleton.

    Loads environment-specific YAML configuration based on APP_ENV.

    Returns:
        AppSettings singleton instance
    """
    global _settings
    if _settings is None:
        env = os.getenv("APP_ENV", "dev")
        config_dir = Path(__file__).parent / "environments"
        config_path = config_dir / f"{env}.yaml"

        _settings = AppSettings.load_with_yaml(config_path)

    return _settings


def reset_settings() -> None:
    """Reset settings singleton. Useful for testing."""
    global _settings
    _settings = None
