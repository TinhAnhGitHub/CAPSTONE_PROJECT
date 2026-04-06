from abc import ABC, abstractmethod
from datetime import timedelta
from pathlib import Path
from typing import Any, Generic, TypeVar

import yaml
from prefect import get_run_logger
from prefect.cache_policies import INPUTS, NO_CACHE
from pydantic import BaseModel, Field, field_validator

from video_pipeline.core.client.storage.minio import MinioStorageClient
from video_pipeline.core.storage.pg_tracker import ArtifactPersistentVisitor
from video_pipeline.task.base.cache_keys import CACHE_KEY_FUNCTIONS

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class TaskConfig(BaseModel):
    """Configuration for a Prefect task."""

    name: str
    description: str
    stage: str
    tags: list[str] = Field(default_factory=list)
    retries: int = Field(default=2, ge=0)
    retry_delay_seconds: int = Field(default=5, ge=0)
    timeout_seconds: int = Field(default=30, ge=1)
    cache_enabled: bool = Field(default=False)
    cache_expiration_seconds: int | None = Field(default=None, ge=1)
    cache_key_fn: str | None = Field(default=None)
    create_summary_artifact: bool = Field(default=True)
    create_progress_artifact: bool = Field(default=True)
    additional_kwargs: dict[str, Any] = Field(default_factory=dict)

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, v: list[str]) -> list[str]:
        return [tag.strip() for tag in v if tag.strip()]

    @classmethod
    def from_yaml(cls, task_name: str, yaml_path: Path | None = None) -> "TaskConfig":
        """Load task config from YAML file."""
        if yaml_path is None:
            config_dir = Path(__file__).parent.parent.parent / "config"
            yaml_path = config_dir / "tasks.yaml"

        if not yaml_path.exists():
            raise FileNotFoundError(f"Task config file not found: {yaml_path}")

        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        if not data or "tasks" not in data:
            raise ValueError(f"Invalid YAML structure in {yaml_path}")

        task_data = data["tasks"].get(task_name)
        if not task_data:
            available = ", ".join(data["tasks"].keys())
            raise ValueError(f"Task config '{task_name}' not found. Available: {available}")

        return cls(**task_data)

    def to_task_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "tags": self.tags,
            "retries": self.retries,
            "retry_delay_seconds": self.retry_delay_seconds,
            "timeout_seconds": self.timeout_seconds,
        }

        if self.cache_enabled:
            if self.cache_key_fn and self.cache_key_fn in CACHE_KEY_FUNCTIONS:
                kwargs["cache_key_fn"] = CACHE_KEY_FUNCTIONS[self.cache_key_fn]
            else:
                kwargs["cache_policy"] = INPUTS

            if self.cache_expiration_seconds:
                kwargs["cache_expiration"] = timedelta(seconds=self.cache_expiration_seconds)
        else:
            kwargs["cache_policy"] = NO_CACHE

        return kwargs


class BaseTask(ABC, Generic[InputT, OutputT]):
    def __init__(
        self,
        artifact_visitor: ArtifactPersistentVisitor,
        minio_client: MinioStorageClient,
        **kwargs,
    ):
        self.artifact_visitor = artifact_visitor
        self.minio_client = minio_client
        self.kwargs = kwargs

    @abstractmethod
    async def preprocess(self, input_data: InputT) -> Any:
        """Transform raw input into a format ready for execution."""
        pass

    @abstractmethod
    async def execute(self, preprocessed: Any, client: Any) -> Any:
        """Process a single item."""
        pass

    @abstractmethod
    async def postprocess(self, result: Any) -> OutputT:
        """Transform raw result into the final artifact."""
        pass

    @staticmethod
    @abstractmethod
    async def summary_artifact(final_result: OutputT) -> None:
        """Create Prefect artifact summary."""
        pass

    def get_item_id(self, item: Any) -> str:
        if hasattr(item, "artifact_id"):
            return item.artifact_id
        if hasattr(item, "id"):
            return item.id
        return str(id(item))

    async def on_task_complete(
        self,
        results: list[OutputT],
    ):
        pass

    async def on_task_failed(self, exception: Exception):
        pass

    async def execute_template(self, input_data: InputT, client: Any) -> OutputT:
        """Orchestrate the task execution lifecycle."""
        logger = get_run_logger()
        results: list[OutputT] = []

        try:
            items_to_process = await self.preprocess(input_data)
            raw_result = await self.execute(items_to_process, client)
            final_artifact = await self.postprocess(raw_result)

        except Exception as critical_error:
            logger.error(
                f"Task failed critically during preprocessing: {critical_error}", exc_info=True
            )
            await self.on_task_failed(critical_error)
            raise

        await self.on_task_complete(results)
        return final_artifact
