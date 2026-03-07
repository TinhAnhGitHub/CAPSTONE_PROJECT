from abc import ABC, abstractmethod
from pydantic import BaseModel, Field, field_validator
from typing import (
    TypeVar,
    Generic,
    Any,
)
from pathlib import Path
import yaml

from prefect import get_run_logger
from video_pipeline.core.storage.pg_tracker import ArtifactPersistentVisitor
from video_pipeline.core.client.storage.minio import MinioStorageClient


InputT = TypeVar('InputT')
OutputT = TypeVar('OutputT')


class TaskConfig(BaseModel):
    """Configuration for a Prefect task.

    Can be loaded from YAML and converted to @task decorator kwargs.
    """

    name: str
    description: str
    stage: str
    tags: list[str] = Field(default_factory=list)
    retries: int = Field(default=2, ge=0)
    retry_delay_seconds: int = Field(default=5, ge=0)
    timeout_seconds: int = Field(default=30, ge=1)
    cache_enabled: bool = Field(default=False)
    create_summary_artifact: bool = Field(default=True)
    create_progress_artifact: bool = Field(default=True)

    additional_kwargs: dict[str, Any] = Field(default_factory=dict)

    @field_validator('tags')
    @classmethod
    def validate_tags(cls, v: list[str]) -> list[str]:
        return [tag.strip() for tag in v if tag.strip()]

    @classmethod
    def from_yaml(cls, task_name: str, yaml_path: Path | None = None) -> "TaskConfig":
        """Load task config from YAML file.

        Args:
            task_name: Key name in the tasks.yaml file
            yaml_path: Optional custom path to YAML file

        Returns:
            TaskConfig instance

        Raises:
            ValueError: If task_name not found in YAML
            FileNotFoundError: If YAML file doesn't exist
        """
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
            raise ValueError(
                f"Task config '{task_name}' not found. Available: {available}"
            )

        return cls(**task_data)

    def to_task_kwargs(self) -> dict[str, Any]:
        from prefect.cache_policies import INPUTS, NO_CACHE
        return {
            "name": self.name,
            "description": self.description,
            "tags": self.tags,
            "retries": self.retries,
            "retry_delay_seconds": self.retry_delay_seconds,
            "timeout_seconds": self.timeout_seconds,
            "cache_policy": INPUTS if self.cache_enabled else NO_CACHE,
        }

class BaseTask(ABC, Generic[InputT, OutputT]):
    def __init__(
        self,
        artifact_visitor: ArtifactPersistentVisitor,
        minio_client: MinioStorageClient,
        **kwargs
    ):
        self.artifact_visitor = artifact_visitor
        self.minio_client = minio_client
        self.kwargs = kwargs
    
    
    @abstractmethod
    async def preprocess(self, input_data: InputT) -> Any:
        """
        Transform raw input into a format ready for execution.
        
        This is where you:
        - Validate input
        - Load additional data
        - Split into batches
        - Prepare API payloads
        
        Returns:
            Preprocessed data ready for execute()
        """
        pass
    
    @abstractmethod
    async def execute(
        self, 
        preprocessed: Any, 
        client: Any,
    ) -> Any:
        """
        Process a single item.
        
        This is the core logic of your task. The base class handles:
        - Iteration over items
        - Progress updates
        - Error handling
        
        Args:
            item: A single preprocessed item
            client: The service client (ASR, LLM, etc.)
            context: Execution context for progress/error tracking
            
        Returns:
            Raw result (will be passed to postprocess)
        """
        pass
    
    @abstractmethod
    async def postprocess(self, result: Any) -> OutputT:
        """
        Transform raw result into the final artifact.
        
        This is where you:
        - Create artifact objects
        - Persist to storage
        - Validate output
        
        Returns:
            Final output artifact
        """
        pass

    @staticmethod
    @abstractmethod
    async def summary_artifact(final_result: OutputT) -> None :
        """
        Implement the final statistic Prefect artifact summary. 
        Using these: links , Markdown, images, tables
        """
        pass

    def get_item_id(self, item: Any) -> str:
        if hasattr(item, 'artifact_id'):
            return item.artifact_id
        if hasattr(item, 'id'):
            return item.id
        return str(id(item))
    
    async def on_task_complete(
        self, 
        results: list[OutputT], 
    ):
        pass

    async def on_task_failed(
        self, 
        exception: Exception
    ):
        pass
    
    async def execute_template(
        self, 
        input_data: InputT, 
        client: Any
    ) -> OutputT:
        """
        The Template Method orchestrating the task execution lifecycle.
        
        Args:
            input_data: The raw input to be preprocessed.
            client: The service client passed down to the execute method.
            
        Returns:
            A list of successfully processed and post-processed artifacts.
        """
        
        logger = get_run_logger()    
        results: list[OutputT] = []
        
        try:
            items_to_process = await self.preprocess(input_data)
            raw_result = await self.execute(items_to_process, client)
            final_artifact = await self.postprocess(raw_result)
               
        except Exception as critical_error:
            logger.error(f"Task failed critically during preprocessing: {critical_error}", exc_info=True)
            await self.on_task_failed(critical_error)
            raise
        await self.on_task_complete(results)
        return final_artifact

    

    


