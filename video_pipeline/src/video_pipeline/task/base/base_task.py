from abc import ABC, abstractmethod
from pydantic import BaseModel, Field, field_validator
from uuid import UUID
from typing import (
    TypeVar,
    Generic,
    Any,
)
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import asyncio
import traceback
import time
import yaml

from prefect import get_run_logger
from prefect.artifacts import (
    create_markdown_artifact,
    create_table_artifact,
    create_progress_artifact,
    update_progress_artifact
)
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

class TaskExecutionContext(BaseModel):
    video_id: str
    run_id: str
    task_name: str
    start_time: datetime = Field(default_factory=datetime.now)

    progress_artifact_id: UUID | None  = Field(default=None)
    total_items: int = 0
    completed_items: int = 0

    preprocessing_time_ms: float = 0
    execution_time_ms: float = 0
    postprocessing_time_ms: float = 0

    errors: list[dict] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @property
    def total_time_ms(self) -> float:
        return self.preprocessing_time_ms + self.execution_time_ms + self.postprocessing_time_ms
    
    @property
    def progress_percent(self) -> float:
        if self.total_items == 0:
            return 0.0
        return (self.completed_items / self.total_items) * 100
    
    def record_error(self, error: Exception, item_id: str | None = None) -> None:
        self.errors.append({
            "item_id": item_id,
            "error_type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
            "timestamp": datetime.now().isoformat(),
        })
    
    def add_warning(self, message: str) -> None:
        self.warnings.append(f"[{datetime.now().isoformat()}] {message}")


class BaseTask(ABC, Generic[InputT, OutputT]):
    """
    Abstract base class for all pipeline tasks.

    LIFECYCLE:
    ----------
    1. preprocess(input) → preprocessed_data
    2. execute(preprocessed_data, client) → async iterator of results
    3. postprocess(result) → final_artifact

    The base class wraps these with:
    - Timing and logging
    - Progress tracking
    - Error handling and retries
    - Artifact creation
    """

    config: TaskConfig

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
    async def execute_single(
        self, 
        item: Any, 
        client: Any,
        context: TaskExecutionContext,
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
    
    @abstractmethod 
    def format_result(self, result: Any) -> str:
        """
        Transforms raw result into a string format for printing 
        """
        pass


    def get_item_id(self, item: Any) -> str:
        if hasattr(item, 'artifact_id'):
            return item.artifact_id
        if hasattr(item, 'id'):
            return item.id
        return str(id(item))
    
    async def on_item_success(
        self, 
        item: Any, 
        result: Any, 
        context: TaskExecutionContext
    ):
        pass

    async def on_item_failure(
        self, 
        item: Any, 
        error: Exception, 
        context: TaskExecutionContext
    ):
        pass

    async def on_task_complete(
        self, 
        results: list[OutputT], 
        context: TaskExecutionContext
    ):
        pass

    async def execute(
        self,
        preprocessed: list[Any],
        client: Any,
        context: TaskExecutionContext
    ):
        logger = get_run_logger()
        context.total_items = len(preprocessed)

        logger.info(
            f"[{self.config.name}] execute() — {context.total_items} item(s) to process | "
            f"video_id={context.video_id} run_id={context.run_id}"
        )

        if self.config.create_progress_artifact and context.total_items > 1:
            artifact_id = await create_progress_artifact(
                progress=0.0,
                description=f"Processing {self.config.name}: 0/{context.total_items}",
            ) #type:ignore
            context.progress_artifact_id = artifact_id

        start_time = time.perf_counter()

        for item in preprocessed:
            item_id = self.get_item_id(item)

            logger.info(
                f"[{self.config.name}] Processing item {context.completed_items + 1}/{context.total_items} "
                f"| item_id={item_id}"
            )
            item_start = time.perf_counter()

            try:
                result = await self.execute_single(item, client, context)
                item_elapsed_ms = (time.perf_counter() - item_start) * 1000
                context.completed_items += 1

                logger.info(
                    f"[{self.config.name}] Item {context.completed_items}/{context.total_items} done "
                    f"| item_id={item_id} elapsed={item_elapsed_ms:.0f}ms"
                )

                await self.on_item_success(item, result, context)

                if context.progress_artifact_id:
                    update_progress_artifact(
                        artifact_id=context.progress_artifact_id,
                        progress=context.progress_percent,
                    )

                yield result
            except Exception as e:
                item_elapsed_ms = (time.perf_counter() - item_start) * 1000
                context.record_error(e, item_id)
                await self.on_item_failure(item, e, context)

                logger.error(
                    f"[{self.config.name}] Item failed | item_id={item_id} "
                    f"elapsed={item_elapsed_ms:.0f}ms error={type(e).__name__}: {e}"
                )
                raise

        context.execution_time_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            f"[{self.config.name}] execute() complete | "
            f"{context.completed_items}/{context.total_items} succeeded | "
            f"{len(context.errors)} error(s) | total={context.execution_time_ms:.0f}ms"
        )

    

    async def create_summary_artifact(
        self,
        results: list[OutputT],
        context: TaskExecutionContext,
        
    ) -> None:
        if not self.config.create_summary_artifact:
            return
        
        success_count = len(results)
        error_count = len(context.errors)
        total = success_count + error_count
        success_rate = (success_count / total * 100) if total > 0 else 0



        error_section = ""
        if context.errors:
            error_rows = "\n".join([
                f"| {e['item_id'] or 'N/A'} | {e['error_type']} | {e['message'][:50]}... |"
                for e in context.errors  # Limit to 10 errors
            ])
            error_section = f"""
            ## Errors ({error_count})

            | Item | Error Type | Message |
            |------|------------|---------|
            {error_rows}
            """
        
        warning_section = ""
        if context.warnings:
            warning_list = "\n".join([f"- {w}" for w in context.warnings[:10]])
            warning_section = f"""
            ## Warnings ({len(context.warnings)})

            {warning_list}
            """

        import random
        sample_results = random.sample(results, min(len(results), 10))

        string_results = []
        for res in sample_results:
            str_format = self.format_result(res)
            string_results.append(str_format)
        
        string_results = '\n\n'.join(string_results)

            
        markdown = f"""# {self.config.name} Summary

        **Video ID:** `{context.video_id}`  
        **Run ID:** `{context.run_id}`  
        **Stage:** {self.config.stage}

        ## Results

        | Metric | Value |
        |--------|-------|
        | Total Items | {total} |
        | Successful | {success_count} |
        | Failed | {error_count} |
        | Success Rate | {success_rate:.1f}% |

        ## Timing

        | Phase | Duration |
        |-------|----------|
        | Preprocessing | {context.preprocessing_time_ms:.0f} ms |
        | Execution | {context.execution_time_ms:.0f} ms |
        | Postprocessing | {context.postprocessing_time_ms:.0f} ms |
        | **Total** | **{context.total_time_ms:.0f} ms** |

        ## Result sample (10 of them)
        {string_results}


        ## Additional Info
        {error_section}
        {warning_section}
        """

    
        await create_markdown_artifact(
            key=f"{self.config.name.lower().replace(' ', '-')}-{context.video_id}",
            markdown=markdown,
            description=f"{self.config.name} execution summary",
        ) #type:ignore
        

    async def create_statistics_artifact(
        self,
        results: list[OutputT],
        context: TaskExecutionContext,
    ) -> None:        
        stats = [
            {"metric": "Video ID", "value": context.video_id},
            {"metric": "Task", "value": self.config.name},
            {"metric": "Stage", "value": self.config.stage},
            {"metric": "Items Processed", "value": str(len(results))},
            {"metric": "Errors", "value": str(len(context.errors))},
            {"metric": "Total Time (ms)", "value": f"{context.total_time_ms:.0f}"},
            {"metric": "Avg Time/Item (ms)", "value": f"{context.total_time_ms / max(len(results), 1):.0f}"},
        ]
        
        await create_table_artifact(
            key=f"stats-{self.config.name.lower().replace(' ', '-')}",
            table=stats,
            description=f"Statistics for {self.config.name}",
        )#type:ignore
    

    


