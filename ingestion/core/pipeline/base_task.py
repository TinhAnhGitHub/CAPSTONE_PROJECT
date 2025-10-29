from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, TypeVar, Generic, AsyncIterator, Sequence
from loguru import logger
from pydantic import BaseModel, Field
from core.artifact.persist import ArtifactPersistentVisitor
from core.clients.base import BaseServiceClient, BaseMilvusClient



InputTask = TypeVar('InputTask', bound=BaseModel | Sequence[BaseModel] | Sequence[Sequence[BaseModel]])
OuputTask = TypeVar('OuputTask', bound=BaseModel | Sequence[BaseModel] | Sequence[Sequence[BaseModel]])
TaskConfig = TypeVar('TaskConfig', bound=BaseModel)

class BaseTask(Generic[InputTask, OuputTask, TaskConfig], ABC):
    """
    This is the base class for ETL pipeline tasks with lifecycle hooks and artifact tracking.
    """
    def __init__(
        self,
        name: str,
        visitor: ArtifactPersistentVisitor,  
        config: TaskConfig

    ):
        self.name = name
        self.visitor = visitor
        self.config = config
    
    @abstractmethod
    def execute(self, input_data: Any, client: BaseServiceClient | BaseMilvusClient | None) -> AsyncIterator[Any]:
        """
        Core task logic; subclasses should implement this as an async generator.
        """
        ...
    
    @abstractmethod
    async def preprocess(self, input_data: InputTask) -> Any:
        """
        This is the preprocess for the input data
        """
        raise NotImplementedError
    
    @abstractmethod
    async def postprocess(self, output_data: Any) -> OuputTask:
        """
        This is post processing
        """
        raise NotImplementedError

    
    # @staticmethod
    # @abstractmethod
    # async def on_failure_batch(state: State, task_run: TaskRun, parameters: dict[str, Any]) -> None:
    #     pass 

    # @staticmethod
    # @abstractmethod
    # async def on_completion_batch(state: State, task_run: TaskRun, parameters: dict[str, Any]) -> None:
    #     pass 
    
    # @staticmethod
    # @abstractmethod
    # async def on_completion(state: State, task_run: TaskRun, parameters: dict[str, Any]) -> None:
    #     pass

    # @staticmethod
    # @abstractmethod
    # async def on_failure(state: State, task_run: TaskRun, parameters: dict[str, Any]) -> None:
    #     pass
    


    # @abstractmethod 
    # def create_task_func(self) -> Callable:
    #     """
    #     Create the template prefect task.
    #     Handling flow management, event and artifact tracking
    #     Return a callable task(a Prefect feature) that can be used inside flow
    #     """

    #     raise NotImplementedError

    
    






    


