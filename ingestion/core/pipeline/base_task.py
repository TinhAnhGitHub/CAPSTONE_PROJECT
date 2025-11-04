from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, TypeVar, Generic, AsyncIterator, Sequence
from loguru import logger
from pydantic import BaseModel, Field
from core.artifact.persist import ArtifactPersistentVisitor
from core.clients.base import BaseServiceClient, BaseMilvusClient



InputTask = TypeVar('InputTask', bound=BaseModel | Sequence[BaseModel] | Sequence[Sequence[BaseModel]])
OuputTask = TypeVar('OuputTask', bound=BaseModel | Sequence[BaseModel] | Sequence[Sequence[BaseModel]])


class BaseTask(Generic[InputTask, OuputTask], ABC):
    """
    This is the base class for ETL pipeline tasks with lifecycle hooks and artifact tracking.
    """
    def __init__(
        self,
        visitor: ArtifactPersistentVisitor,  
        **kwargs
    ):
        self.visitor = visitor
        self.kwargs = kwargs
    
    @abstractmethod
    def execute(self, input_data: Any, client: BaseServiceClient | BaseMilvusClient | None) -> AsyncIterator[Any]:
        raise NotImplementedError
    
    @abstractmethod
    async def preprocess(self, input_data: InputTask) -> Any:
        raise NotImplementedError
    
    @abstractmethod
    async def postprocess(self, output_data: Any) -> OuputTask:
        raise NotImplementedError
