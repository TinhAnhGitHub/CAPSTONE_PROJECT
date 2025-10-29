from llama_index.core.tools import FunctionTool
from functools import partial, wraps
from typing import Callable
import inspect
from agentic_ai.tools.clients.milvus.client import ImageMilvusClient, SegmentCaptionImageMilvusClient
from agentic_ai.tools.clients.postgre.client import PostgresClient
from agentic_ai.tools.clients.minio.client import StorageClient
from agentic_ai.tools.clients.external.encode_client import ExternalEncodeClient
from llama_index.core.llms import LLM


from .registry import tool_registry



class ToolFactory:
    """
    Factory -> bind dependencies to registered tools
    Use registry + explicit dependency binding for control
    """
    def __init__(
        self,
        image_milvus_client: ImageMilvusClient,
        segment_milvus_client: SegmentCaptionImageMilvusClient,
        postgres_client: PostgresClient,
        minio_client: StorageClient,
        external_client: ExternalEncodeClient,
        llm_as_tools: LLM,
    ):
        self.dependency_map = {
            'visual_milvus_client': image_milvus_client,
            'segment_milvus_client': segment_milvus_client,
            'postgres_client': postgres_client,
            'minio_client': minio_client,
            'external_client': external_client,
            'llm_as_tools': llm_as_tools
        }

        self._tool_cache: dict[str, Callable] = {}
    
    def _bind_tool(self, tool_name:str) -> Callable | None:
        if tool_name in self._tool_cache:
            return self._tool_cache[tool_name]

        metadata = tool_registry.get(tool_name)
        if not metadata:
            return None
        
        bound_kwargs = {}
        missing_deps = []

        for dep_name in metadata.dependencies:
            if dep_name in self.dependency_map:
                bound_kwargs[dep_name] = self.dependency_map[dep_name]
            else:
                missing_deps.append(dep_name)
        
        if missing_deps:
            raise ValueError(
                f"Tool '{tool_name}' requires dependencies {missing_deps} "
                f"but they were not provided to ToolFactory"
            )
        partial_func= partial(metadata.func, **bound_kwargs)
        @wraps(metadata.func)
        def bound_func(*args, **kwargs):
            return partial_func(*args, **kwargs)
        bound_func.__name__ = metadata.name
        bound_func.__doc__ = metadata.docstring or metadata.func.__doc__

        self._tool_cache[tool_name] = bound_func
        return bound_func
    
    def get_tool(self, name: str) -> Callable | None:
        return self._bind_tool(name)

    def get_tools_by_category(self, category:str)->dict[str, Callable | None]:
        tool_names = tool_registry.list_by_category(category)
        return {
            name: self._bind_tool(name)
            for name in tool_names
            if self._bind_tool(name) is not None
        }
    
    def get_tools_by_tags(self, tags: list[str]) -> dict[str, Callable | None]:
        """Get tools with specific tags"""
        tool_names = tool_registry.list_by_tags(tags)
        return {
            name: self._bind_tool(name)
            for name in tool_names
            if self._bind_tool(name) is not None
        }
    
    def get_tools_by_names(self, names: list[str]) -> dict[str, Callable | None]:
        return {
            name: self._bind_tool(name)
            for name in names
            if self._bind_tool(name) is not None
        }
    

    def get_all_tools(self) -> dict[str, Callable | None]:
        return {
            name: self._bind_tool(name)
            for name in tool_registry.list_all()
            if self._bind_tool(name) is not None
        }
    
    
    

    


        

        



        
