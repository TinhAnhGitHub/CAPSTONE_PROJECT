from llama_index.core.tools import FunctionTool
from functools import partial, wraps
import json
from collections import defaultdict
from typing import Callable, Annotated, get_origin, get_args, get_type_hints
import inspect
from agentic_ai.tools.clients.milvus.client import ImageMilvusClient, SegmentCaptionImageMilvusClient
from agentic_ai.tools.clients.postgre.client import PostgresClient
from agentic_ai.tools.clients.minio.client import StorageClient
from agentic_ai.tools.clients.external.encode_client import ExternalEncodeClient
from llama_index.core.llms import LLM
from agentic_ai.tools.schema.artifact import VideoInterface, ImageObjectInterface, SegmentObjectInterface

from llama_index.core.llms import  TextBlock, ImageBlock, VideoBlock
from llama_index.core.base.llms.types import ContentBlock


from .registry import tool_registry




class ToolOutputFormatter:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._formatters: dict[type, Callable] = {
            list[ImageObjectInterface]: self.format_image_interface,
            list[SegmentObjectInterface]: self.format_segment_interface,
            list[tuple[bytes, str]]: self.format_bytes_object,
            list[VideoInterface]: self.format_video_interface
            
        }
        self._initialized = True
    
    def get_formatter(self, return_type: type) -> Callable | None:
        if return_type in self._formatters:
            return self._formatters[return_type]        
        origin = get_origin(return_type)
        if origin is list:
            args = get_args(return_type)
            if args:
                list_type = list[args[0]]
                if list_type in self._formatters:
                    return self._formatters[list_type]
        
        return None
    
    @staticmethod
    def format_image_interface(
        list_images: list[ImageObjectInterface]
    ) -> ContentBlock:
        """Format list of ImageObjectInterface to structured JSON ContentBlock."""
        result_dict: dict[str, list[dict]] = defaultdict(list)
    
        for image in list_images:
            query = image.query if image.query else "No query related"
            image_query = image.reference_query_image.expr() if image.reference_query_image else "No reference image"

            if isinstance(query, list):
                query = ', '.join(query)

            result_dict[image.related_video_id].append({
                "frame_index": image.frame_index,
                "timestamp": image.timestamp,
                "caption": image.caption_info,
                "score": round(image.score, 4) if image.score else "No score",
                "minio_path": image.minio_path,
                "query_relation": f"Match for query: '{query}'",
                "reference_query_image": f"The match reference image: {image_query}"
            })
        
        readable_result = {
            "type": "visual_search_result",
            "summary": f"Retrieved {len(list_images)} visually similar frames across {len(result_dict)} videos.",
            "results": result_dict
        }
        
        return TextBlock(text=json.dumps(readable_result, indent=2))

    @staticmethod
    def format_segment_interface(
        list_segments: list[SegmentObjectInterface]
    ) -> ContentBlock:
        """Format list of SegmentObjectInterface to structured JSON ContentBlock."""
        result_dict: dict[str, list[dict]] = defaultdict(list)

        for segment in list_segments:
            query = segment.segment_caption_query if segment.segment_caption_query else "No query related"
            result_dict[segment.related_video_id].append({
                "start_frame_index": segment.start_frame_index,
                "end_frame_index": segment.end_frame_index,
                "start_time": segment.start_time,
                "end_time": segment.end_time,
                "caption": segment.caption_info,
                "score": round(segment.score, 4) if segment.score else "No score",
                "duration": f"{segment.start_time} → {segment.end_time}",
                "query_relation": f"Segment semantically related to query: '{query}'"
            })

        for video_id in result_dict:
            result_dict[video_id].sort(key=lambda x: x["score"], reverse=True)

        readable_result = {
            "type": "visual_segment_search_result",
            "summary": (
                f"Retrieved {len(list_segments)} relevant segments across "
                f"{len(result_dict)} videos based on semantic similarity to the query."
            ),
            "results": result_dict
        }

        return TextBlock(text=json.dumps(readable_result, indent=2))

    @staticmethod
    def format_bytes_object(
        byte_mime_type: list[tuple[bytes,str]]
    ) -> list[ContentBlock]:


        result: list[ContentBlock] = []
        for byte_object, mime_type  in byte_mime_type:
            if 'image' in mime_type:
                result.append(ImageBlock(image=byte_object, image_mimetype=mime_type))
            elif 'video' in mime_type:
                result.append(VideoBlock(video=byte_object, video_mimetype=mime_type))
            else:
                raise ValueError(f"Unsupported mime_type: {mime_type}. Must contain 'image' or 'video'.")
        return result

    @staticmethod
    def format_video_interface(
        list_videos: list[VideoInterface]
    )->list[ContentBlock]:
        result = []
        for video in list_videos:
            result.append(
                TextBlock(text=str(video))
            )
        return result



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
        self.formatter =  ToolOutputFormatter()
    
    def _validate_type_annotations(self, func: Callable) -> None:
        sig = inspect.signature(func)
        type_hints = get_type_hints(func)

        if 'return' not in type_hints:
            raise ValueError(
                f"Function '{func.__name__}' must have a return type annotation. "
                f"Add '-> ReturnType' to the function signature."
            )
    
        missing_annotations = []
        for param_name, param in sig.parameters.items():
            if param_name not in type_hints and param.annotation == inspect.Parameter.empty:
                missing_annotations.append(param_name)
        
        if missing_annotations:
            raise ValueError(
                f"Function '{func.__name__}' has parameters without type annotations: {missing_annotations}. "
                f"All parameters must have type annotations (use Annotated for descriptions)."
            )

    def _get_return_type(self, func: Callable)->type:
        type_hints = get_type_hints(func)
        return_type = type_hints['return']
        if get_origin(return_type) is Annotated:
            return_type = get_args(return_type)[0]
        return return_type


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
        return_type = self._get_return_type(metadata.func)
        formatter_func = self.formatter.get_formatter(return_type)

        @wraps(metadata.func)
        async def bound_func(*args, **kwargs): #enforce async function
            result = await partial_func(*args, **kwargs)
            if formatter_func:
                return formatter_func(result)
            return result
        
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




        



    


        

        



        
